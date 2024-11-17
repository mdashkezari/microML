"""
Author: Mohammad Dehghani Ashkezari <mdehghan@uw.edu>

Add environmental data to the raw datasets.
"""
import os
import logging
import glob
import time
import concurrent.futures
from copy import deepcopy
import pandas as pd
import datetime
from dateutil.parser import parse
import pycmap
from tqdm.auto import tqdm
from microML.settings import COLOCALIZED_DIR, RAW_DATA_DIR
from microML.tokens import API_KEY
from microML.common import environmental_datasets


logger = logging.getLogger("root_logger")


class Augment():
    def __init__(self) -> None:
        self.api = pycmap.API(API_KEY)
        self.augment_dir = COLOCALIZED_DIR
        os.makedirs(self.augment_dir, exist_ok=True)

    @staticmethod
    def raw_datasets_list(collect_dir: str):
        """
        Returns a list of path to raw datasets.

        Parameters
        ----------
        collect_dir: str
            Path to the directory where the raw dataset files are stored.

        Returns
        ----------
        list[str]
            List of csv file paths.
        """
        return glob.glob(f"{collect_dir}*.csv")

    @staticmethod
    def add_env_columns(df: pd.DataFrame, envs: dict) -> pd.DataFrame:
        """
        Adds new empty placeholder columns to the dataframe `df` for each environmental variable.
        """
        for env in envs.values():
            for v in env.get("variables"):
                if v not in df.columns:
                    df[v] = None
        return df

    def add_env_temporal_coverage(self, envs: dict) -> dict:
        """
        Adds new entries to the envs dictionary indicating the temporal coverage of each
        environmental variable dataset.
        """
        pbar = tqdm(envs.items())
        for table, env in pbar:
            pbar.set_description(f"Get temporal coverage for {table}")
            df = self.api.query(f"SELECT MIN([time]) startTime, MAX([time]) endTime FROM {table}")
            if len(df) > 0:
                envs[table]["startTime"] = df.loc[0, "startTime"]
                envs[table]["endTime"] = df.loc[0, "endTime"]
        return envs

    def colocalize(self,
                   df: pd.DataFrame,
                   envs: dict,
                   cyanoFile: str,
                   rowCount: int) -> pd.DataFrame:
        """
        Takes a single-row dataframe containing cyano observations and colocalizes with the
        environmental variables included in the `envs` argument. The tolerance parametrs 
        are also included in the `envs` argument.

        No match is made between a surface target dataset (such as satellite) and observations
        deeper than `MAX_SURFACE_DEPTH`.

        Parameters
        ----------
        df: pandas.DataFrame
            A single-row dataframe from raw dataset. The environmental data will be added
            to this row.
        envs: dict
            A dictionary containing metadata about the environmental datasets.
        cyanoFile: str
            Path to the raw dataset.
        rowCount: int
            Total number of rows in the raw dataset.

        Returns
        ----------
        pandas.DataFrame
            The input single-row dataframe joined with new columns containing environmental data.
        """
        def get_month(dt):
            return parse(dt).month

        def shift_dt(dt, delta):
            delta = float(delta)
            dt = parse(dt)
            dt += datetime.timedelta(days=delta)
            return dt.strftime("%Y-%m-%d %H:%M:%S")

        def in_time_window(sourceDT, targetMinDT, targetMaxDT):
            targetMinDT = targetMinDT.split(".000Z")[0]
            targetMaxDT = targetMaxDT.split(".000Z")[0]
            return not (
                        parse(sourceDT) < parse(targetMinDT) or
                        parse(sourceDT) > parse(targetMaxDT)
                        )

        def make_query(table, env, t, lat, lon, depth):
            variables = env["variables"]
            timeTolerance = env["tolerances"][0]
            latTolerance = env["tolerances"][1]
            lonTolerance = env["tolerances"][2]
            depthTolerance = env["tolerances"][3]
            hasDepth = env["hasDepth"]
            isClimatology = env["isClimatology"]
            inTimeRange = True
            if not isClimatology:
                startTime = env["startTime"]
                endTime = env["endTime"]
                inTimeRange = in_time_window(t, startTime, endTime)
            selectClause = "SELECT " + ", ".join([f"AVG({v}) {v}" for v in variables]) + " FROM " + table
            timeClause = f" WHERE [time] BETWEEN '{shift_dt(t, -timeTolerance)}' AND '{shift_dt(t, timeTolerance)}' "
            if not inTimeRange or isClimatology:
                timeClause = f" WHERE [month]={get_month(t)} "
            latClause = f" AND lat BETWEEN {lat-latTolerance} AND {lat+latTolerance} "
            lonClause = f" AND lon BETWEEN {lon-lonTolerance} AND {lon+lonTolerance} "
            depthClause = f" AND depth BETWEEN {depth-depthTolerance} AND {depth+depthTolerance} "
            if not hasDepth:
                depthClause = ""
            return selectClause + timeClause + latClause + lonClause + depthClause
        try:
            if len(df) != 1:
                msg = f"Invalid dataframe input.\nExpected a single row dataframe but received {len(df)} rows."
                logger.exception(msg)
            MAX_SURFACE_DEPTH = 10
            rowIndex = df.index.values[0]
            df.reset_index(drop=True, inplace=True)
            t = df.iloc[0]["time"]
            lat = df.iloc[0]["lat"]
            lon = df.iloc[0]["lon"]
            depth = 0
            if "depth" in df.columns:
                depth = df.iloc[0]["depth"]
            for table, env in envs.items():
                # print(f"{rowIndex} / {rowCount-1}\n\t{datetime.datetime.now()}: Colocalizing {table} with {cyanoFile} ...")
                time.sleep(0.1)
                if env["hasDepth"] or depth <= MAX_SURFACE_DEPTH:
                    query = make_query(table, env, t, lat, lon, depth)
                    matchedEnv = self.api.query(query, servers=["rossby"])
                    if len(matchedEnv) > 0:
                        for v in env["variables"]:
                            df.at[0, v] = matchedEnv.iloc[0][v]
        except Exception as e:
            matchedEnv.to_csv(f"error_colocalize_{os.path.basename(cyanoFile)}_{rowIndex}.csv", index=False)
            # msg = (f"Error in colocalize:\n{str(e)}\n{'*'*50}")
            # logger.exception(msg)
            # return df
        return df

    def add_environment(self):
        """
        Loop through the raw datasets and add environmental data to them.
        Save the new augmented datasets to the `self.augment_dir`.
        """
        raw_datasets = self.raw_datasets_list(RAW_DATA_DIR)
        os.makedirs(self.augment_dir, exist_ok=True)
        envs = environmental_datasets()
        envs = self.add_env_temporal_coverage(envs)
        try:
            for ds in raw_datasets:
                df = pd.read_csv(ds)
                df = self.add_env_columns(df, envs)
                dfs = [df.loc[i].to_frame().T for i in range(len(df))]
                colocalized_list, columns = [], []
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futureObjs = executor.map(self.colocalize,
                                              dfs,
                                              [envs] * len(dfs),
                                              [ds] * len(dfs),
                                              [len(dfs)] * len(dfs)
                                              )
                    pbar = tqdm(futureObjs, leave=True, total=len(dfs), desc=f"Augmenting {os.path.basename(ds)}")
                    for _, fo in enumerate(pbar):
                        if len(colocalized_list) < 1:
                            columns = list(fo.columns)
                        colocalized_list.append(fo.values.tolist()[0])

                    colocalized_df = pd.DataFrame(colocalized_list, columns=columns)
                colocalized_df.to_csv(f"{self.augment_dir}{os.path.basename(ds)}", index=False)
        except Exception as e:
            msg = (f"Error while adding env to {os.path.basename(ds)}.\n"
                   f"Error Message: {str(e)}\n"
                   f"fo.values.tolist()[0]: {fo.values.tolist()[0]}"
                   f"\n{'*'*50}")
            logger.exception(msg)
