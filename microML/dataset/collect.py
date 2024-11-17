"""
Author: Mohammad Dehghani Ashkezari <mdehghan@uw.edu>

Retrieve Marine microbial measurements from Simons CMAP and store them on local disk.
"""
import os
import logging
import pandas as pd
import pycmap
from tqdm.auto import tqdm
from microML.settings import DEPTH1, DEPTH2, RAW_DATA_DIR
from microML.tokens import API_KEY
from microML.common import cyano_datasets


logger = logging.getLogger("root_logger")


class Collect():
    def __init__(self) -> None:
        self.api = pycmap.API(API_KEY)
        assert len(self.api.query("select top 2 * from tblMakes")) == 2, "CMAP API call failure."
        self.collect_dir = RAW_DATA_DIR
        os.makedirs(self.collect_dir, exist_ok=True)

    def single_retrieve(self,
                        dataset: tuple,
                        depth1: float,
                        depth2: float) -> pd.DataFrame:
        """
        Retrieve a dataset, rename the measurement columns using the passed aliases
        (`dataset[2]`), and transforms them using the passed unit conversion coefficients
        (`dataset[3]`).

        Parameters
        ----------
        dataset: tuple
            Tuple object specifying the dataset to be retrieved.
        depth1: float
            Depth filter start range.
        depth2: float
            Depth filter end range.

        Raises
        ----------
        ValueError:
            If the length of lists within the `dataset` tuple are not equal.

        Returns
        ----------
        pandas.DataFrame:
            The retrieved dataset in form of a Pandas DataFrame object.
        """
        if not len(dataset[1]) == len(dataset[2]) == len(dataset[3]):
            msg = "The length of column name, alias, and unit conversion lists must be identical."
            logging.exception(msg)
            raise ValueError(msg)
        table, fields = dataset[0], ", ".join(dataset[1])
        has_depth = self.api.has_field(table, "depth")
        at_least_one_field = "(" + " OR ".join([f"[{f}] IS NOT NULL" for f in dataset[1]]) + ")"
        if has_depth:
            fields = f" [time], lat, lon, depth, {fields} "
            where_clause = f" WHERE depth BETWEEN {depth1} AND {depth2} AND {at_least_one_field}"
        else:
            fields = f" [time], lat, lon, {fields} "
            where_clause = f" WHERE {at_least_one_field}"
        query = f"SELECT {'*'} FROM {table} {where_clause}"
        data = self.api.query(query)
        data = data.rename(columns={dataset[1][i]: dataset[2][i] for i in range(len(dataset[1]))},
                           errors="raise")
        for ind, alias in enumerate(dataset[2]):
            if dataset[3][ind] is not None:
                data[alias] *= dataset[3][ind]
        return data

    def retrieve_all_datasets(self):
        """
        Iterates through the list of datasets containing measurements of cyanobacteria.
        The measurements are retrieved and stored in individual csv files on local disk.
        """
        cyanos = cyano_datasets()
        pbar = tqdm(cyanos, desc="Collecting datasets from Simons CMAP database", leave=False)
        for dataset in pbar:
            pbar.set_description(f"Downloading {dataset[0]}")
            data = self.single_retrieve(dataset, DEPTH1, DEPTH2)
            data.to_csv(f"{self.collect_dir}{dataset[0]}.csv", index=False)
        logger.info(f"{len(cyanos)} datasets downloaded.")
