"""
Author: Mohammad Dehghani Ashkezari <mdehghan@uw.edu>

Compile all of the augmented datasets into a single csv file.
"""
import os
import warnings
import logging
import glob
import pandas as pd
from tqdm.auto import tqdm
from microML.settings import PROC, SYNC, PICO, HETB, COLOCALIZED_DIR, COMPILED_DIR
from microML.tokens import API_KEY
from microML.common import env_vars


logger = logging.getLogger("root_logger")


class Compile():
    def __init__(self) -> None:
        self.compile_dir = COMPILED_DIR
        os.makedirs(self.compile_dir, exist_ok=True)

    @staticmethod
    def unify(augmented_dataset: str) -> pd.DataFrame:
        """
        Takes a colocalized (augmented) dataset filepath and ensures that it will have identical
        columns (schema) as other colocalized files.

        Parameters
        ----------
        augmented_dataset: str
            Path to the augmented dataset (augmented dataset: raw dataset joined with environmental data).

        Returns
        ----------
        pandas.DataFrame:
            The input datasets with rearranged and renamed columns so that all augmented datasets
            have identical schema.
        """
        def insert_column(df: pd.DataFrame,
                          col_title: str,
                          col_index: int,
                          fill_value: float) -> pd.DataFrame:
            """
            Inserts a new column `col_title`, to the dataframe `df` at location `col_index`
            with initial value `fill_value`.
            """
            if col_title not in df.columns:
                df.insert(col_index, col_title, fill_value, True)
            return df

        # output dataframe columns:
        # time | lat | lon | depth | table | cruise | <PROC> | <SYNC> | <PICO> | <HETB> | env_var1 | ... | env_var_n     
        df = pd.read_csv(augmented_dataset)
        table = os.path.splitext(os.path.basename(augmented_dataset))[0]

        df = insert_column(df, "depth", 3, 0)
        df = insert_column(df, "table", 4, table)
        df = insert_column(df, "cruise", 5, None)
        df = insert_column(df, PROC, 6, None)
        df = insert_column(df, SYNC, 7, None)
        df = insert_column(df, PICO, 8, None)
        df = insert_column(df, HETB, 9, None)

        columns = ["time", "lat", "lon", "depth", "table", "cruise", PROC, SYNC, PICO, HETB] + env_vars()
        # if list(df.columns) != columns:
        #     msg = f"Invalid columns at {augmented_dataset}:\n{df.columns}"
        #     logger.error(msg)
        df = df[columns]
        return df

    def make_single_dataset(self):
        """
        Iterates through the list of augmented datasets and compile them into a single csv file.
        The compiled file is stored in the "COMPILED_DIR" as a csv file.
        """
        files = glob.glob(f"{COLOCALIZED_DIR}*.csv")
        df_compiled = pd.DataFrame({})
        pbar = tqdm(files, leave=False)
        for f in pbar:
            pbar.set_description(f"Compiling {os.path.basename(f)}")
            data = self.unify(f)
            if len(df_compiled) < 1:
                df_compiled = data
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    df_compiled = pd.concat([df_compiled, data], ignore_index=True)
        df_compiled.to_csv(f"{COMPILED_DIR}compiled.csv", index=False)
        return
