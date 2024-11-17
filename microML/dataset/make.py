"""
Author: Mohammad Dehghani Ashkezari <mdehghan@uw.edu>

Retrieve the raw datasets, augment them with environmental data, and compile them into a single dataset.
"""

import pandas as pd
from microML.dataset.collect import Collect
from microML.dataset.augment import Augment
from microML.dataset.compile import Compile


def make_dataset(collect: bool,
                 augment: bool,
                 compile: bool) -> pd.DataFrame:
    """
    Download all raw datasets from Simons CMAP database, join them with environmental
    parameters, and finally compile them all into a single csv file. The compiled file
    is stored on local disk (`COMPILE_DIR`).

    Parameters
    ----------
    collect: bool
        If True, all raw datasets defined in `common.cyano_datasets()` are downloaded
        into the `Collect.collect_dir`.
    augment: bool
        If True, the downloaded datasets in the `Collect.collect_dir` are augmented
        with environmental parameters defined in `common.environmental_datasets()`.
    compile: bool
        If True, all augmented datasets (raw data + environmental data) are compile
        into a single csv file and is stored on local dist.

    Parameters
    ----------
    pandas.DataFRame
        Compiled dataset in form of a pandas dataframe.
    """
    if collect:
        Collect().retrieve_all_datasets()
    if augment:
        Augment().add_environment()
    if compile:
        Compile().make_single_dataset()
    return


if __name__ == "__main__":
    make_dataset(collect=False, augment=False, compile=True)
