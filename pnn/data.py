"""
Functions for reading the (split) input data.
"""
from pathlib import Path

import pandas as pd

from . import constants as c

rename_org = {"aph_443": "org_aph_443", "aNAP_443": "org_anap_443", "aCDOM_443": "org_acdom_443"}
def read_all_data(folder: Path | str=c.data_path) -> tuple[pd.DataFrame]:
    """
    Read all split data from a given folder into a number of DataFrames.
    The output cannot be a single DataFrame because of differing indices.
    """
    train_set_random = pd.read_csv(folder/"random_df_train_org.csv").rename(columns=rename_org)
    test_set_random = pd.read_csv(folder/"random_df_test_org.csv").rename(columns=rename_org)

    train_set_wd = pd.read_csv(folder/"wd_train_set_org.csv").rename(columns=rename_org)
    test_set_wd = pd.read_csv(folder/"wd_test_set_org.csv").rename(columns=rename_org)

    train_set_ood = pd.read_csv(folder/"ood_train_set_2.csv")
    test_set_ood = pd.read_csv(folder/"ood_test_set_2.csv")

    return train_set_random, test_set_random, train_set_wd, test_set_wd, train_set_ood, test_set_ood
