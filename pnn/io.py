"""
Functions for reading the PNN output dataframes.
"""
import itertools
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import pred_path, network_types, split_types

### LOADING / PROCESSING DATA
def filter_df(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """
    Reorganise a single dataframe to have Instance as its index, only contain data columns, and (optionally) have a suffix added to all columns.
    """
    df_filtered = df.loc[df["Category"] == category]
    return df_filtered.set_index("Instance") \
                      .drop(columns=["Category"])


def reorganise_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorganise an input dataframe so that "Category" and "Instance" become a multi-index.
    """
    return df.set_index(["Category", "Instance"])


def calculate_percentage_uncertainty(df: pd.DataFrame, *,
                                     reference_key: str="pred_scaled_for_unc", uncertainty_keys: Iterable[str]=("total_unc", "ale_unc", "epi_unc")) -> pd.DataFrame:
    """
    Calculates the percentage uncertainty (total, aleatoric, epistemic) relative to the scaled prediction.

    Parameters:
    - df: the main dataframe containing all data.
    Optional:
    - reference_key: the index for the denominator (default: "pred_scaled_for_unc")
    - uncertainty_keys: the indices for the numerators (default: "total_unc, "ale_unc", "epi_unc")
    """
    # Define helper functions
    to_percentage = lambda data, key: np.abs(df.loc[key] / df.loc[reference_key]) * 100
    update_key = lambda key: key + "_pct"

    # Perform the operation on the specified keys
    result = {update_key(key): to_percentage(df, key) for key in uncertainty_keys}
    result = pd.concat(result)

    return result


def read_data(filename: Path | str) -> pd.DataFrame:
    """
    Read data from a dataframe and process it.
    """
    df = pd.read_csv(filename)
    df = reorganise_df(df)

    df_percent = calculate_percentage_uncertainty(df)
    df = pd.concat([df, df_percent])

    return df


def read_all_data(folder: Path | str=pred_path) -> dict[str, pd.DataFrame]:
    """
    Read all data from a given folder into dataframes.
    """
    results = {f"{network}_{split}": read_data(pred_path/f"{network}_{split}_preds.csv") for network, split in itertools.product(network_types, split_types)}
    return results
