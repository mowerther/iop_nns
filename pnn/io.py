"""
Functions for reading the PNN output dataframes.
"""
import itertools
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import constants as c

LEVEL_ORDER = ["category", "split", "network", "instance"]


### LOADING / PROCESSING DATA
def reorganise_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorganise an input dataframe so that "Category" and "Instance" become a multi-index.
    """
    df = df.set_index(["Category", "Instance"])
    df.index.rename({"Category": "category", "Instance": "instance"}, inplace=True)
    return df


def variance_to_uncertainty(df: pd.DataFrame, *, input_keys: Iterable[str]=[c.ale_var, c.epi_var]) -> pd.DataFrame:
    """
    Calculate the uncertainty corresponding to variances in the given dataframe.
    Default: aleatoric (ale_var -> ale_unc) and epistemic (epi_var -> epi_unc).
    """
    output_keys = [key.name.replace("var", "unc") for key in input_keys]
    stds = np.sqrt(df.loc[input_keys])
    stds.index = stds.index.set_levels(output_keys, level=0)
    return stds


def calculate_percentage_uncertainty(df: pd.DataFrame, *,
                                     reference_key: str=c.y_pred, uncertainty_keys: Iterable[str]=[c.total_unc, c.ale_unc, c.epi_unc]) -> pd.DataFrame:
    """
    Calculates the percentage uncertainty (total, aleatoric, epistemic) relative to the scaled prediction.

    Parameters:
    - df: the main dataframe containing all data.
    Optional:
    - reference_key: the index for the denominator (default: predictions)
    - uncertainty_keys: the indices for the numerators (default: total uncertainty, aleatoric uncertainty, epistemic uncertainty)
    """
    # Define helper functions
    to_percentage = lambda data, key: np.abs(df.loc[key] / df.loc[reference_key]) * 100
    update_key = lambda key: key.name + "_pct"

    # Perform the operation on the specified keys
    result = {update_key(key): to_percentage(df, key) for key in uncertainty_keys}
    result = pd.concat(result, names=["category"])

    return result


def calculate_aleatoric_fraction(df: pd.DataFrame, *,
                                 aleatoric_key: str=c.ale_var, total_key: str=c.total_var,
                                 fraction_key: str=c.ale_frac.name) -> pd.DataFrame:
    """
    Calculate what fraction of the total variance consists of aleatoric variance.
    """
    fraction = df.loc[aleatoric_key] / df.loc[total_key]
    result = pd.concat({fraction_key: fraction}, names=["category"])
    return result


def read_model_outputs(filename: Path | str) -> pd.DataFrame:
    """
    Read data from a dataframe and process it.
    """
    df = pd.read_csv(filename)
    df = reorganise_df(df)
    return df


def read_all_model_outputs(folder: Path | str=c.pred_path) -> pd.DataFrame:
    """
    Read all data from a given folder into one big dataframe.
    """
    # Read data
    results = {split.name: pd.concat({network.name: read_model_outputs(folder/f"{network.name}_{split.name}_preds.csv") for network in c.networks}, names=["network"]) for split in c.splits}
    results = pd.concat(results, names=["split"])

    # Reorder
    results = results.reorder_levels(LEVEL_ORDER)

    # Convert variance to uncertainty (std)
    stds = variance_to_uncertainty(results)
    results = pd.concat([results, stds])

    # Add additional information
    df_percent = calculate_percentage_uncertainty(results)
    df_aleatoric_fraction = calculate_aleatoric_fraction(results)

    # Put it all together
    results = pd.concat([results, df_percent, df_aleatoric_fraction])
    results = results.sort_index()
    return results
