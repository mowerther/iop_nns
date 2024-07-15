"""
Functions for reading the PNN output dataframes.
"""
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import constants as c

LEVEL_ORDER = ["category", "split", "network", "instance"]
LEVEL_ORDER_SINGLE = ["category", "instance"]


### LOADING / PROCESSING DATA
def variance_to_uncertainty(df: pd.DataFrame, *, input_keys: Iterable[str]=c.variances) -> pd.DataFrame:
    """
    Calculate the uncertainty corresponding to variances in the given dataframe.
    Default: aleatoric (ale_var -> ale_unc) and epistemic (epi_var -> epi_unc).
    """
    output_keys = [key.name.replace("var", "unc") for key in input_keys]
    stds = np.sqrt(df.loc[input_keys])
    stds.index = stds.index.set_levels(output_keys, level=0)
    return stds


def calculate_percentage_uncertainty(df: pd.DataFrame, *,
                                     reference_key: str=c.y_pred, uncertainty_keys: Iterable[str]=c.uncertainties) -> pd.DataFrame:
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
    Calculate what fraction (in %) of the total variance consists of aleatoric variance.
    """
    fraction = df.loc[aleatoric_key] / df.loc[total_key] * 100
    result = pd.concat({fraction_key: fraction}, names=["category"])
    return result


def save_model_outputs(y_true: np.ndarray, mean_predictions: np.ndarray, total_variance: np.ndarray, aleatoric_variance: np.ndarray, epistemic_variance: np.ndarray, filename: Path | str, *,
                       columns=c.iops) -> None:
    """
    Collate the true values and model outputs into a DataFrame and save it to file.
    """
    # Combine data
    data_combined = {c.y_true: y_true,
                     c.y_pred: mean_predictions,
                     c.total_var: total_variance,
                     c.ale_var: aleatoric_variance,
                     c.epi_var: epistemic_variance,}

    data_combined = {str(key): pd.DataFrame(arr, columns=columns) for key, arr in data_combined.items()}
    data_combined = pd.concat(data_combined, names=LEVEL_ORDER_SINGLE)

    # Save
    data_combined.to_csv(filename)


def read_model_outputs(filename: Path | str) -> pd.DataFrame:
    """
    Read data from a dataframe and process it.
    """
    df = pd.read_csv(filename, index_col=LEVEL_ORDER_SINGLE)
    return df


def read_all_model_outputs(folder: Path | str=c.pred_path, *,
                           use_recalibration_data=False) -> pd.DataFrame:
    """
    Read all data from a given folder into one big dataframe.
    """
    # Setup
    filename_base = "preds"
    if use_recalibration_data:
        filename_base = "recal_" + filename_base

    # Read data
    results = {split.name: pd.concat({network.name: read_model_outputs(folder/f"{network.name}_{split.name}_{filename_base}.csv") for network in c.networks}, names=["network"]) for split in c.splits}
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

    # Sort
    results = results[c.iops]
    return results
