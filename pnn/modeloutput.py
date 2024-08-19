"""
Functions for reading the PNN outputs.
"""
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from . import constants as c

LEVEL_ORDER_METRICS = ["scenario", "network", "model", "variable"]
LEVEL_ORDER_SINGLE_METRICS = ["model", "variable"]
LEVEL_ORDER_OUTPUTS = ["category", "scenario", "network", "instance"]
LEVEL_ORDER_SINGLE_OUTPUT = ["category", "instance"]


### LOADING / PROCESSING MODEL METRICS
def read_model_metrics(filename: Path | str, **kwargs) -> pd.DataFrame:
    """
    Read data from a dataframe and process it.
    """
    df = pd.read_csv(filename, index_col=LEVEL_ORDER_SINGLE_METRICS, **kwargs)
    return df


def read_all_model_metrics(folder: Path | str=c.model_estimates_path, *,
                           scenarios: Iterable[c.Parameter]=c.scenarios_123,
                           use_recalibration_data=False) -> pd.DataFrame:
    """
    Read all data from a given folder into one big dataframe.
    """
    # Setup
    filename_base = "metrics.csv"
    if use_recalibration_data:
        filename_base = "recal_" + filename_base

    # Read data
    metrics = {scenario.name: pd.concat({network.name: read_model_metrics(folder/f"{network.name}_{scenario.name}_{filename_base}") for network in c.networks}, names=["network"]) for scenario in scenarios}
    metrics = pd.concat(metrics, names=["scenario"])

    # Reorder
    metrics = metrics.reorder_levels(LEVEL_ORDER_METRICS)

    return metrics


### FINDING THE MEDIAN MODEL
def find_median_indices(metrics: pd.DataFrame, *,
                        median_by_metric: c.Parameter=c.mdsa, columns: Iterable[c.Parameter]=c.iops_443,
                        mode: Optional[str]="median") -> pd.Series:
    """
    Find the median (or best) model out of N, for each combination of scenario and network.
    """
    # Reduce DataFrame and argsort
    values = metrics[median_by_metric]
    values = values.unstack()
    values = values[columns]
    values = values.sum(axis=1)
    indices = values.groupby(level=c.scenario_network).transform(pd.Series.argsort)

    # Index
    if mode == "median":
        m = values.index.levshape[-1] // 2  # Size of last ("model") index, divide by 2
    elif mode == "best":
        m = 0
    else:
        raise ValueError(f"Cannot handle mode '{mode}' -- please use 'median' or 'best'.")

    # Find the median (or best) indices
    median_indices = indices.unstack()[m]
    median_indices.rename("model", inplace=True)

    return median_indices


def select_median_metrics(metrics: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Pull out metrics for the median model.
    """
    # Get indices
    median_indices = find_median_indices(metrics, **kwargs)

    # Select
    # This can probably be done with clever indexing but I haven't figured it out yet
    def _selector(df: pd.DataFrame) -> pd.DataFrame:
        i = median_indices[df.index].iloc[0]  # Get the corresponding index
        return df.loc[:, :, i]  # Select element from df; hardcoded index levels

    median_metrics = metrics.groupby(c.scenario_network, group_keys=False).apply(_selector)

    return median_indices, median_metrics


### LOADING / PROCESSING INDIVIDUAL MODEL OUTPUTS
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
    data_combined = pd.concat(data_combined, names=LEVEL_ORDER_SINGLE_OUTPUT)

    # Save
    data_combined.to_csv(filename)


def read_model_outputs(filename: Path | str, **kwargs) -> pd.DataFrame:
    """
    Read data from a dataframe and process it.
    """
    df = pd.read_csv(filename, index_col=LEVEL_ORDER_SINGLE_OUTPUT, **kwargs)
    return df


def read_all_model_outputs(folder: Path | str=c.model_estimates_path, *,
                           scenarios: Iterable[c.Parameter]=c.scenarios_123,
                           subfolder_indices: Optional[pd.Series]=None,
                           use_recalibration_data=False) -> pd.DataFrame:
    """
    Read all data from a given folder into one big dataframe.

    If `subfolder_indices` is provided, use the indices to find the appropriate subfolder for each scenario/network combination.
    Example: if subfolder_indices.loc["ood_split", "rnn"] is 3, load from `folder/"3"/...` instead of `folder/...`.
    """
    # Setup: filenames
    filename_base = "estimates.csv"
    if use_recalibration_data:
        filename_base = "recal_" + filename_base

    # Setup: subfolders
    if subfolder_indices is None:
        read_single = lambda folder, scenario, network, filename_base: read_model_outputs(folder/f"{network.name}_{scenario.name}_{filename_base}")
    else:
        def read_single(folder, scenario, network, filename_base):
            i = subfolder_indices.loc[scenario, network]  # Get the corresponding index
            return read_model_outputs(folder/str(i)/f"{network.name}_{scenario.name}_{filename_base}")

    # Read data
    results = {scenario.name: pd.concat({network.name: read_single(folder, scenario, network, filename_base) for network in c.networks}, names=["network"]) for scenario in scenarios}
    results = pd.concat(results, names=["scenario"])

    # Reorder
    results = results.reorder_levels(LEVEL_ORDER_OUTPUTS)

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
