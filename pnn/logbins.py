"""
Log-binned statistics, for line plots.
"""
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import iops, uncertainty_types

def log_binned_statistics(x: pd.Series, y: pd.Series, *,
                          vmin: float=1e-4, vmax: float=40, binwidth: float=0.2, n: int=100) -> pd.DataFrame:
    """
    Calculate statistics (mean, std) for y as a function of x, in log-space bins.
    binwidth and n can be chosen separately to allow for overlapping bins.
    Example:
        x = in situ data
        y = total uncertainty in prediction
        vmin = 0.1 ; vmax = 10 ; binwidth = 1 ; n = 2
        Calculates mean and std uncertainty in prediction at (0.1 < x < 1), (1 < x < 10)
    """
    # Setup
    x_log = np.log10(x)
    bins_log = np.linspace(np.log10(vmin), np.log10(vmax), n)
    bins = np.power(10, bins_log)

    # Find the indices per bin
    slices = [(x_log >= centre - binwidth) & (x_log <= centre + binwidth) for centre in bins_log]
    binned = [y.loc[s] for s in slices]

    # Calculate statistics
    binned_means = [b.mean() for b in binned]
    binned_stds = [b.std() for b in binned]

    # Wrap into dataframe
    binned = pd.DataFrame(index=bins, data={"mean": binned_means, "std": binned_stds})
    return binned


def log_binned_statistics_all_variables(reference: pd.DataFrame, data: pd.DataFrame, *,
                                        columns: Iterable[str]=iops) -> pd.DataFrame:
    """
    Calculate log binned statistics for each of the variables in one dataframe.
    """
    # Perform calculations
    binned = {key: log_binned_statistics(reference.loc[:,key], data.loc[:,key]) for key in columns}

    # Add suffices to columns and merge
    binned = [df.add_prefix(f"{key}_") for key, df in binned.items()]
    binned = binned[0].join(binned[1:])

    return binned


def log_binned_statistics_combined(df: pd.DataFrame, *,
                                   reference_key: str="y_true", uncertainty_keys: Iterable[str]=uncertainty_types.keys()) -> pd.DataFrame:
    """
    Calculate log binned statistics for each of the uncertainty dataframes relative to x, and combine them into a single dataframe.
    """
    # Bin individual uncertainty keys
    binned = {key: log_binned_statistics_all_variables(df.loc[reference_key], df.loc[key]) for key in uncertainty_keys}

    # Combine into one multi-index DataFrame
    binned = pd.concat(binned)

    return binned
