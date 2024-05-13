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


def log_binned_statistics_df(df: pd.DataFrame, *,
                             reference_key: str="y_true", uncertainty_keys: Iterable[str]=uncertainty_types.keys(),
                             columns: Iterable[str]=iops) -> pd.DataFrame:
    """
    Calculate log binned statistics for each of the variables in one dataframe.
    """
    # Perform calculations
    binned = pd.concat({unc_key: pd.concat({col: log_binned_statistics(df.loc[reference_key, col], df.loc[unc_key, col]) for col in columns}, axis=1) for unc_key in uncertainty_keys})

    return binned


def log_binned_statistics_combined(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate log binned statistics for each of the uncertainty dataframes relative to x, and combine them into a single dataframe.
    """
    # Reorder data and apply
    df = df.swaplevel("category", "split")
    binned = df.groupby(level=["split", "network"]).apply(log_binned_statistics_df)

    return binned
