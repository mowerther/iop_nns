"""
Log-binned statistics, for line plots.
"""
from typing import Iterable

import numpy as np
import pandas as pd

from . import constants as c

def log_binned_statistics(x: pd.Series, y: pd.Series, *,
                          vmin: float=1e-4, vmax: float=40, binwidth: float=0.2, n: int=100) -> pd.DataFrame:
    """
    Calculate statistics (median, std) for y as a function of x, in log-space bins.
    binwidth and n can be chosen separately to allow for overlapping bins.
    Example:
        x = in situ data
        y = total uncertainty in prediction
        vmin = 0.1 ; vmax = 10 ; binwidth = 1 ; n = 2
        Calculates median and std uncertainty in prediction at (0.1 < x < 1), (1 < x < 10)
    """
    # Setup
    x_log = np.log10(x)
    bins_log = np.linspace(np.log10(vmin), np.log10(vmax), n)
    bins = np.power(10, bins_log)

    # Find the indices per bin
    slices = [(x_log >= centre - binwidth) & (x_log <= centre + binwidth) for centre in bins_log]
    binned = [y.loc[s] for s in slices]

    # Calculate statistics
    binned_averages = [b.median() for b in binned]
    binned_stds = [b.std() for b in binned]

    # Wrap into dataframe
    binned = pd.DataFrame(index=bins, data={"median": binned_averages, "std": binned_stds})
    return binned


def log_binned_statistics_df(df: pd.DataFrame, *,
                             reference_key: str=c.y_true, uncertainties: Iterable[str]=c.relative_uncertainties,
                             columns: Iterable[str]=c.iops) -> pd.DataFrame:
    """
    Calculate log binned statistics for each of the variables in one dataframe.
    """
    # Perform calculations
    binned = pd.concat({unc.name: pd.concat({col.name: log_binned_statistics(df.loc[reference_key, col.name], df.loc[unc.name, col.name]) for col in columns}, axis=1) for unc in uncertainties})

    return binned


def log_binned_statistics_combined(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate log binned statistics for each of the uncertainty dataframes relative to x, and combine them into a single dataframe.
    """
    # Reorder data and apply
    binned = df.groupby(level=["split", "network"]).apply(log_binned_statistics_df)

    return binned
