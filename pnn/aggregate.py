"""
Functions for aggregation.
"""
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import uncertainty_toolbox as uct

from . import constants as c, metrics
from .recalibration import calibration_curve_single

### CONSTANTS
split_network = ["split", "network"]  # Aggregation levels


### HELPER FUNCTIONS
def metric_to_groupby(func: Callable, *df_keys: Iterable[str]) -> Callable:
    """
    Convert a metric function to one that can be applied to groupby dataframes.
    Example:
        def sspb(y, y_hat): ...

        new_sspb = metric_to_groupby(sspb, "y_true", "y_pred")
        sspb_results = df.groupby(level=["split", "network"]).apply(new_sspb)
    """
    def newfunc(df):
        return func(*[df.loc[key] for key in df_keys])
    return newfunc


### AGGREGATE METRICS
sspb = metric_to_groupby(metrics.sspb, c.y_true, c.y_pred)
mdsa = metric_to_groupby(metrics.mdsa, c.y_true, c.y_pred)
mape = metric_to_groupby(metrics.mape, c.y_true, c.y_pred)
log_r_squared = metric_to_groupby(metrics.log_r_squared, c.y_true, c.y_pred)
interval_sharpness = metric_to_groupby(metrics.interval_sharpness, c.y_true, c.y_pred, c.total_unc)
coverage = metric_to_groupby(metrics.coverage, c.y_true, c.y_pred, c.total_unc)
miscalibration_area = metric_to_groupby(metrics.miscalibration_area, c.y_true, c.y_pred, c.total_unc)


### FUNCTIONS
_MASK_THRESHOLD = 1e-4
def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure non-negative and non-zero filtering
    mask = (df.loc["y_true"] > _MASK_THRESHOLD) & (df.loc["y_pred"] > _MASK_THRESHOLD)
    df = df[mask]  # Masked items become np.nan

    # Use groupby to apply metrics across combinations
    df = df.groupby(level=split_network)

    df_metrics = {"sspb": df.apply(sspb),
                  "mdsa": df.apply(mdsa),
                  "mape": df.apply(mape),
                  "r_squared": df.apply(log_r_squared),
                  "sharpness": df.apply(interval_sharpness),
                  "coverage": df.apply(coverage),
                  "miscalibration area": df.apply(miscalibration_area),
                  }

    # Reorganise results
    df_metrics = pd.concat(df_metrics, names=["metric"])
    df_metrics = df_metrics.reorder_levels([*split_network, "metric"])

    # Sort
    df_metrics = df_metrics[c.iops]

    return df_metrics


def average_uncertainty(results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the uncertainty in given dataframes.
    Note: previous version of heatmaps filtered entries <0 and >200/>1000 in total/epistemic/aleatoric.
    """
    level = results.index.names.difference(["instance"])  # Aggregate only over instance
    results_agg = results.groupby(level=level).median()
    return results_agg


# Recalibration
def _calibration_curve_pernetwork(df: pd.DataFrame, *, columns=c.iops) -> pd.DataFrame:
    """
    Apply calibration_curve_single to the columns of a DataFrame - by default the IOPs for a single split/network combination.
    """
    observed = {key.name: calibration_curve_single(df[key]) for key in columns}
    observed = pd.concat(observed, axis=1)
    observed.columns = observed.columns.droplevel(1)
    return observed


def calibration_curve(results: pd.DataFrame) -> pd.DataFrame:
    """
    Determine calibration curves for results.
    """
    observed = results.groupby(level=split_network).apply(_calibration_curve_pernetwork)
    return observed
