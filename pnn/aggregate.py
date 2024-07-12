"""
Functions for aggregation.
"""
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import uncertainty_toolbox as uct

from . import constants as c, metrics
from .recalibration import calibration_curve_single, miscalibration_area_single

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
sharpness = metric_to_groupby(metrics.sharpness, c.total_unc)
coverage = metric_to_groupby(metrics.coverage, c.y_true, c.y_pred, c.total_unc)


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
                  "sharpness": df.apply(sharpness),
                  "coverage": df.apply(coverage),
                  }

    # Reorganise results
    df_metrics = pd.concat(df_metrics, names=["metric"])
    df_metrics = df_metrics.reorder_levels([*split_network, "metric"])

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


def _miscalibration_area_pernetwork(df: pd.DataFrame, *, columns=c.iops) -> pd.Series:
    """
    Apply miscalibration_area_single to the columns of a DataFrame - by default the IOPs for a single split/network combination.
    """
    miscalibration_areas = {key.name: miscalibration_area_single(df[key]) for key in columns}
    miscalibration_areas = pd.Series(miscalibration_areas)
    return miscalibration_areas


def miscalibration_area(results: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the miscalibration area for each combination of split, network, and IOP.
    """
    miscalibration_areas = results.groupby(level=split_network).apply(_miscalibration_area_pernetwork)
    return miscalibration_areas
