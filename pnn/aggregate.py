"""
Functions for aggregation.
"""
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from . import metrics

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
_TRUE_KEY = "y_true"
_MEAN_KEY = "y_pred"
_STD_KEY = "pred_std"
sspb = metric_to_groupby(metrics.sspb, _TRUE_KEY, _MEAN_KEY)
mdsa = metric_to_groupby(metrics.mdsa, _TRUE_KEY, _MEAN_KEY)
mape = metric_to_groupby(metrics.mape, _TRUE_KEY, _MEAN_KEY)
log_r_squared = metric_to_groupby(metrics.log_r_squared, _TRUE_KEY, _MEAN_KEY)
sharpness = metric_to_groupby(metrics.sharpness, _STD_KEY)
coverage = metric_to_groupby(metrics.coverage, _TRUE_KEY, _MEAN_KEY, _STD_KEY)


### FUNCTIONS
_MASK_THRESHOLD = 1e-4
def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure non-negative and non-zero filtering
    df = df.swaplevel("category", "split")
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


def average_uncertainty(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate the uncertainty in given dataframes.
    Note: previous version of heatmaps filtered entries <0 and >200/>1000 in total/epistemic/aleatoric.
    """
    results_agg = {key: df.groupby(level=0).mean() for key, df in results.items()}
    results_agg = pd.concat(results_agg)
    return results_agg
