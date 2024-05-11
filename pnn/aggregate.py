"""
Functions for aggregation.
"""
from typing import Iterable

import numpy as np
import pandas as pd

from . import metrics

def average_uncertainty(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate the uncertainty in given dataframes.
    Note: previous version of heatmaps filtered entries <0 and >200/>1000 in total/epistemic/aleatoric.
    """
    results_agg = {key: df.groupby(level=0).mean() for key, df in results.items()}
    results_agg = pd.concat(results_agg)
    return results_agg


_TRUE_KEY = "y_true"
_MEAN_KEY = "y_pred"
_STD_KEY = "pred_std"
def sharpness_coverage(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate the sharpness etc. per IOP per dataframe, then combine them into one.
    """
    sharpness = pd.DataFrame({key: metrics.sharpness(df.loc[_STD_KEY]) for key, df in results.items()}).T
    coverage_factors = pd.DataFrame({key: metrics.coverage(df.loc[_TRUE_KEY], df.loc[_MEAN_KEY], df.loc[_STD_KEY]) for key, df in results.items()}).T

    combined = pd.concat({"sharpness": sharpness,
                          "coverage": coverage_factors,})
    combined = combined.reorder_levels([1, 0])

    return combined
