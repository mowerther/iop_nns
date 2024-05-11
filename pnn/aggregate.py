"""
Functions for aggregation.
"""
from typing import Iterable

import numpy as np
import pandas as pd

from . import metrics

def aggregate_uncertainty(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate the uncertainty in given dataframes.
    Note: previous version of heatmaps filtered entries <0 and >200/>1000 in total/epistemic/aleatoric.
    """
    results_agg = {key: df.groupby(level=0).mean() for key, df in results.items()}
    results_agg = pd.concat(results_agg)
    return results_agg


_STD_KEY = "pred_std"
def aggregate_sharpness(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate the sharpness etc. per IOP per dataframe, then combine them into one.
    """
    sharpness_df = pd.DataFrame({key: metrics.sharpness(df.loc[_STD_KEY]) for key, df in results.items()}).T
    return sharpness_df
