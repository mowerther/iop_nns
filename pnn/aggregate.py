"""
Functions for aggregation.
"""
from typing import Iterable

import pandas as pd


def aggregate_uncertainty(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate the uncertainty in given dataframes.
    """
    results_agg = {key: df.groupby(level=0).mean() for key, df in results.items()}
    results_agg = pd.concat(results_agg)
    return results_agg
