"""
Metrics for assessing data/model quality.
These are optimised for pandas DataFrames.
"""
import functools
from typing import Callable, Iterable

import numpy as np
import pandas as pd

### HELPER DECORATOR FUNCTIONS
def only_positive(func: Callable) -> Callable:
    """
    Masks all non-positive (zero or negative) values in any inputs.
    Assumes the function has some number of DataFrame args, and checks those.
    """
    @functools.wraps(func)
    def helper(*args: Iterable[pd.DataFrame], **kwargs):
        # Mask non-positive inputs
        masks = [(df > 0) for df in args]
        masked_dfs = [df[mask] for df, mask in zip(args, masks)]

        # Check for fully masked items
        any_positive = [np.any(df.notna()) for df in masked_dfs]
        assert all(any_positive), f"Metric '{func.__name__}' requires positive inputs, but was provided an input without any."

        # Call main function on masked data
        return func(*masked_dfs, **kwargs)
    return helper


def label(name: str) -> Callable:
    """
    Label a function for when it is printed.
    """
    def helper(f):
        f.__name__ = name
        return f
    return helper


### METRICS
@label("RMSE")
def rmse(y: pd.DataFrame, y_hat: pd.DataFrame, **kwargs) -> pd.Series:
	""" Root Mean Squared Error """
	return np.sqrt(((y - y_hat)**2).mean(**kwargs))

@only_positive
@label("RMSLE")
def rmsle(y: pd.DataFrame, y_hat: pd.DataFrame, **kwargs) -> pd.Series:
    """ Root Mean Squared Logarithmic Error """
    y, y_hat = np.log(1 + y), np.log(1 + y_hat)
    return rmse(y, y_hat, **kwargs)

@label("NRMSE")
def nrmse(y: pd.DataFrame, y_hat: pd.DataFrame, **kwargs) -> pd.Series:
	""" Normalized Root Mean Squared Error """
	return rmse(y, y_hat) / y.mean(**kwargs)

@label("R²")
def r_squared(y: pd.DataFrame, y_hat: pd.DataFrame, **kwargs) -> pd.Series:
    """ Coefficient of determination (R²) """
    SSres = ((y - y_hat)**2).sum(**kwargs)
    SStot = ((y - y.mean())**2).sum(**kwargs)
    R2 = 1 - SSres/SStot
    return R2

@only_positive
@label("R² (log)")
def log_r_squared(y: pd.DataFrame, y_hat: pd.DataFrame, **kwargs) -> pd.Series:
    """ Coefficient of determination (R²) of the logarithms """
    y, y_hat = np.log10(y), np.log10(y_hat)
    return r_squared(y, y_hat, **kwargs)

@label("MAD")
def MAD(y: pd.DataFrame, y_hat: pd.DataFrame, **kwargs) -> pd.Series:
    """ Median Absolute Deviation """
    return (y_hat - y).abs().median(**kwargs)

@label("MdAPE")
def mape(y: pd.DataFrame, y_hat: pd.DataFrame, **kwargs) -> pd.Series:
    """ Mean Absolute Percentage Error """
    med = np.abs( (y - y_hat) / y).median(**kwargs)
    MAPE = 100 * med
    return MAPE

@only_positive
@label("MSA")
def msa(y: pd.DataFrame, y_hat: pd.DataFrame, **kwargs) -> pd.Series:
    """ Mean Symmetric Accuracy """
    # https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017SW001669
    m = np.abs(np.log(y_hat / y)).mean(**kwargs)
    MSA = 100 * (np.exp(m) - 1)
    return MSA

@only_positive
@label("MdSA")
def mdsa(y: pd.DataFrame, y_hat: pd.DataFrame, **kwargs) -> pd.Series:
    """
    Median Symmetric Accuracy
    https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017SW001669
    """
    med = np.abs(np.log(y_hat / y)).median(**kwargs)
    MDSA = 100 * (np.exp(med) - 1)
    return MDSA

# @only_positive
@label("SSPB")
def sspb(y: pd.DataFrame, y_hat: pd.DataFrame, **kwargs) -> pd.Series:
    """
    Symmetric Signed Percentage Bias
    https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017SW001669
    """
    med = np.log(y_hat / y).median(**kwargs)
    SSPB = 100 * np.sign(med) * (np.exp(np.abs(med)) - 1)
    return SSPB

@label("Bias")
def bias(y: pd.DataFrame, y_hat: pd.DataFrame, **kwargs) -> pd.Series:
	""" Bias (mean difference) """
	return (y_hat - y).mean(**kwargs)


@label("Sharpness")
def sharpness(y_std: pd.DataFrame, **kwargs) -> pd.Series:
    """ Sharpness (square root of mean of variance per sample) """
    return np.sqrt((y_std**2).median(**kwargs))

@label("Coverage")
def coverage(y_true: pd.DataFrame, y_pred: pd.DataFrame, y_pred_std: pd.DataFrame, *, k=1, **kwargs) -> pd.Series:
    """ Coverage factor (how often does the true value fall within the predicted range?) """
    lower_bounds = y_pred - k * y_pred_std
    upper_bounds = y_pred + k * y_pred_std
    within_bounds = (y_true >= lower_bounds) & (y_true <= upper_bounds)
    coverage_factors = within_bounds.mean(**kwargs) * 100  # [%]
    return coverage_factors

@label("Interval sharpness")
def interval_sharpness(y_true: pd.DataFrame, y_pred: pd.DataFrame, y_pred_std: pd.DataFrame, *, alpha: float=0.31731050787, **kwargs) -> pd.Series:
    """ Interval sharpness (how narrow are the prediction ranges?) """
    lower, upper = y_pred - y_pred_std, y_pred + y_pred_std

    # Calculate individual IS
    IS = alpha * (upper - lower)
    IS[y_true < lower] += 2 * (lower - y_true)
    IS[y_true > upper] += 2 * (y_true - upper)

    # Normalise
    ISnorm = (IS - IS.min(**kwargs)) / (IS.max(**kwargs) - IS.min(**kwargs))

    # Take mean
    ISbar = ISnorm.mean(**kwargs)

    return ISbar
