"""
Metrics for assessing data/model quality.
These are optimised for pandas DataFrames.
"""
import functools
import warnings
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress

### CONSTANTS
# Metric display names and their data column names mapping
metrics_display = {
    "mdsa": "MDSA [%]",
    "sspb": "Bias [%]",
    "r_squared": r"$R^2$",
    }


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


def label(name):
    """
    Label a function for when it is printed.
    """
    def helper(f):
        f.__name__ = name
        return f
    return helper


### METRICS
@label('RMSE')
def rmse(y: pd.DataFrame, y_hat: pd.DataFrame) -> pd.Series:
	""" Root Mean Squared Error """
	return np.sqrt(((y - y_hat)**2).mean())

@only_positive
@label('RMSLE')
def rmsle(y: pd.DataFrame, y_hat: pd.DataFrame) -> pd.Series:
    """ Root Mean Squared Logarithmic Error """
    y, y_hat = np.log(1 + y), np.log(1 + y_hat)
    return rmse(y, y_hat)
    # return np.mean(np.abs(np.log(y) - np.log(y_hat)) ** 2) ** 0.5

@label('NRMSE')
def nrmse(y: pd.DataFrame, y_hat: pd.DataFrame) -> pd.Series:
	""" Normalized Root Mean Squared Error """
	return rmse(y, y_hat) / y.mean()

@label('R^2')
def r_squared(y: pd.DataFrame, y_hat: pd.DataFrame) -> pd.Series:
    """ R² """
    SSres = ((y - y_hat)**2).sum()
    SStot = ((y - y.mean())**2).sum()
    R2 = 1 - SSres/SStot
    return R2

@only_positive
def log_r_squared(y: pd.DataFrame, y_hat: pd.DataFrame) -> pd.Series:
    """ R² of the logs """
    y, y_hat = np.log10(y), np.log10(y_hat)
    return r_squared(y, y_hat)


@label('<=0')
def leqz(y1, y2=None):
    """ Less than or equal to zero (y2) """
    if y2 is None: y2 = y1
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        return (y2 <= 0).sum()

@label('<=0|NaN')
def leqznan(y1, y2=None):
    """ Less than or equal to zero (y2) """
    if y2 is None: y2 = y1
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        return np.logical_or(np.isnan(y2), y2 <= 0).sum()

@label('MAD')
def MAD(y1, y2):
    """ Mean Absolute Error """
    i  = np.logical_and(y1 > 0, y2 > 0)
    y1 = np.log10(y1[i])
    y2 = np.log10(y2[i])
    i  = np.logical_and(np.isfinite(y1), np.isfinite(y2))
    y1 = y1[i]
    y2 = y2[i]
    return 10**np.nanmean(np.abs(y1 - y2))-1

@label('MdAPE')
def mape(y, y_hat):
    """ Mean Absolute Percentage Error """
    med = np.abs( (y - y_hat) / y).median()
    MAPE = 100 * med
    return MAPE

@only_positive
@label('MSA')
def msa(y, y_hat):
	""" Mean Symmetric Accuracy """
	# https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017SW001669
	return 100 * (np.exp(np.nanmean(np.abs(np.log(y_hat / y)))) - 1)


@only_positive
@label('MdSA')
def mdsa(y, y_hat):
    """ Median Symmetric Accuracy """
    # https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017SW001669
    med = np.abs(np.log(y_hat / y)).median()
    MDSA = 100 * (np.exp(med) - 1)
    return MDSA

@only_positive
@label('SSPB')
def sspb(y, y_hat):
    """ Symmetric Signed Percentage Bias """
    # https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017SW001669
    med = np.log(y_hat / y).median()
    SSPB = 100 * np.sign(med) * (np.exp(np.abs(med)) - 1)
    return SSPB

@label('Bias')
def bias(y, y_hat):
	""" Mean Bias """
	return (y_hat - y).mean()

@only_positive
@label('Slope')
def slope(y, y_hat):
	""" Logarithmic slope """
	slope_, intercept_, r_value, p_value, std_err = linregress(np.log10(y), np.log10(y_hat))
	return slope_

@only_positive
@label('Intercept')
def intercept(y, y_hat):
	""" Locarithmic intercept """
	slope_, intercept_, r_value, p_value, std_err = linregress(np.log10(y), np.log10(y_hat))
	return intercept_

@label('MWR')
def mwr(y, y_hat, y_bench):
	"""
	Model Win Rate - Percent of samples in which model has a closer
	estimate than the benchmark.
		y: true, y_hat: model, y_bench: benchmark
	"""
	y_bench[y_bench < 0] = np.nan
	y_hat[y_hat < 0] = np.nan
	y[y < 0] = np.nan
	valid = np.logical_and(np.isfinite(y_hat), np.isfinite(y_bench))
	diff1 = np.abs(y[valid] - y_hat[valid])
	diff2 = np.abs(y[valid] - y_bench[valid])
	stats = np.zeros(len(y))
	stats[valid]  = diff1 < diff2
	stats[~np.isfinite(y_bench)] = 1
	stats[~np.isfinite(y_hat)] = 0
	return stats.sum() / np.isfinite(y).sum()


### AGGREGATE FUNCTIONS
def performance(key, y1, y2, metrics=[rmse, slope, msa, rmsle, sspb, MAD, leqznan]):#[rmse, rmsle, mape, r_squared, bias, mae, leqznan, slope]):
    """ Return a string containing performance using various metrics.
        y1 should be the true value, y2 the estimated value. """
    return '%8s | %s' % (key, '   '.join([
            '%s: %6.3f' % (f.__name__, f(y1,y2)) for f in metrics]))


_MASK_THRESHOLD = 1e-4
def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure non-negative and non-zero filtering
    mask = (df.loc["y_true"] > _MASK_THRESHOLD) & (df.loc["y_pred"] > _MASK_THRESHOLD)
    df = df[mask].loc[["y_true", "y_pred"]]  # Masked items become np.nan

    # Temporary: make condition work per column
    if mask.sum().sum() > 0:
        metrics = {
            'sspb': sspb(df.loc["y_true"], df.loc["y_pred"]),
            'mdsa': mdsa(df.loc["y_true"], df.loc["y_pred"]),
            'mape': mape(df.loc["y_true"], df.loc["y_pred"]),
            'r_squared': log_r_squared(df.loc["y_true"], df.loc["y_pred"])
            }
    else:
        metrics = {'sspb': None, 'mdsa': None, 'mape': None, 'r_squared': None}

    metrics = pd.DataFrame(metrics).T
    return metrics
