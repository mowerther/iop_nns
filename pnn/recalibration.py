"""
Functions relating to recalibration, e.g. pre-processing.
"""
from functools import partial
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import uncertainty_toolbox as uct
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from . import constants as c

### DATA HANDLING
def split(training_data: Iterable[pd.DataFrame] | pd.DataFrame, *, recalibration_fraction=0.2) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """
    Split the training data into training and recalibration data.
    """
    # Handle the case where a single DataFrame is being split
    SINGLE = isinstance(training_data, pd.DataFrame)
    if SINGLE:
        training_data = [training_data]

    # Split individually - cannot be done with train_test_split(*args) because of differing array sizes.
    individual_splits = [train_test_split(df, test_size=recalibration_fraction, random_state=9) for df in training_data]
    training_data = [l[0] for l in individual_splits]
    recalibration_data = [l[1] for l in individual_splits]

    if SINGLE:
        training_data, recalibration_data = training_data[0], recalibration_data[0]

    return training_data, recalibration_data


### RECALIBRATION - FITTING
def fit_recalibration_function_single(y_true: np.ndarray, predicted_mean: np.ndarray, total_variance: np.ndarray) -> Callable:
    """
    Fit a recalibration function to one row/column of data.
    """
    total_uncertainty = np.sqrt(total_variance)
    recalibrator = uct.recalibration.get_quantile_recalibrator(predicted_mean, total_uncertainty, y_true)
    return recalibrator


def fit_recalibration_functions(y_true: np.ndarray, predicted_mean: np.ndarray, total_variance: np.ndarray) -> list[Callable]:
    """
    For each output (column in the input arrays), fit a recalibration function.
    """
    recalibrators = [fit_recalibration_function_single(y_true[:, j], predicted_mean[:, j], total_variance[:, j]) for j in range(y_true.shape[1])]
    return recalibrators


### RECALIBRATION - APPLICATION
def apply_recalibration_single(recalibrator: Callable, predicted_mean: np.ndarray, total_variance: np.ndarray) -> np.ndarray:
    """
    Apply a quantile-based recalibration function to one row/column of data.
    The lower/upper bounds are taken at mu-sigma, mu+sigma.
    """
    # Filter NaNs
    total_uncertainty = np.sqrt(total_variance)
    valid = np.isfinite(total_uncertainty)

    # Recalibrate
    lower, upper = recalibrator(predicted_mean[valid], total_uncertainty[valid], c.k1_lower), recalibrator(predicted_mean[valid], total_uncertainty[valid], c.k1_upper)
    new_uncertainty = (upper - lower) / 2

    # Convert back to variance, accounting for NaNs
    new_variance = np.tile(np.nan, total_uncertainty.shape)
    new_variance[valid] = new_uncertainty**2
    return new_variance


def apply_recalibration(recalibrators: Iterable[Callable], predicted_mean: np.ndarray, total_variance: np.ndarray) -> np.ndarray:
    """
    For each output (column in the input arrays), apply the respective recalibration function.
    """
    new_variances = [apply_recalibration_single(func, predicted_mean[:, j], total_variance[:, j]) for j, func in enumerate(recalibrators)]
    new_variances = np.array(new_variances).T
    return new_variances


### CALIBRATION CURVES
def calibration_curve_single(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the calibration curve for a single DataFrame with predicted mean, predicted uncertainty (std), and reference ("true") values.
    To do: Output Series rather than DataFrame.
    """
    # Filter NaNs
    y_pred, total_uncertainty, y_true = [df.loc[key].to_numpy() for key in (c.y_pred, c.total_unc, c.y_true)]
    valid = np.isfinite(total_uncertainty)
    y_pred, total_uncertainty, y_true = [arr[valid] for arr in (y_pred, total_uncertainty, y_true)]

    # Curves
    expected, observed = uct.get_proportion_lists_vectorized(y_pred, total_uncertainty, y_true)
    observed = pd.DataFrame(index=expected, data=observed).rename_axis("expected")
    return observed
