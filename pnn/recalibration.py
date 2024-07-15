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

####
# recalibration procedure
####

# 1. train the model using X_train, y_train_scaled
# 2. apply the trained model to the test dataset:
    # mean_preds, total_var, aleatoric_var, epistemic_var, std_preds = predict_with_uncertainty(model, X_test, scaler_y, n_samples=100)
# 3. apply the trainedmodel to the recalibration dataset:
    # cal_mean_preds, cal_total_var, cal_aleatoric_var, cal_epistemic_var, cal_std_preds = predict_with_uncertainty(model, X_recalib, scaler_y, n_samples=100)
# All per individual target IOP:
# 4. Obtain expected (model) and obs proportions (in situ) from recalibration dataset
# 5. Fit an isotonic regression recalibration model with exp_props, obs_props
# 6. Obtain the recalibrated exp_props and obs_props from the test dataset (org_ in the script) using the previously fitted recalibration model
# 7. Calculate before and after recalibration metrics and plot

### DATA HANDLING
def split(training_data: Iterable[pd.DataFrame], *, recalibration_fraction=0.2) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """
    Split the training data into training and recalibration data.
    """
    # Split individually - cannot be done with train_test_split(*args) because of differing array sizes.
    individual_splits = [train_test_split(df, test_size=recalibration_fraction, random_state=9) for df in training_data]
    training_data = [l[0] for l in individual_splits]
    recalibration_data = [l[1] for l in individual_splits]
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
    total_uncertainty = np.sqrt(total_variance)
    lower, upper = recalibrator(predicted_mean, total_uncertainty, 0.15865525393), recalibrator(predicted_mean, total_uncertainty, 0.84134474606)
    new_uncertainty = (upper - lower) / 2
    new_variance = new_uncertainty**2
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
    expected, observed = uct.get_proportion_lists_vectorized(df.loc[c.y_pred].to_numpy(), df.loc[c.total_unc].to_numpy(), df.loc[c.y_true].to_numpy())
    observed = pd.DataFrame(index=expected, data=observed).rename_axis("expected")
    return observed



### METRICS
def miscalibration_area_single(df: pd.DataFrame) -> float:
    """
    Calculate the miscalibration area for a single DataFrame with predicted mean, predicted uncertainty (std), and reference ("true") values.
    """
    return uct.miscalibration_area(df.loc[c.y_pred].to_numpy(), df.loc[c.total_unc].to_numpy(), df.loc[c.y_true].to_numpy())
