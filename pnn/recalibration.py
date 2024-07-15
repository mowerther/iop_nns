"""
Functions relating to recalibration, e.g. pre-processing.
"""
from functools import partial
from typing import Iterable

import numpy as np
import pandas as pd
import uncertainty_toolbox as uct
from sklearn.isotonic import IsotonicRegression
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

def split(training_data: Iterable[pd.DataFrame], *, recalibration_fraction=0.2) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """
    Split the training data into training and recalibration data.
    """
    # Split individually - cannot be done with train_test_split(*args) because of differing array sizes.
    individual_splits = [train_test_split(df, test_size=recalibration_fraction, random_state=9) for df in training_data]
    training_data = [l[0] for l in individual_splits]
    recalibration_data = [l[1] for l in individual_splits]
    return training_data, recalibration_data


def fit_recalibration_function_single(y_true: np.ndarray, predicted_mean: np.ndarray, total_variance: np.ndarray) -> IsotonicRegression:
    """
    Fit a recalibration function to one row/column of data.
    """
    total_uncertainty = np.sqrt(total_variance)
    exp_props, obs_props = uct.get_proportion_lists_vectorized(predicted_mean, total_uncertainty, y_true)
    recalibrator = uct.iso_recal(exp_props, obs_props)
    return recalibrator


def fit_recalibration_functions(y_true: np.ndarray, predicted_mean: np.ndarray, total_variance: np.ndarray) -> list[IsotonicRegression]:
    """
    For each output (column in the input arrays), fit a recalibration function.
    """
    recalibrators = [fit_recalibration_function_single(y_true[:, i], predicted_mean[:, i], total_variance[:, i]) for i in range(y_true.shape[1])]
    return recalibrators


def calibration_curve_single(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the calibration curve for a single DataFrame with predicted mean, predicted uncertainty (std), and reference ("true") values.
    To do: Output Series rather than DataFrame.
    """
    expected, observed = uct.get_proportion_lists_vectorized(df.loc[c.y_pred].to_numpy(), df.loc[c.total_unc].to_numpy(), df.loc[c.y_true].to_numpy())
    observed = pd.DataFrame(index=expected, data=observed).rename_axis("expected")
    return observed


def miscalibration_area_single(df: pd.DataFrame) -> float:
    """
    Calculate the miscalibration area for a single DataFrame with predicted mean, predicted uncertainty (std), and reference ("true") values.
    """
    return uct.miscalibration_area(df.loc[c.y_pred].to_numpy(), df.loc[c.total_unc].to_numpy(), df.loc[c.y_true].to_numpy())


# titles = ['aCDOM_443', 'aCDOM_675', 'aNAP_443', 'aNAP_675', 'aph_443', 'aph_675']
# for tit, var_idx in zip(titles, range(6)):  # Per IOP
#     cal_pred_mean = cal_mean_preds[:, var_idx]
#     cal_pred_std = cal_std_preds[:, var_idx]
#     cal_actual_values = y_recalib[:, var_idx]

#     org_mean_preds = mean_preds[:, var_idx]
#     org_std_preds = std_preds[:, var_idx]
#     org_y_test = y_test[:, var_idx]

#     # prepare with recalibrated data, fit a model on the recalibrated data
#     exp_props, obs_props = uct.get_proportion_lists_vectorized(cal_pred_mean, cal_pred_std, cal_actual_values)
#     calib_model = uct.iso_recal(exp_props, obs_props)

#     # then adjust the props on the test data with the recal model from recalibrated data
#     recal_exp_props, recal_obs_props = uct.get_proportion_lists_vectorized(org_mean_preds, org_std_preds, org_y_test, recal_model=calib_model)

#     # plot with and without the recal data
#     mace = uct.mean_absolute_calibration_error(org_mean_preds, org_std_preds, org_y_test, recal_model=None)
#     rmsce = uct.root_mean_squared_calibration_error(org_mean_preds, org_std_preds, org_y_test, recal_model=None)
#     ma = uct.miscalibration_area(org_mean_preds, org_std_preds, org_y_test, recal_model=None)
#     print("Before Recalibration:  ", end="")
#     print("MACE: {:.5f}, RMSCE: {:.5f}, MA: {:.5f}".format(mace, rmsce, ma))

#     mace = uct.mean_absolute_calibration_error(org_mean_preds, org_std_preds, org_y_test, recal_model=calib_model)
#     rmsce = uct.root_mean_squared_calibration_error(org_mean_preds, org_std_preds, org_y_test, recal_model=calib_model)
#     ma = uct.miscalibration_area(org_mean_preds, org_std_preds, org_y_test, recal_model=calib_model)
#     print("After Recalibration:  ", end="")
#     print("MACE: {:.5f}, RMSCE: {:.5f}, MA: {:.5f}".format(mace, rmsce, ma))

#     ax = axes[var_idx]
#     ax.plot(exp_props, obs_props, label='Before Recalibration', marker='o')
#     ax.plot(recal_exp_props, recal_obs_props, label='After Recalibration', marker='x')
#     ax.plot([0, 1], [0, 1], 'k--', label='1:1 Line')
#     ax.set_title(tit)
#     ax.set_xlabel('Expected Proportions')
#     ax.set_ylabel('Observed Proportions')
#     ax.legend()

# Test set predictions
mean_preds, total_var, aleatoric_var, epistemic_var, std_preds = predict_with_uncertainty(model, X_test, scaler_y, n_samples=100)

# Predictions with trained model on recalib data
cal_mean_preds, cal_total_var, cal_aleatoric_var, cal_epistemic_var, cal_std_preds = predict_with_uncertainty(model, X_recalib, scaler_y, n_samples=100)

# Fit recal models
calib_models = []
for var_idx in range(6):
    cal_pred_mean = cal_mean_preds[:, var_idx]
    cal_pred_std = cal_std_preds[:, var_idx]
    cal_actual_values = y_recalib[:, var_idx]

    exp_props, obs_props = uct.get_proportion_lists_vectorized(cal_pred_mean, cal_pred_std, cal_actual_values)
    calib_model = uct.iso_recal(exp_props, obs_props)
    calib_models.append(calib_model)

for var_idx in range(6):
    # Test set preds, std, insitu/true
    test_variable_mean_preds = mean_preds[:, var_idx]
    test_variable_std_preds = std_preds[:, var_idx]
    test_variable_y_test = y_test[:, var_idx]

    # compute the expected and observed proportions using the recalibration model per variable
    recal_exp_props, recal_obs_props = uct.get_proportion_lists_vectorized(
        test_variable_mean_preds, test_variable_std_preds, test_variable_y_test, recal_model=calib_models[var_idx]
    )
    # Transform the test set standard deviations using the fitted isotonic regression model per variable
    test_recal_total_std = calib_models[var_idx].transform(test_variable_std_preds)
    # Calculate recalibrated total variances by squaring the recalibrated standard deviations per variable, but may not be necessary
    test_recal_total_var = np.zeros_like(total_var)
    test_recal_total_var[:, var_idx] = test_recal_total_std**2
