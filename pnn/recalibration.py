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
