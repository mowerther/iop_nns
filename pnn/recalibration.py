
#pre-processing for recalibration data

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

ood_train_df = pd.read_csv(f'{base_path}/ood_train_set_2.csv')
ood_test_df = pd.read_csv(f'{base_path}/ood_test_set_2.csv')

rrs_columns = [f'Rrs_{nm}' for nm in range(400, 701, 5)]
y_columns = ['org_acdom_443', 'org_acdom_675', 'org_anap_443', 'org_anap_675','org_aph_443', 'org_aph_675']

X_train_full = ood_train_df[rrs_columns].values
y_train_full = ood_train_df[y_columns].values

train_size = 0.4 / 0.5
X_train, X_recalib, y_train, y_recalib = train_test_split(X_train_full, y_train_full, train_size=train_size, random_state=9)

X_test = ood_test_df[rrs_columns].values
y_test = ood_test_df[y_columns].values

y_train_log = np.log(y_train)
y_recalib_log = np.log(y_recalib)
y_test_log = np.log(y_test)

scaler_y = MinMaxScaler(feature_range=(-1, 1))
y_train_scaled = scaler_y.fit_transform(y_train_log)
y_recalib_scaled = scaler_y.transform(y_recalib_log)
y_test_scaled = scaler_y.transform(y_test_log)

# Shapes
print("Training set shape:", X_train.shape, y_train_scaled.shape)
print("Recalibration set shape:", X_recalib.shape, y_recalib_scaled.shape)
print("Test set shape:", X_test.shape, y_test_scaled.shape)


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

import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()  

titles = ['aCDOM_443', 'aCDOM_675', 'aNAP_443', 'aNAP_675', 'aph_443', 'aph_675']
for tit, var_idx in zip(titles, range(6)):  # Per IOP
    cal_pred_mean = cal_mean_preds[:, var_idx]
    cal_pred_std = cal_std_preds[:, var_idx]
    cal_actual_values = y_recalib[:, var_idx]

    org_mean_preds = mean_preds[:, var_idx]
    org_std_preds = std_preds[:, var_idx]
    org_y_test = y_test[:, var_idx]

    # prepare with recalibrated data, fit a model on the recalibrated data
    exp_props, obs_props = uct.get_proportion_lists_vectorized(cal_pred_mean, cal_pred_std, cal_actual_values)
    calib_model = uct.iso_recal(exp_props, obs_props)

    # then adjust the props on the test data with the recal model from recalibrated data
    recal_exp_props, recal_obs_props = uct.get_proportion_lists_vectorized(org_mean_preds, org_std_preds, org_y_test, recal_model=calib_model)

    # plot with and without the recal data
    mace = uct.mean_absolute_calibration_error(org_mean_preds, org_std_preds, org_y_test, recal_model=None)
    rmsce = uct.root_mean_squared_calibration_error(org_mean_preds, org_std_preds, org_y_test, recal_model=None)
    ma = uct.miscalibration_area(org_mean_preds, org_std_preds, org_y_test, recal_model=None)
    print("Before Recalibration:  ", end="")
    print("MACE: {:.5f}, RMSCE: {:.5f}, MA: {:.5f}".format(mace, rmsce, ma))

    mace = uct.mean_absolute_calibration_error(org_mean_preds, org_std_preds, org_y_test, recal_model=calib_model)
    rmsce = uct.root_mean_squared_calibration_error(org_mean_preds, org_std_preds, org_y_test, recal_model=calib_model)
    ma = uct.miscalibration_area(org_mean_preds, org_std_preds, org_y_test, recal_model=calib_model)
    print("After Recalibration:  ", end="")
    print("MACE: {:.5f}, RMSCE: {:.5f}, MA: {:.5f}".format(mace, rmsce, ma))

    ax = axes[var_idx]
    ax.plot(exp_props, obs_props, label='Before Recalibration', marker='o')
    ax.plot(recal_exp_props, recal_obs_props, label='After Recalibration', marker='x')
    ax.plot([0, 1], [0, 1], 'k--', label='1:1 Line')
    ax.set_title(tit)
    ax.set_xlabel('Expected Proportions')
    ax.set_ylabel('Observed Proportions')
    ax.legend()

plt.tight_layout()
plt.show()
