"""
Functions etc. to be shared between network architectures, e.g. loss functions.
"""
from typing import Iterable

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from .. import constants as c, metrics as m

### DATA HANDLING
def extract_inputs_outputs(data: pd.DataFrame, *,
                           Rrs_stepsize: int=5, y_columns: Iterable[str]=c.iops) -> tuple[np.ndarray, np.ndarray]:
    """
    For a given DataFrame, extract the Rrs columns (X) and IOP columns (y).
    """
    rrs_columns = [f"Rrs_{nm}" for nm in range(400, 701, 5)]
    X = data[rrs_columns].values
    y = data[y_columns].values
    return X, y


def scale_y(y_train: np.ndarray, y_test: np.ndarray, *,
            scaled_range: tuple[int]=(-1, 1)) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply log and minmax scaling to the input arrays, training the scaler on y_train only (per IOP).
    """
    # Apply log transformation to the target variables
    y_train_log = np.log(y_train)
    y_test_log = np.log(y_test)

    # Apply Min-Max scaling to log-transformed target variables
    scaler_y = MinMaxScaler(feature_range=scaled_range)
    y_train_scaled = scaler_y.fit_transform(y_train_log)
    y_test_scaled = scaler_y.transform(y_test_log)

    return y_train_scaled, y_test_scaled, scaler_y


def inverse_scale_y(mean_scaled: np.ndarray, variance_scaled: np.ndarray, scaler_y: MinMaxScaler) -> tuple[np.ndarray, np.ndarray]:
    """
    Go back from log-minmax-scaled space to real units.
    """
    # Setup
    N = scaler_y.n_features_in_  # Number of predicted values
    original_shape = mean_scaled.shape

    # Convert from scaled space to log space: Means
    mean_scaled = mean_scaled.reshape(-1, N)
    mean_log = scaler_y.inverse_transform(mean_scaled)
    mean_log = mean_log.reshape(original_shape)

    # Convert from scaled space to log space: Variance
    scaling_factor = (scaler_y.data_max_ - scaler_y.data_min_) / 2
    variance_log = variance_scaled * (scaling_factor**2)  # Uncertainty propagation for linear equations

    # Convert from log space to the original space, i.e. actual IOPs in [m^-1]
    mean = np.exp(mean_log)  # Geometric mean / median
    variance = np.exp(2*mean_log + variance_log) * (np.exp(variance_log) - 1)  # Arithmetic variance

    return mean, variance


### TRAINING
def nll_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Negative Log Likelihood (NLL) loss function.
    `y_true` contains N reference values per row.
    `y_pred` contains N predicted mean values, followed by N predicted variances, per row:
        [mean1, mean2, ..., meanN, var1, var2, ..., varN]
    """
    N = y_true.shape[1]
    mean = y_pred[:, :N]
    var = tf.nn.softplus(y_pred[:, N:])

    return tf.reduce_mean(0.5 * (tf.math.log(var) + (tf.square(y_true - mean) / var) + tf.math.log(2 * np.pi)))



### ASSESSMENT
def calculate_metrics(y_true, y_pred, *, columns: Iterable[str]=c.iops_names) -> pd.DataFrame:
    """
    Calculate the mean absolute percentage error (MAPE) and other metrics between true and predicted values.

    Args:
    - y_true: Actual values (numpy array).
    - y_pred: Predicted values (numpy array).

    Returns:
    - DataFrame of metrics (MAPD, MAD, sspb, mdsa) for the predictions.
    """
    # Ensure y_true and y_pred are numpy arrays for element-wise operations
    y_true = pd.DataFrame(y_true, columns=columns)
    y_pred = pd.DataFrame(y_pred, columns=columns)

    # Calculate metrics
    metrics_combined = {"MAPE": m.mape(y_true, y_pred),
                        "MAD": m.MAD(y_true, y_pred),
                        "SSPB": m.sspb(y_true, y_pred),
                        "MdSA": m.mdsa(y_true, y_pred),
                        "r_squared": m.r_squared(y_true, y_pred),
                        "log_r_squared": m.log_r_squared(y_true, y_pred),}

    metrics_combined = pd.DataFrame(metrics_combined)

    return metrics_combined
