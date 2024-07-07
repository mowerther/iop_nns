"""
Functions for reading the (split) input data.
"""
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from . import constants as c


### INPUT / OUTPUT
rename_org = {"org_aph_443": "aph_443", "org_anap_443": "aNAP_443", "org_acdom_443": "aCDOM_443",
              "org_aph_675": "aph_675", "org_anap_675": "aNAP_675", "org_acdom_675": "aCDOM_675",}
def read_all_data(folder: Path | str=c.data_path) -> tuple[pd.DataFrame]:
    """
    Read all split data from a given folder into a number of DataFrames.
    The output cannot be a single DataFrame because of differing indices.
    """
    train_set_random = pd.read_csv(folder/"random_df_train_org.csv")
    test_set_random = pd.read_csv(folder/"random_df_test_org.csv")

    train_set_wd = pd.read_csv(folder/"wd_train_set_org.csv")
    test_set_wd = pd.read_csv(folder/"wd_test_set_org.csv")

    train_set_ood = pd.read_csv(folder/"ood_train_set_2.csv").drop(columns=rename_org.values()).rename(columns=rename_org)
    test_set_ood = pd.read_csv(folder/"ood_test_set_2.csv").drop(columns=rename_org.values()).rename(columns=rename_org)

    return train_set_random, test_set_random, train_set_wd, test_set_wd, train_set_ood, test_set_ood


def extract_inputs_outputs(data: pd.DataFrame, *,
                           Rrs_stepsize: int=5, y_columns: Iterable[str]=c.iops) -> tuple[np.ndarray, np.ndarray]:
    """
    For a given DataFrame, extract the Rrs columns (X) and IOP columns (y).
    """
    rrs_columns = [f"Rrs_{nm}" for nm in range(400, 701, 5)]
    X = data[rrs_columns].values
    y = data[y_columns].values
    return X, y


### SCALING
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
