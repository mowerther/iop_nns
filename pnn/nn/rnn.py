"""
Recurrent Neural Network with Gated Recurrent Units and Monte Carlo Dropout (RNN MCD).
"""
from functools import partial
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, History
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler

from .common import calculate_metrics, inverse_scale_y, nll_loss, predict_with_dropout
from .common import _train_general, _build_and_train_general, _predict_with_uncertainty_general, _train_and_evaluate_models_general
from .. import constants as c


### DATA HANDLING
def reshape_data(X_train: np.ndarray, *, X_test: Optional[np.ndarray]=None, n_features_per_timestep: int=1) -> tuple[np.ndarray, np.ndarray]:
    """
    Reshape data for the RNN, adding a timestep axis.
    """
    # Calculate the number of wavelength steps and features
    n_samples_train, n_features = X_train.shape
    n_timesteps = n_features // n_features_per_timestep  # Here: One wavelength per step

    # Reshape data
    X_train_reshaped = X_train.reshape((n_samples_train, n_timesteps, n_features_per_timestep))
    if X_test is not None:
        n_samples_test, _ = X_test.shape  # We already know the features count
        X_test_reshaped = X_test.reshape((n_samples_test, n_timesteps, n_features_per_timestep))

        return X_train_reshaped, X_test_reshaped
    else:
        return X_train_reshaped


### ARCHITECTURE
def build(input_shape: tuple, *, output_size: int=6,
          hidden_units: int=100, n_layers: int=5, dropout_rate: float=0.25, l2_reg: float=1e-3, activation="tanh") -> Model:
    """
    Construct an RNN with MCD based on the input parameters.
    To do: use functools.partial for GRU?
    To do: relu or tanh?
    """
    model = Sequential()
    model.add(Input(shape=input_shape))

    # Add the first recurrent layer with input shape
    model.add(GRU(hidden_units, return_sequences=(n_layers > 1),
                  activation=activation, kernel_regularizer=l2(l2_reg)))

    # Add additional GRU layers if n_layers > 1, with Dropout layers between them
    for i in range(1, n_layers):
        model.add(Dropout(dropout_rate))
        model.add(GRU(hidden_units, return_sequences=(i < n_layers-1),
                      activation=activation, kernel_regularizer=l2(l2_reg)))

    # Output layer: Adjust for 6 means and 6 variances (12 outputs in total)
    model.add(Dense(output_size * 2, activation="linear"))

    return model


### TRAINING
train = partial(_train_general, epochs=1000, batch_size=512)


def build_and_train(X_train: np.ndarray, y_train: np.ndarray) -> Model:
    """
    Build and train an RNN model on the provided X and y data.
    Same as the general function, but reshapes the data for the RNN specifically.
    """
    # Setup
    X_train = reshape_data(X_train)

    model = _build_and_train_general(build, train, X_train, y_train)
    return model


### APPLICATION
def predict_samples(model: Model, X: np.ndarray, *, n_samples: int=100) -> np.ndarray:
    """
    Predict y values for given X values using the RNN.
    """
    pred_samples = [predict_with_dropout(model, X, enable_dropout=True).numpy() for _ in range(n_samples)]
    pred_samples = np.array(pred_samples)
    return pred_samples


predict_with_uncertainty = partial(_predict_with_uncertainty_general, predict_samples)
train_and_evaluate_models = partial(_train_and_evaluate_models_general, build_and_train, predict_with_uncertainty)
