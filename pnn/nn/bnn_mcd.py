"""
Bayesian Neural Network with Monte Carlo Dropout (RNN MCD).
"""
from functools import partial
from typing import Iterable, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, History
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2

from .common import calculate_metrics, inverse_scale_y, nll_loss, predict_with_dropout
from .common import _train_general, _build_and_train_general, _predict_with_uncertainty_general, _train_and_evaluate_models_general
from .. import constants as c

### ARCHITECTURE
def build(input_shape: tuple, *, output_size: int=6,
          hidden_units: int=50, n_layers: int=5, dropout_rate: float=0.25, l2_reg: float=1e-3, activation="tanh") -> Model:
    """
    Construct a BNN with MCD based on the input parameters.
    """
    model = Sequential()

    # Add the first layer with input shape
    model.add(Dense(hidden_units, activation=activation, input_shape=input_shape, kernel_regularizer=l2(l2_reg)))

    # Add additional layers with Dropout layers between them
    for i in range(n_layers - 1):
        model.add(Dropout(dropout_rate))
        model.add(Dense(hidden_units, activation=activation, kernel_regularizer=l2(l2_reg)))

    # Output layer: Adjust for 6 means and 6 variances (12 outputs in total)
    model.add(Dropout(dropout_rate))
    model.add(Dense(output_size * 2, activation="linear"))

    return model


### TRAINING
train = partial(_train_general, epochs=1000, batch_size=32)
build_and_train = partial(_build_and_train_general, build, train)


### APPLICATION
def predict_samples(model: Model, X: np.ndarray, *, n_samples: int=100) -> np.ndarray:
    """
    Predict y values for given X values using the BNN-MCD.
    """
    pred_samples = [predict_with_dropout(model, X, enable_dropout=True).numpy() for _ in range(n_samples)]
    pred_samples = np.array(pred_samples)
    return pred_samples


predict_with_uncertainty = partial(_predict_with_uncertainty_general, predict_samples)
train_and_evaluate_models = partial(_train_and_evaluate_models_general, build_and_train, predict_with_uncertainty)
