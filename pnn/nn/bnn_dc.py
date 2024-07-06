"""
Bayesian Neural Network with DropConnect (BNN DC).
"""
from functools import partial
from typing import Iterable, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, History
from tensorflow.keras.layers import Dense, Dropout, Layer
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2

from .common import calculate_metrics, inverse_scale_y, nll_loss, predict_with_dropout
from .common import _train_general, _build_and_train_general, _predict_with_uncertainty_general, _train_and_evaluate_models_general
from .. import constants as c

### ARCHITECTURE
class DropConnectDense(Layer):
    """
    Dense layer with DropConnect implemented.
    """
    def __init__(self, units, *, activation=None, dropout_rate=0.25, l2_reg=1e-3, **kwargs):
        super(DropConnectDense, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.dropout_rate = dropout_rate
        self.kernel_regularizer = l2(l2_reg)

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer="random_normal",
                                 regularizer=self.kernel_regularizer,
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, inputs, training=None):
        if training:
            # Generate dropout mask for weights
            dropout_mask = tf.nn.dropout(tf.ones_like(self.w), rate=self.dropout_rate)
            w = self.w * dropout_mask
        else:
            w = self.w

        output = tf.matmul(inputs, w) + self.b
        if self.activation is not None:
            output = self.activation(output)
        return output


def build(input_shape: tuple, *, output_size: int=6,
          hidden_units: int=100, n_layers: int=5, dropout_rate: float=0.25, l2_reg: float=1e-3, activation="relu") -> Model:
    """
    Construct a BNN with DC based on the input parameters.
    """
    model = Sequential()

    # Add the first layer with input shape
    model.add(DropConnectDense(hidden_units, activation=activation, dropout_rate=dropout_rate, l2_reg=l2_reg, input_shape=input_shape, name="drop_connect_dense_1"))

    # Add additional layers
    for i in range(1, n_layers):
        model.add(DropConnectDense(hidden_units, activation=activation, dropout_rate=dropout_rate, l2_reg=l2_reg, name=f"drop_connect_dense_{i+1}"))

    # Output layer: Adjust for 6 means and 6 variances (12 outputs in total)
    model.add(Dense(output_size * 2, activation="linear", name="output_dense"))  # Naming the output layer explicitly

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
