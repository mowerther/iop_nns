"""
Bayesian Neural Network with DropConnect (BNN DC).
"""
from typing import Self

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

from .pnn_base import DropoutPNN
from .. import constants as c

### ARCHITECTURE
class DropConnectDense(Layer):
    """
    Dense layer with DropConnect implemented.
    """
    def __init__(self, units, *, activation=None, dropout_rate=0.25, l2_reg=1e-3, **kwargs):
        super().__init__(**kwargs)
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


class BNN_DC(DropoutPNN):
    ### CONFIGURATION
    name = c.bnn_dc

    ### CREATION
    @classmethod
    def build(cls, input_shape: tuple, output_size: int, *,
              hidden_units: int=100, n_layers: int=5, dropout_rate: float=0.25, l2_reg: float=1e-3, activation="relu") -> Self:
        """
        Construct a BNN with DC based on the input hyperparameters.
        """
        model = Sequential()
        model.add(Input(shape=input_shape))

        # Add additional layers
        for i in range(n_layers):
            model.add(DropConnectDense(hidden_units, activation=activation, dropout_rate=dropout_rate, l2_reg=l2_reg, name=f"drop_connect_dense_{i+1}"))

        # Output layer: Adjust for 6 means and 6 variances (12 outputs in total)
        model.add(Dense(output_size * 2, activation="linear", name="output_dense"))  # Naming the output layer explicitly

        return cls(model)
