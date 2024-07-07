"""
Bayesian Neural Network with Monte Carlo Dropout (BNN MCD).
"""
from typing import Self

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

from .pnn_base import DropoutPNN
from .. import constants as c


class BNN_MCD(DropoutPNN):
    ### CONFIGURATION
    name = c.bnn_mcd

    ### CREATION
    @classmethod
    def build(cls, input_shape: tuple, output_size: int, *,
              hidden_units: int=100, n_layers: int=5, dropout_rate: float=0.25, l2_reg: float=1e-3, activation="relu") -> Self:
        """
        Construct a BNN with MCD based on the input hyperparameters.
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

        return cls(model)
