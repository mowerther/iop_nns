"""
Recurrent Neural Network with Gated Recurrent Units and Monte Carlo Dropout (RNN MCD).
"""
from typing import Optional, Self

import numpy as np
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

from .pnn_base import DropoutPNN
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
class RNN_MCD(DropoutPNN):
    ### CONFIGURATION
    name = c.rnn

    ### CREATION
    @classmethod
    def build(cls, input_shape: tuple, output_size: int, *,
              hidden_units: int=100, n_layers: int=5, dropout_rate: float=0.25, l2_reg: float=1e-3, activation="tanh") -> Self:
        """
        Construct an RNN with MCD based on the input hyperparameters.
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

        return cls(model)


    @classmethod
    def build_and_train(cls, X_train: np.ndarray, y_train: np.ndarray, *args, batch_size: int=32, **kwargs) -> Self:
        """
        Build and train a model on the provided X and y data, with early stopping.
        Reshapes the data for the RNN first.
        Note: trains a lot slower with the same hyperparameters; increasing the batch_size makes it faster, but we keep it the same here for consistency in the results.
        """
        X_train_reshaped = reshape_data(X_train)
        return super().build_and_train(X_train_reshaped, y_train, *args, batch_size=batch_size, **kwargs)
