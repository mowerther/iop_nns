"""
Recurrent Neural Network with Gated Recurrent Units and Monte Carlo Dropout (RNN MCD).
"""
from typing import Iterable

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, History
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler

from .common import calculate_metrics, inverse_scale_y, nll_loss
from .. import constants as c


### DATA HANDLING
def reshape_data(X_train: np.ndarray, X_test: np.ndarray, *, n_features_per_timestep: int=1) -> tuple[np.ndarray, np.ndarray]:
    """
    Reshape data for the RNN, adding a timestep axis.
    """
    # Calculate the number of wavelength steps and features
    n_samples_train, n_features = X_train.shape
    n_samples_test, _ = X_test.shape  # We already know the features count

    n_timesteps = n_features // n_features_per_timestep  # Here: One wavelength per step

    # Reshape data
    X_train_reshaped = X_train.reshape((n_samples_train, n_timesteps, n_features_per_timestep))
    X_test_reshaped = X_test.reshape((n_samples_test, n_timesteps, n_features_per_timestep))

    return X_train_reshaped, X_test_reshaped


### ARCHITECTURE
def build_rnn_mcd(input_shape: tuple, *, output_size: int=6,
                  hidden_units: int=100, n_layers: int=5, dropout_rate: float=0.25, l2_reg: float=1e-3, activation="tanh") -> Model:
    """
    Construct an RNN with MCD based on the input parameters.
    To do: use functools.partial for GRU?
    To do: relu or tanh?
    """
    model = Sequential()
    model.add(Input(shape=input_shape))

    # Adding the first recurrent layer with input shape
    model.add(GRU(hidden_units, return_sequences=(n_layers > 1),
                  activation=activation, kernel_regularizer=l2(l2_reg)))

    # Adding additional GRU layers if n_layers > 1, with Dropout layers between them
    for i in range(1, n_layers):
        model.add(Dropout(dropout_rate))
        model.add(GRU(hidden_units, return_sequences=(i < n_layers-1),
                      activation=activation, kernel_regularizer=l2(l2_reg)))

    # Output layer: Adjust for 6 means and 6 variances (12 outputs in total)
    model.add(Dense(output_size * 2, activation="linear"))

    return model


### TRAINING
def train_rnn_mcd(model: Model, X_train: np.ndarray, y_train: np.ndarray, *,
                  epochs: int=1000, batch_size: int=512, learning_rate: float=0.001, validation_split: float=0.1) -> tuple[Model, History]:
    """
    Train an RNN Model on the provided X and y data, with early stopping.
    """
    # Setup
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    early_stopping = EarlyStopping(monitor="val_loss", patience=80, verbose=1, mode="min", restore_best_weights=True)

    # Training
    model.compile(optimizer=optimizer, loss=nll_loss)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[early_stopping])

    return model, history


def build_and_train_rnn_mcd(X_train: np.ndarray, y_train: np.ndarray) -> Model:
    """
    Build and train an RNN model on the provided X and y data, with early stopping.
    Convenience function combining the build and train functions.
    """
    # Setup
    input_shape = X_train.shape[1:]
    output_size = y_train.shape[-1]

    model = build_rnn_mcd(input_shape, output_size=output_size)
    model, history = train_rnn_mcd(model, X_train, y_train)

    return model


### APPLICATION
@tf.function  # 4x Speed-up
def predict_with_dropout(model, inputs, enable_dropout=True):
    return model(inputs, training=enable_dropout)  # `training=True` just turns the dropout on, it does not affect the model parameters


def predict_with_uncertainty(model: Model, X: np.ndarray, scaler_y: MinMaxScaler, *, n_samples=100):
    """
    Use the given model to predict y values for given X values, including the rescaling back to regular units.
    """
    # Generate predictions in scaled space
    pred_samples = [predict_with_dropout(model, X, enable_dropout=True).numpy() for _ in range(n_samples)]
    pred_samples = np.array(pred_samples)

    N = scaler_y.n_features_in_  # Number of predicted values
    mean_predictions_scaled = pred_samples[..., :N]
    raw_variances_scaled = pred_samples[..., N:]
    variance_predictions_scaled = tf.nn.softplus(raw_variances_scaled)

    # Convert from scaled space to log space
    mean_predictions, variance_predictions = inverse_scale_y(mean_predictions_scaled, variance_predictions_scaled, scaler_y)

    # Calculate aleatoric and epistemic variance in the original space
    aleatoric_variance = np.mean(variance_predictions, axis=0)
    epistemic_variance = np.var(mean_predictions, axis=0)
    total_variance = aleatoric_variance + epistemic_variance
    std_devs = np.sqrt(total_variance)

    mean_predictions = np.mean(mean_predictions, axis=0)  # Average over n_samples

    return mean_predictions, total_variance, aleatoric_variance, epistemic_variance, std_devs


def train_and_evaluate_models(X_train: np.ndarray, y_train_scaled: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, scaler_y: MinMaxScaler, *,
                              n_models: int=10, n_samples: int=100, mdsa_columns: Iterable[str]=c.iops_443) -> tuple[Model, pd.DataFrame]:
    """
    Train and evaluate a model, `n_models` times, then pick the best one based on the MdSA from a comparison on the testing data.
    Returns the best model and a DataFrame with the metrics of all models, for comparison purposes.
    """
    all_models, all_metrics = [], []

    best_overall_model = None
    best_mdsa = np.inf

    for i in range(n_models):
        label = f"{i+1}/{n_models}"

        # Train model
        model = build_and_train_rnn_mcd(X_train, y_train_scaled)
        all_models.append(model)
        print(f"Model {label}: Finished training.")

        # Assess model
        mean_preds, total_var, aleatoric_var, epistemic_var, std_preds = predict_with_uncertainty(model, X_test, scaler_y, n_samples=n_samples)
        print(f"Model {label}: Finished prediction.")

        metrics_df = calculate_metrics(y_test, mean_preds)
        all_metrics.append(metrics_df)
        print(f"Model {label}: Calculated performance metrics.")

        evaluation_mdsa = metrics_df.loc[mdsa_columns, "MdSA"].mean()  # Average the MdSA of the specified variables
        if evaluation_mdsa < best_mdsa:
            best_mdsa = evaluation_mdsa
            best_overall_model = model
            print(f"Model {label} is the new best model (mean MdSA: {evaluation_mdsa:.0f}%).")

        print("\n\n")

    all_metrics = pd.concat({i+1: df for i, df in enumerate(all_metrics)}, names=["model", "variable"])

    return best_overall_model, all_metrics
