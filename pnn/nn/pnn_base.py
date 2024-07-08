"""
Base class for PNNs, to be imported elsewhere.
"""
from typing import Iterable, Optional, Self

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, History
from tensorflow.keras.models import Model, load_model
from sklearn.preprocessing import MinMaxScaler

from .. import constants as c, metrics as m, modeloutput as mo
from ..data import inverse_scale_y


### LOSS FUNCTIONS
@tf.keras.utils.register_keras_serializable()  # Enables saving/loading models with this custom loss function
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


### MAIN PNN CLASS
class BasePNN:
    ### CONFIGURATION
    name = "BasePNN"
    def __init__(self, model: Model | Iterable[Model]) -> None:
        """
        Initialisation with just a Model so it can be used for training new models or loading from file.
        """
        self.model = model


    def __repr__(self) -> str:
        return f"{self.name}: {self.model}"


    ### CREATION
    @classmethod
    def build(cls, input_shape: tuple, output_size: int, *args, **kwargs) -> Self:
        """
        Build the underlying model.
        To be overridden by subclasses.
        """
        return NotImplemented


    def train(self, X_train: np.ndarray, y_train: np.ndarray, *,
              epochs: int=1000, batch_size: int=32, learning_rate: float=0.001, validation_split: float=0.1, **kwargs) -> None:
        """
        Train on the provided X and y data, with early stopping.
        **kwargs are passed to self.model.fit.
        """
        # Setup
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        early_stopping = EarlyStopping(monitor="val_loss", patience=80, verbose=1, mode="min", restore_best_weights=True)

        # Training
        self.model.compile(optimizer=optimizer, loss=nll_loss)
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[early_stopping], **kwargs)


    @classmethod
    def build_and_train(cls, X_train: np.ndarray, y_train: np.ndarray, *,
                        build_kwargs: Optional[dict]={}, **train_kwargs) -> Self:
        """
        Build and train a model on the provided X and y data, with early stopping.
        Convenience function combining the build and train functions.
        """
        # Setup
        input_shape = X_train.shape[1:]
        output_size = y_train.shape[-1]

        newpnn = cls.build(input_shape, output_size, **build_kwargs)
        newpnn.train(X_train, y_train, **train_kwargs)

        return newpnn


    ### SAVING / LOADING
    def save(self, *args, **kwargs) -> None:
        self.model.save(*args, **kwargs)


    @classmethod
    def load(cls, *args, **kwargs) -> Self:
        model = load_model(*args, **kwargs)
        return cls(model)


    ### APPLICATION
    def _predict_samples(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Use the model to predict y values for X.
        """
        return self.model.predict(X, **kwargs)


    def predict_with_uncertainty(self, X: np.ndarray, scaler_y: MinMaxScaler, **predict_kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Use the given model to predict y values for given X values, including the rescaling back to regular units.
        """
        # Generate predictions in scaled space
        pred_samples = self._predict_samples(X, **predict_kwargs)

        # Separate predicted means, predicted variances
        N = scaler_y.n_features_in_  # Number of predicted mean parameters
        mean_predictions_scaled = pred_samples[..., :N]
        raw_variances_scaled = pred_samples[..., N:]
        variance_predictions_scaled = tf.nn.softplus(raw_variances_scaled)

        # Convert from scaled space to real units
        mean_predictions, variance_predictions = inverse_scale_y(mean_predictions_scaled, variance_predictions_scaled, scaler_y)

        # Calculate aleatoric and epistemic variance in the original space
        aleatoric_variance = np.mean(variance_predictions, axis=0)
        epistemic_variance = np.var(mean_predictions, axis=0)
        total_variance = aleatoric_variance + epistemic_variance

        mean_predictions = np.mean(mean_predictions, axis=0)  # Average over n_samples

        return mean_predictions, total_variance, aleatoric_variance, epistemic_variance


### DROPOUT/DROPCONNECT VERSION, FOR CONVENIENCE
class DropoutPNN(BasePNN):
    ### APPLICATION
    @tf.function  # 4x Speed-up
    def _predict_with_dropout(self, X: np.ndarray):
        return self.model(X, training=True)  # `training=True` just turns the dropout on, it does not affect the model parameters


    def _predict_samples(self, X: np.ndarray, *, n_samples: int=100) -> np.ndarray:
        """
        Predict y values for given X values using dropout/dropconnect.
        """
        pred_samples = [self._predict_with_dropout(X).numpy() for _ in range(n_samples)]
        pred_samples = np.array(pred_samples)
        return pred_samples
