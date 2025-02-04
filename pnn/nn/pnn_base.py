"""
Base class for PNNs, to be imported elsewhere.
"""
from pathlib import Path
from shutil import rmtree
from time import time
from typing import Any, Iterable, Optional, Self
from zipfile import ZipFile, ZIP_DEFLATED

import dill as pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model, load_model
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from .. import constants as c, data as d


### SAVING/LOADING
def timestamp() -> str:
    """
    Create a timestamp based on time().
    """
    timestamp = str(time())
    timestamp = timestamp.replace(".", "")
    return timestamp


def dump_into_zipfile(zipfile: ZipFile, filename: str, data, **kwargs) -> None:
    """
    Pickle data (any object) and write them into an open zipfile.
    """
    zipfile.writestr(filename, pickle.dumps(data), compress_type=ZIP_DEFLATED, **kwargs)


def load_dump(filename: Path | str) -> Any:
    """
    Load pickled data from a file.
    """
    with open(filename, mode="rb") as file:
        data = pickle.load(file)
    return data


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
    def __init__(self, model: Model | Iterable[Model], *,
                 scaler_X: Optional[RobustScaler]=None, scaler_y: Optional[MinMaxScaler]=None) -> None:
        """
        Initialisation with just a Model so it can be used for training new models or loading from file.
        Scalers for X and y are optional.
        """
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y


    def __repr__(self) -> str:
        return f"{self.name}: {self.model}"


    ### DATA RESCALING
    def scale_X(self, X: np.ndarray) -> np.ndarray:
        """
        Rescale X using the included scaler.
        """
        assert self.scaler_X is not None, f"Model `{self}` does not have a rescaler for X."
        assert (self_shape := self.scaler_X.center_.shape[0]) == (data_shape := X.shape[-1]), f"Data ({data_shape}) and scaler ({self_shape}) have incompatible shapes."

        return self.scaler_X.transform(X)


    def scale_y(self, y: np.ndarray) -> np.ndarray:
        """
        Rescale log(y) using the included scaler.
        """
        assert self.scaler_y is not None, f"Model `{self}` does not have a rescaler for y."
        assert (self_shape := self.scaler_y.data_range_.shape[0]) == (data_shape := y.shape[-1]), f"Data ({data_shape}) and scaler ({self_shape}) have incompatible shapes."

        return d.scale_y(self.scaler_y, y)


    def inverse_scale_y(self, mean: np.ndarray, variance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Rescale network outputs using the included scaler.
        """
        assert (self_shape := self.scaler_y.data_range_.shape[0]) == (mean_shape := mean.shape[-1]) == (var_shape := variance.shape[-1]), f"Mean ({mean_shape}), variance ({var_shape}), and scaler ({self_shape}) have incompatible shapes."

        return d.inverse_scale_y(self.scaler_y, mean, variance)


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
        # Data scaling
        X_train = self.scale_X(X_train) if self.scaler_X is not None else X_train
        y_train = self.scale_y(y_train) if self.scaler_y is not None else y_train

        # Setup
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        early_stopping = EarlyStopping(monitor="val_loss", patience=80, verbose=1, mode="min", restore_best_weights=True)

        # Training
        self.model.compile(optimizer=optimizer, loss=nll_loss)
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[early_stopping], **kwargs)


    @classmethod
    def build_and_train(cls, X_train: np.ndarray, y_train: np.ndarray, *,
                        scaler_X: Optional[RobustScaler]=None, scaler_y: Optional[MinMaxScaler]=None,
                        build_kwargs: Optional[dict]={}, **train_kwargs) -> Self:
        """
        Build and train a model on the provided X and y data, with early stopping.
        Convenience function combining the build and train functions.
        """
        # Setup
        input_shape = X_train.shape[1:]
        output_size = y_train.shape[-1]

        newpnn = cls.build(input_shape, output_size, scaler_X=scaler_X, scaler_y=scaler_y, **build_kwargs)
        newpnn.train(X_train, y_train, **train_kwargs)

        return newpnn


    ### SAVING / LOADING
    def _save_model(self, filename: Path | str, *args, **kwargs) -> Path:
        """
        Save the underlying model; can be overridden while maintaining general save function.
        Returns its Path so that any modifications are passed on.
        """
        filename = Path(filename)
        self.model.save(filename, *args, **kwargs)
        return filename


    def save(self, filename: Path | str, **kwargs) -> None:
        """
        Save the full model into a ZIP file, including scalers (can be None).
        """
        filename = Path(filename)

        # Save underlying
        model_filename = filename.parent/f"temp_{timestamp()}.keras"  # Temporary
        model_filename = self._save_model(model_filename)  # May have been modified

        # Save components into ZIP file
        with ZipFile(filename, mode="w") as zipfile:
            # Move underlying model file into ZIP folder
            model_filename_zip = model_filename.with_stem("model").name  # Maintain extension, drop parents
            zipfile.write(model_filename, model_filename_zip, compress_type=ZIP_DEFLATED)

            # Save scalers
            dump_into_zipfile(zipfile, "X.scaler", self.scaler_X)
            dump_into_zipfile(zipfile, "y.scaler", self.scaler_y)

        # Delete temporary files
        model_filename.unlink()


    @staticmethod
    def _load_model(filename: Path | str, *args, **kwargs) -> Model | Iterable[Model]:
        """
        Load the underlying model; can be overridden while maintaining general load function.
        Assumes the model has already been unzipped (but can itself be another zip file).
        """
        return load_model(filename, *args, **kwargs)


    @classmethod
    def load(cls, filename: Path | str, *args, **kwargs) -> Self:
        """
        Load a model from a ZIP file, including scalers (can be None).
        """
        filename = Path(filename)

        # Find ZIP file, create temporary folder
        temp_folder = filename.parent/f"temp_{timestamp()}/"
        temp_folder.mkdir()

        # Extract, find, and load individual files
        with ZipFile(filename, mode="r") as zipfile:
            zipfile.extractall(temp_folder)

        # Load unpacked files
        model = cls._load_model(temp_folder/"model.keras")
        scaler_X = load_dump(temp_folder/"X.scaler")
        scaler_y = load_dump(temp_folder/"y.scaler")

        # Delete temporary files
        rmtree(temp_folder)

        return cls(model, scaler_X=scaler_X, scaler_y=scaler_y)


    ### APPLICATION
    def _predict_samples(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Use the model to predict y values for X.
        """
        return self.model.predict(X, **kwargs)


    def predict_with_uncertainty(self, X: np.ndarray, **predict_kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Use the given model to predict y values for given X values, including the rescaling back to regular units.
        """
        # Data scaling
        X = self.scale_X(X) if self.scaler_X is not None else X

        # Generate predictions in scaled space
        pred_samples = self._predict_samples(X, **predict_kwargs)

        # Separate predicted means, predicted variances
        N = pred_samples.shape[-1] // 2  # Number of predicted mean parameters
        mean_predictions_scaled = pred_samples[..., :N]
        raw_variances_scaled = pred_samples[..., N:]
        variance_predictions_scaled = tf.nn.softplus(raw_variances_scaled)

        # Convert from scaled space to real units
        if self.scaler_y is not None:
            mean_predictions, variance_predictions = self.inverse_scale_y(mean_predictions_scaled, variance_predictions_scaled)
        else:
            mean_predictions, variance_predictions = mean_predictions_scaled, variance_predictions_scaled

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
