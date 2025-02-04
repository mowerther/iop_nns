"""
Ensemble neural networks (ENS-NN).
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Self
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

from .pnn_base import BasePNN, timestamp
from .. import constants as c


### SINGLE NN
class _SimpleNN(BasePNN):
    ### CONFIGURATION
    name = "_SimpleNN"

    ### CREATION
    @classmethod
    def build(cls, input_shape: tuple, output_size: int, *,
              hidden_units: int=100, n_layers: int=5, l2_reg: float=1e-3, activation="relu", **kwargs) -> Self:
        """
        Construct a single regular NN based on the input hyperparameters.
        """
        model = Sequential()
        model.add(Input(shape=input_shape))

        # Add layers
        for i in range(n_layers):
            model.add(Dense(hidden_units, activation=activation, kernel_regularizer=l2(l2_reg)))

        # Output layer: Adjust for 6 means and 6 variances (12 outputs in total)
        model.add(Dense(output_size * 2, activation="linear"))

        return cls(model, **kwargs)

    @classmethod
    def build_and_train(cls, index: int, X_train: np.ndarray, y_train: np.ndarray, *args, **kwargs) -> Self:
        """
        Build and train a model on the provided X and y data, with early stopping.
        Same as BasePNN training, but sets the random seed based on the index for reproducibility.
        """
        np.random.seed(index)
        tf.random.set_seed(index)
        return super().build_and_train(X_train, y_train, *args, **kwargs)


### ENSEMBLE
class Ensemble(BasePNN):
    ### CONFIGURATION
    name = c.ensemble

    ### CREATION
    def train(cls, *args, **kwargs):
        """
        Overridden, use build_and_train instead.
        """
        return NotImplemented


    @classmethod
    def build_and_train(cls, X_train: np.ndarray, y_train: np.ndarray, *, N: int=10, **kwargs) -> Self:
        """
        Build and train N models on the provided X and y data, with early stopping.
        **kwargs are passed to _SimpleNN.build_and_train.
        """
        models = []
        with ThreadPoolExecutor(max_workers=N) as executor:  # Run concurrently
            futures = [
                executor.submit(_SimpleNN.build_and_train, i, X_train, y_train, verbose=(i == 0), **kwargs)
                for i in range(N)
            ]
            for future in as_completed(futures):
                model = future.result()
                models.append(model)

        return cls(models)


    ### SAVING / LOADING
    def _save_model(self, filename: Path | str, **kwargs) -> None:
        """
        Save the separate NNs in one ZIP file.
        Creates a temporary file in the parent folder, timestamped to prevent overwriting unrelated files.
        """
        # Create folder, filename for temporary Keras files
        temp_filename = filename.parent/f"temp_{timestamp()}.keras"

        # Save individual models into ZIP file
        with ZipFile(filename, mode="w") as zipfile:
            for i, m in enumerate(self.model):
                # Save temporary file
                m.save(temp_filename)

                # Move into ZIP folder
                zipfile.write(temp_filename, f"model_{i}.keras", compress_type=ZIP_DEFLATED)

        # Delete temporary files
        temp_filename.unlink()


    @classmethod
    def load(cls, filename: Path | str, **kwargs) -> Self:
        """
        Load the separate NNs from file, then combine them into an ensemble.
        """
        # Find ZIP file, create temporary folder
        temp_folder = filename.parent/f"temp_{timestamp()}/"
        temp_folder.mkdir()

        # Extract, find, and load individual files
        with ZipFile(filename, mode="r") as zipfile:
            zipfile.extractall(temp_folder)

        model_filenames = list(temp_folder.glob("*.keras"))

        models = [_SimpleNN.load(fn) for fn in model_filenames]
        assert len(models) >= 1, f"Could not find any ensemble models in `{filename.absolute()}`"

        # Delete temporary files
        for mf in model_filenames:
            mf.unlink()
        temp_folder.rmdir()

        return cls(models)


    ### APPLICATION
    def _predict_samples(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Use the model to predict y values for X.
        """
        pred_samples = [m._predict_samples(X, verbose=0) for m in self.model]
        pred_samples = np.array(pred_samples)
        return pred_samples
