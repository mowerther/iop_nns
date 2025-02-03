"""
Ensemble neural networks (ENS-NN).
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Self
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

from .pnn_base import BasePNN
from .. import constants as c


### HELPER FUNCTIONS
def _clean_up_folder_name(folder: Path | str) -> Path:
    """
    Remove a ".keras" suffix, which the other models require in their filenames.
    """
    folder = Path(folder)
    if folder.suffix == ".keras":
        folder = folder.with_suffix("")
    return folder


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
    def save(self, folder: Path | str, **kwargs) -> None:
        """
        Save the separate NNs in one folder (created if necessary).
        """
        # Create folder
        folder = _clean_up_folder_name(folder)
        folder.mkdir(parents=True, exist_ok=True)

        # Save individual models
        for i, m in enumerate(self.model):
            filename = folder / f"model_{i}.keras"
            m.save(filename)


    @classmethod
    def load(cls, folder: Path | str, **kwargs) -> Self:
        """
        Load the separate NNs from file, then combine them into an ensemble.
        """
        folder = _clean_up_folder_name(folder)
        filenames = folder.glob("*.keras")

        models = [_SimpleNN.load(fn) for fn in filenames]
        assert len(models) >= 1, f"Could not find any models in folder `{folder}`"

        return cls(models)


    ### APPLICATION
    def _predict_samples(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Use the model to predict y values for X.
        """
        pred_samples = [m._predict_samples(X, verbose=0) for m in self.model]
        pred_samples = np.array(pred_samples)
        return pred_samples
