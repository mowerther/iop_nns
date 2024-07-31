"""
PNN with recalibration.
"""
import pickle
from pathlib import Path
from typing import Callable, Iterable, Optional, Self

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .pnn_base import BasePNN
from ..recalibration import apply_recalibration, fit_recalibration_functions


_PICKLE_SUFFIX = ".funcs"


class RecalibratedPNN:
    """
    PNN with recalibrators attached.
    """
    ### CONFIGURATION
    def __init__(self, model: BasePNN, recalibrators: Iterable[Callable]) -> None:
        self.model = model
        self.recalibrators = recalibrators

    @property
    def name(self) -> str:
        return f"{self.model.name} (recalibrated)"

    def __repr__(self) -> str:
        return f"Recalibrated {self.model}"


    ### CREATION
    @classmethod
    def recalibrate_pnn(cls, model: BasePNN, X: np.ndarray, y: np.ndarray, scaler_y: MinMaxScaler) -> Self:
        """
        Take an existing PNN and train recalibration functions.
        """
        mean_predictions, total_variance, *_ = model.predict_with_uncertainty(X, scaler_y)
        recalibrators = fit_recalibration_functions(y, mean_predictions, total_variance)
        return cls(model, recalibrators)


    ### SAVING / LOADING
    def save(self, filename: Path | str, *args, **kwargs) -> None:
        # Save model normally
        self.model.save(filename, *args, **kwargs)

        # Pickle and save recalibrators
        filename = Path(filename)
        filename_funcs = filename.with_suffix(_PICKLE_SUFFIX)
        with open(filename_funcs, mode="w") as file:
            pass

        print("Note: saving not fully implemented yet.")


    @classmethod
    def load(cls, *args, **kwargs) -> Self:
        return NotImplemented


    ### APPLICATION
    def recalibrate(self, predicted_mean: np.ndarray, total_variance: np.ndarray) -> np.ndarray:
        """
        For each output (column in the input arrays), apply the respective recalibration function.
        """
        # Check inputs
        N_functions, N_means, N_variances = len(self.recalibrators), predicted_mean.shape[-1], total_variance.shape[-1]
        assert N_functions == N_means == N_variances, f"Mismatch in dimensions between recalibration functions ({N_functions}) and mean {predicted_mean.shape} and variance {total_variance.shape}."

        # Apply
        return apply_recalibration(self.recalibrators, predicted_mean, total_variance)


    def predict_with_uncertainty(self, X: np.ndarray, scaler_y: MinMaxScaler, **predict_kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Use the PNN to predict y values for given X values, including the rescaling back to regular units.
        Then use the recalibrators to adjust the variances.
        """
        # Generate uncalibrated predictions
        mean_predictions, total_variance_uncalibrated, aleatoric_variance, epistemic_variance = self.model.predict_with_uncertainty(X, scaler_y, **predict_kwargs)

        # Apply recalibration
        total_variance = self.recalibrate(mean_predictions, total_variance_uncalibrated)

        return mean_predictions, total_variance, aleatoric_variance, epistemic_variance



### CONVENIENCE FUNCTIONS
def recalibrate_pnn(models: BasePNN, X: np.ndarray, y: np.ndarray, scaler_y: MinMaxScaler) -> list[RecalibratedPNN] | RecalibratedPNN:
    """
    Recalibrate existing PNNs.
    Returns a list if multiple models are provided; returns a single object if only one.
    """
    SINGLE = (not isinstance(models, Iterable))
    if SINGLE:
        models = [models]

    new_models = [RecalibratedPNN.recalibrate_pnn(model, X, y, scaler_y) for model in models]

    if SINGLE:
        new_models = new_models[0]

    return new_models
