"""
PNN with recalibration.
"""
from pathlib import Path
from typing import Callable, Iterable, Self
from zipfile import ZipFile

import dill as pickle
import numpy as np

from .pnn_base import BasePNN, dump_into_zipfile
from ..recalibration import apply_recalibration, fit_recalibration_functions

RECALIBRATOR_FILENAME = "recalibrators.funcs"


class RecalibratedPNN:
    """
    PNN with recalibrators attached.
    """
    ### CONFIGURATION
    def __init__(self, model: BasePNN, recalibrators: Iterable[Callable]) -> None:
        self.model = model
        self.scaler_X = model.scaler_X
        self.scaler_y = model.scaler_y
        self.recalibrators = recalibrators

    @property
    def name(self) -> str:
        return f"{self.model.name} (recalibrated)"

    def __repr__(self) -> str:
        return f"Recalibrated {self.model}"


    ### CREATION
    @classmethod
    def recalibrate_pnn(cls, model: BasePNN, X: np.ndarray, y: np.ndarray) -> Self:
        """
        Take an existing PNN and train recalibration functions.
        """
        mean_predictions, total_variance, *_ = model.predict_with_uncertainty(X)
        recalibrators = fit_recalibration_functions(y, mean_predictions, total_variance)
        return cls(model, recalibrators)


    ### SAVING / LOADING
    def save(self, filename: Path | str, *args, **kwargs) -> None:
        filename = Path(filename)

        # Save model normally
        self.model.save(filename, *args, **kwargs)

        # Pickle and save recalibrators
        with ZipFile(filename, mode="a") as zipfile:
            dump_into_zipfile(zipfile, RECALIBRATOR_FILENAME, self.recalibrators)


    @classmethod
    def load(cls, filename: Path | str, PNN_Class: type, *args, **kwargs) -> Self:
        filename = Path(filename)

        # Load model
        model = PNN_Class.load(filename)

        # Load recalibrators
        with ZipFile(filename, mode="r") as zipfile:
            with zipfile.open(RECALIBRATOR_FILENAME, mode="r") as recal_file:
                recalibrators = pickle.load(recal_file)

        return cls(model, recalibrators)


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


    def predict_with_uncertainty(self, X: np.ndarray, **predict_kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Use the PNN to predict y values for given X values, including the rescaling back to regular units.
        Then use the recalibrators to adjust the variances.
        """
        # Generate uncalibrated predictions
        mean_predictions, total_variance_uncalibrated, aleatoric_variance, epistemic_variance = self.model.predict_with_uncertainty(X, **predict_kwargs)

        # Apply recalibration
        total_variance = self.recalibrate(mean_predictions, total_variance_uncalibrated)

        return mean_predictions, total_variance, aleatoric_variance, epistemic_variance



### CONVENIENCE FUNCTIONS
def recalibrate_pnn(models: BasePNN, X: np.ndarray, y: np.ndarray) -> list[RecalibratedPNN] | RecalibratedPNN:
    """
    Recalibrate existing PNNs.
    Returns a list if multiple models are provided; returns a single object if only one.
    """
    SINGLE = (not isinstance(models, Iterable))
    if SINGLE:
        models = [models]

    new_models = [RecalibratedPNN.recalibrate_pnn(model, X, y) for model in models]

    if SINGLE:
        new_models = new_models[0]

    return new_models
