"""
Different (variations on) loss functions.
"""
import numpy as np
import tensorflow as tf
from tensorflow import reduce_mean, square
from tensorflow.nn import softplus
from tensorflow.math import log
from tensorflow.keras.utils import register_keras_serializable  # Enables saving/loading models with this custom loss function

### HELPER FUNCTIONS
@register_keras_serializable()
def estimates_to_mean_var(y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Split out [mu1, mu2, ..., muN, var1, var2, ..., varN] into separate arrays
    with softplus regularisation.
    """
    N = y_pred.shape[1] // 2
    mean = y_pred[:, :N]
    var = softplus(y_pred[:, N:])

    return mean, var

### LOSS FUNCTIONS
@register_keras_serializable()
def nll_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Negative Log Likelihood (NLL) loss function.
    `y_true` contains N reference values per row.
    `y_pred` contains N predicted mean values, followed by N predicted variances, per row:
        [mean1, mean2, ..., meanN, var1, var2, ..., varN]
    """
    mean, var = estimates_to_mean_var(y_pred)

    return reduce_mean(0.5 * (log(var) + (square(y_true - mean) / var) + log(2 * np.pi)))
