"""
Different (variations on) loss functions.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable

### LOSS FUNCTIONS
@register_keras_serializable()  # Enables saving/loading models with this custom loss function
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
