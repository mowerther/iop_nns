"""
Mixture Density Networks (MDN).
"""
from typing import Callable, Iterable, Self

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler

from .pnn_base import BasePNN
from .. import constants as c
from ..data import inverse_scale_y


### MATHS
def split_mdn_outputs(raw_predictions: tf.Tensor, num_components: int, num_targets: int) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Split the raw predictions into pi, mu and chol parameters.

    Args:
        raw_predictions (tf.Tensor): The raw output from the MDN model, with shape [batch_size, num_components * (1 + num_targets + num_targets * (num_targets + 1) // 2)].
        num_components (int): The number of mixture components (i.e., the number of Gaussian distributions in the mixture).
        num_targets (int): The dimensionality of the target space.

    Returns:
        pi (tf.Tensor): The mixing coefficients for each component, with shape [batch_size, num_components].
        mu (tf.Tensor): The means of the Gaussian components, with shape [batch_size, num_components, num_targets].
        chol_params (tf.Tensor): The Cholesky decomposition parameters for the covariance matrices of the Gaussian components, with shape [batch_size, num_components, num_targets * (num_targets + 1) // 2].

    The function operates as follows:

    1. Splits the `raw_predictions` tensor into three parts:
       - `pi`: Mixing coefficients of shape [batch_size, num_components].
       - `mu`: Means of shape [batch_size, num_components * num_targets].
       - `chol_params`: Cholesky decomposition parameters of shape [batch_size, num_components * num_targets * (num_targets + 1) // 2].
    2. Applies a softmax function to `pi` to ensure the mixing coefficients sum to 1.
    3. Reshapes `mu` to [batch_size, num_components, num_targets].
    4. Reshapes `chol_params` to [batch_size, num_components, num_targets * (num_targets + 1) // 2].
    """
    split_sizes = [num_components, num_components * num_targets, num_components * num_targets * (num_targets + 1) // 2]
    pi, mu, chol_params = tf.split(raw_predictions, num_or_size_splits=split_sizes, axis=-1)
    pi = tf.nn.softmax(pi, axis=-1)  # Ensure mixing coefficients sum to 1
    return pi, mu, chol_params


def mdn_loss_constructor(num_components: int, num_targets: int, *, epsilon: float=1e-6) -> Callable:
    """
    The loss function computes the negative log likelihood of the true data given the predicted mixture distribution,
    which consists of a specified number of multivariate normal components. The NN outputs are split into
    mixing coefficients, means, and Cholesky decomposition parameters for the covariance matrices.

    Args:
        num_components (int): The number of mixture components (e.g., the number of Gaussian distributions in the mixture).
        num_targets (int): The dimensionality of the target space (e.g., the number of variables being predicted).
        epsilon (float, optional): Small value to ensure stability when constructing covariance matrices.
                                   Default is 1e-6.

    Returns:
        function: A loss function that can be used in model compilation.

    The returned loss function operates as follows:

    1. Splits the predicted values (`y_pred`) into mixing coefficients (`pi`), means (`mu`), and Cholesky decomposition
       parameters (`chol_params`).
    2. Applies a softmax function to the mixing coefficients to ensure they sum to 1.
    3. Reshapes the means and Cholesky decomposition parameters to the appropriate dimensions.
    4. Constructs the Cholesky lower triangular matrices from the Cholesky parameters.
    5. Adds `epsilon` to the diagonal elements of the Cholesky matrices to ensure positive definiteness.
    6. Defines a mixture of multivariate normal distributions using the reshaped parameters.
    7. Computes and returns the negative log likelihood of the true y given the mixture distribution (mix).

    Note:
          `num_components * (1 + num_targets + num_targets * (num_targets + 1) // 2)`.
    """
    def loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # # Split y_pred into its components: mixing coefficients (pi), means (mu), and Cholesky decomposition parameters (chol_params)
        # split_sizes = [num_components, num_components * num_targets, num_components * num_targets * (num_targets + 1) // 2]
        # pi, mu, chol_params = tf.split(y_pred, num_or_size_splits=split_sizes, axis=-1)
        # pi = tf.nn.softmax(pi, axis=-1)  # Softmax to get mixing coefficients
        pi, mu, chol_params = split_mdn_outputs(y_pred, num_components, num_targets)

        # Reshape mu and chol_params to match the desired dimensions
        mu = tf.reshape(mu, [-1, num_components, num_targets])
        chol_params = tf.reshape(chol_params, [-1, num_components, num_targets * (num_targets + 1) // 2])

        # Create covariance matrices from the Cholesky parameters
        chol_matrices = tfp.math.fill_triangular(chol_params)
        # Ensure positive definiteness of covariance matrices
        chol_matrices += epsilon * tf.linalg.eye(num_targets, batch_shape=[tf.shape(y_pred)[0], num_components])

        # Define the mixture of Multivariate Normal distributions
        components = tfp.distributions.MultivariateNormalTriL(loc=mu, scale_tril=chol_matrices)
        mixture = tfp.distributions.Categorical(probs=pi)
        mix = tfp.distributions.MixtureSameFamily(mixture_distribution=mixture, components_distribution=components)

        # Use mix: compute the log likelihood of the data given the model
        likelihood = mix.log_prob(y_true)

        # Return the negative log likelihood as the loss
        return -tf.reduce_mean(likelihood)

    return loss


### DEFINE MDN
class MDN(BasePNN):
    ### CONFIGURATION
    name = c.mdn

    def __init__(self, model: Model | Iterable[Model]) -> None:
        """
        Initialisation with just a Model so it can be used for training new models or loading from file.
        """
        self.model = model

    ### CREATION
    @classmethod
    def build(cls, input_shape: tuple, output_size: int, *,
              n_mix: int=5, hidden_layers: Iterable[int]=5*[100], lr: float=1e-3, l2_reg=1e-3, activation="relu",
              dropout=False, dropout_rate: float=0.25) -> Self:
        """
        Build the MDN.

        Args:
            input_shape (tuple): The shape of the input data.
            output_size (int): The dimensionality of the target space.
            n_mix (int): The number of mixture components (default is 5).
            hidden (list): A list of integers specifying the number of units in each hidden layer (default is [100, 100, 100, 100, 100]).
            lr (float): The learning rate for the optimizer (default is 1e-3).
            l2_reg (float): The L2 regularization factor (default is 1e-3).
            activation (str): The activation function to use in the hidden layers (default is "relu").
            dropout (bool): Use dropout in training (never in application/testing).
            dropout_rate (float): Dropout rate (0--1) if `dropout` is True.

        Returns:
            MDN(Model): ready for training.
        """
        inputs = Input(shape=input_shape)
        x = inputs

        for units in hidden_layers:
            x = Dense(units, activation=activation, kernel_regularizer=l2(l2_reg))(x)
            if dropout:
                x = Dropout(dropout_rate)(x)

        pi = Dense(n_mix, activation="softmax", name="pi")(x)
        mu = Dense(n_mix * output_size, activation=None, name="mu")(x)

        # Calculate the size for Cholesky decomposition parameters
        chol_size = n_mix * output_size * (output_size + 1) // 2
        chol_params = Dense(chol_size, activation=None, name="chol")(x)

        outputs = Concatenate(axis=-1)([pi, mu, chol_params])

        model = Model(inputs=inputs, outputs=outputs)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # Pass the correct parameters to the custom MDN loss function
        mdn_loss = mdn_loss_constructor(n_mix, output_size)
        model.compile(optimizer=optimizer, loss=mdn_loss)
        return cls(model)


    @property
    def n_mix(self) -> int:
        return self.model.get_layer("pi").output.shape[-1]

    @property
    def output_size(self) -> int:
        return self.model.get_layer("mu").output.shape[-1] // self.n_mix


    def train(self, X_train: np.ndarray, y_train: np.ndarray, *,
              epochs: int=1000, batch_size: int=32, learning_rate: float=0.001, validation_split: float=0.1, **kwargs) -> None:
        """
        Train on the provided X and y data, with early stopping.
        **kwargs are passed to self.model.fit.
        """
        # Setup
        early_stopping = EarlyStopping(monitor="val_loss", patience=80, verbose=1, mode="min", restore_best_weights=True)

        # Training
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[early_stopping], **kwargs)


    ### APPLICATION
    def predict_with_uncertainty(self, X: np.ndarray, scaler_y: MinMaxScaler, **predict_kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Use the given model to predict y values for given X values, including the rescaling back to regular units.
        """
        # Generate MDN predictions
        raw_predictions = self._predict_samples(X, **predict_kwargs)
        pi, mu, chol_params = split_mdn_outputs(raw_predictions, self.n_mix, self.output_size)
        pi = tf.reshape(pi, [-1, self.n_mix, 1])
        mu = tf.reshape(mu, [-1, self.n_mix, self.output_size])

        # Calculate the mean predictions (expected means)
        mean_predictions_scaled = tf.reduce_sum(pi * mu, axis=1)  # mean_predictions shape: [batch_size, output_size]

        # Reshape chol_params and calculate Sigma
        chol_elements = self.output_size * (self.output_size + 1) // 2
        chol_matrices = tf.reshape(chol_params, [-1, self.n_mix, chol_elements])
        L = tfp.math.fill_triangular(chol_matrices)
        Sigma = tf.matmul(L, L, transpose_b=True)  # [batch_size, n_mix, output_size, output_size]

        # Aleatoric variance
        variances = tf.linalg.diag_part(Sigma)  # [batch_size, n_mix, output_size]
        aleatoric_variance_scaled = tf.reduce_sum(pi * variances, axis=1)  # [batch_size, output_size]

        # Epistemic variance
        mu_weighted = pi * mu  # Weight means by their mixture probabilities
        mu_squared_weighted = pi * tf.square(mu)  # Weight squared means by their mixture probabilities
        expected_mu_squared = tf.reduce_sum(mu_squared_weighted, axis=1)  # [batch_size, output_size]
        square_of_expected_mu = tf.square(tf.reduce_sum(mu_weighted, axis=1))  # [batch_size, output_size]
        epistemic_variance_scaled = expected_mu_squared - square_of_expected_mu  # [batch_size, output_size]

        # Go from scaled space to real units
        mean_predictions_scaled = mean_predictions_scaled.numpy()
        mean_predictions, aleatoric_variance = inverse_scale_y(mean_predictions_scaled, aleatoric_variance_scaled, scaler_y)
        mean_predictions, epistemic_variance = inverse_scale_y(mean_predictions_scaled, epistemic_variance_scaled, scaler_y)

        total_variance = aleatoric_variance + epistemic_variance

        return mean_predictions, total_variance, aleatoric_variance, epistemic_variance
