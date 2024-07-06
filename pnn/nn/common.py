"""
Functions etc. to be shared between network architectures, e.g. loss functions.
"""
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, History
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler

from .. import constants as c, metrics as m, modeloutput as mo

### DATA HANDLING
def extract_inputs_outputs(data: pd.DataFrame, *,
                           Rrs_stepsize: int=5, y_columns: Iterable[str]=c.iops) -> tuple[np.ndarray, np.ndarray]:
    """
    For a given DataFrame, extract the Rrs columns (X) and IOP columns (y).
    """
    rrs_columns = [f"Rrs_{nm}" for nm in range(400, 701, 5)]
    X = data[rrs_columns].values
    y = data[y_columns].values
    return X, y


def scale_y(y_train: np.ndarray, y_test: np.ndarray, *,
            scaled_range: tuple[int]=(-1, 1)) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply log and minmax scaling to the input arrays, training the scaler on y_train only (per IOP).
    """
    # Apply log transformation to the target variables
    y_train_log = np.log(y_train)
    y_test_log = np.log(y_test)

    # Apply Min-Max scaling to log-transformed target variables
    scaler_y = MinMaxScaler(feature_range=scaled_range)
    y_train_scaled = scaler_y.fit_transform(y_train_log)
    y_test_scaled = scaler_y.transform(y_test_log)

    return y_train_scaled, y_test_scaled, scaler_y


def inverse_scale_y(mean_scaled: np.ndarray, variance_scaled: np.ndarray, scaler_y: MinMaxScaler) -> tuple[np.ndarray, np.ndarray]:
    """
    Go back from log-minmax-scaled space to real units.
    """
    # Setup
    N = scaler_y.n_features_in_  # Number of predicted values
    original_shape = mean_scaled.shape

    # Convert from scaled space to log space: Means
    mean_scaled = mean_scaled.reshape(-1, N)
    mean_log = scaler_y.inverse_transform(mean_scaled)
    mean_log = mean_log.reshape(original_shape)

    # Convert from scaled space to log space: Variance
    scaling_factor = (scaler_y.data_max_ - scaler_y.data_min_) / 2
    variance_log = variance_scaled * (scaling_factor**2)  # Uncertainty propagation for linear equations

    # Convert from log space to the original space, i.e. actual IOPs in [m^-1]
    mean = np.exp(mean_log)  # Geometric mean / median
    variance = np.exp(2*mean_log + variance_log) * (np.exp(variance_log) - 1)  # Arithmetic variance

    return mean, variance



### TRAINING
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


def _train_general(model: Model, X_train: np.ndarray, y_train: np.ndarray, *,
                   epochs: int=1000, batch_size: int=32, learning_rate: float=0.001, validation_split: float=0.1) -> tuple[Model, History]:
    """
    Train a general Model on the provided X and y data, with early stopping.
    """
    # Setup
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    early_stopping = EarlyStopping(monitor="val_loss", patience=80, verbose=1, mode="min", restore_best_weights=True)

    # Training
    model.compile(optimizer=optimizer, loss=nll_loss)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[early_stopping])

    return model, history


def _build_and_train_general(build: Callable, train: Callable, X_train: np.ndarray, y_train: np.ndarray) -> Model:
    """
    Build and train a model on the provided X and y data, with early stopping.
    Convenience function combining the build and train functions.
    """
    # Setup
    input_shape = X_train.shape[1:]
    output_size = y_train.shape[-1]

    model = build(input_shape, output_size=output_size)
    model, history = train(model, X_train, y_train)

    return model


### APPLICATION
@tf.function  # 4x Speed-up
def predict_with_dropout(model: Model, inputs: np.ndarray, *, enable_dropout=True):
    return model(inputs, training=enable_dropout)  # `training=True` just turns the dropout on, it does not affect the model parameters


def _predict_with_uncertainty_general(predict_samples: Callable, model: Model, X: np.ndarray, scaler_y: MinMaxScaler, *, n_samples=100):
    """
    Use the given model to predict y values for given X values, including the rescaling back to regular units.
    """
    # Generate predictions in scaled space
    pred_samples = predict_samples(model, X, n_samples=n_samples)

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

    mean_predictions = np.mean(mean_predictions, axis=0)  # Average over n_samples

    return mean_predictions, total_variance, aleatoric_variance, epistemic_variance


def _train_and_evaluate_models_general(build_and_train: Callable, predict_with_uncertainty: Callable, X_train: np.ndarray, y_train_scaled: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, scaler_y: MinMaxScaler, *,
                                       n_models: int=10, mdsa_columns: Iterable[str]=c.iops_443, **predict_kwargs) -> tuple[Model, pd.DataFrame]:
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
        model = build_and_train(X_train, y_train_scaled)
        all_models.append(model)
        print(f"Model {label}: Finished training.")

        # Assess model
        mean_preds, total_var, aleatoric_var, epistemic_var = predict_with_uncertainty(model, X_test, scaler_y, **predict_kwargs)
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



### ASSESSMENT
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, columns: Iterable[str]=c.iops_names) -> pd.DataFrame:
    """
    Calculate the mean absolute percentage error (MAPE) and other metrics between true and predicted values.

    Args:
    - y_true: Actual values (numpy array).
    - y_pred: Predicted values (numpy array).

    Returns:
    - DataFrame of metrics (MAPD, MAD, sspb, mdsa) for the predictions.
    """
    # Ensure y_true and y_pred are DataFrames
    y_true = pd.DataFrame(y_true, columns=columns)
    y_pred = pd.DataFrame(y_pred, columns=columns)

    # Calculate metrics
    metrics_combined = {"MdSA": m.mdsa(y_true, y_pred),
                        "SSPB": m.sspb(y_true, y_pred),
                        "MAD": m.MAD(y_true, y_pred),
                        "MAPE": m.mape(y_true, y_pred),
                        "r_squared": m.r_squared(y_true, y_pred),
                        "log_r_squared": m.log_r_squared(y_true, y_pred),}

    metrics_combined = pd.DataFrame(metrics_combined)

    return metrics_combined


def scatterplot(y_true: np.ndarray, mean_predictions: np.ndarray, *, labels: Iterable[str]=c.iops, title: Optional[str]=None) -> None:
    """
    Make a quick scatter plot of the different variables.
    Not saved to file.
    """
    # Constants
    lims = (1e-4, 1e1)
    scale = "log"
    N = mean_predictions.shape[1]  # Number of variables

    # Plot data
    fig, axs = plt.subplots(nrows=2, ncols=N//2, sharex=True, sharey=True, layout="constrained")
    axs = axs.ravel()

    for i in range(N):
        axs[i].scatter(y_true[:, i], mean_predictions[:, i], color="black", s=3)
        axs[i].set_title(labels[i].label)

    # Matchup plot settings
    for ax in axs.ravel():
        # ax.set_aspect("equal")
        ax.axline((0, 0), slope=1, color="black")
        ax.grid(True, color="black", alpha=0.5, linestyle="--")

    # Plot settings
    axs[0].set_xscale(scale)
    axs[0].set_yscale(scale)
    axs[0].set_xlim(*lims)
    axs[0].set_ylim(*lims)
    fig.suptitle(title)

    # Show
    plt.show()
    plt.close()


def uncertainty_histogram(mean_predictions: np.ndarray, total_variance: np.ndarray, aleatoric_variance: np.ndarray, epistemic_variance: np.ndarray, *,
                          title: Optional[str]=None) -> None:
    """
    Make a quick histogram of the various uncertainties to check them.
    Not saved to file.
    """
    # Prepare data
    aleatoric_fraction = aleatoric_variance / total_variance * 100
    total_unc_pct, ale_unc_pct, epi_unc_pct = [np.sqrt(var) / mean_predictions * 100 for var in (total_variance, aleatoric_variance, epistemic_variance)]
    N = mean_predictions.shape[1]  # Number of variables
    unc_pct_bins = np.linspace(0, 200, 50)

    # Plot histograms
    fig, axs = plt.subplots(nrows=2, ncols=2, layout="constrained")
    axs = axs.ravel()
    for i in range(N):
        axs[0].hist(aleatoric_fraction[:, i], bins=np.linspace(0, 100, 50), alpha=0.5)
        axs[1].hist(total_unc_pct[:, i], bins=unc_pct_bins, alpha=0.5)
        axs[2].hist(ale_unc_pct[:, i], bins=unc_pct_bins, alpha=0.5)
        axs[3].hist(epi_unc_pct[:, i], bins=unc_pct_bins, alpha=0.5)

    # Labels
    axs[0].set_xlabel(c.ale_frac.label)
    axs[1].set_xlabel(c.total_unc_pct.label)
    axs[2].set_xlabel(c.ale_unc_pct.label)
    axs[3].set_xlabel(c.epi_unc_pct.label)
    fig.suptitle(title)

    # Show
    plt.show()
    plt.close()
