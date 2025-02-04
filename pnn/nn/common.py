"""
Functions etc. to be shared between network architectures, e.g. loss functions.
"""
from os import makedirs
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .pnn_base import BasePNN
from .. import constants as c, metrics as m
from ..modeloutput import save_model_outputs


### TRAINING
def train_N_models(model_class: type, X_train: np.ndarray, y_train: np.ndarray, *,
                   n_models: int=25, **kwargs) -> list[BasePNN]:
    """
    Train N instances of the provided model class.
    **kwargs are passed to the build_and_train method of the desired class.
    """
    all_models = []

    for i in range(n_models):
        label = f"{i+1}/{n_models}"

        # Train model
        model = model_class.build_and_train(X_train, y_train, **kwargs)
        all_models.append(model)
        print(f"\n\n   Model {label}: Finished training.   \n\n")

    return all_models


### APPLICATION
def estimate_N_models(models: Iterable[BasePNN], X: np.ndarray, **kwargs) -> list[tuple[np.ndarray]]:
    """
    Run predictions on testing data (X) for N models.
    The different estimates (mean, variance, etc.) are not separated.
    """
    estimates = [model.predict_with_uncertainty(X, **kwargs) for model in models]
    return estimates


def calculate_N_metrics(y_true: np.ndarray, estimates: Iterable[tuple[np.ndarray]]) -> pd.DataFrame:
    """
    Calculate metrics for estimates from N models.
    Metrics are collated into a single DataFrame.
    """
    all_metrics = [calculate_metrics(y_true, mean_preds, total_var) for mean_preds, total_var, *_ in estimates]
    all_metrics = pd.concat({i: df for i, df in enumerate(all_metrics)}, names=["model", "variable"])
    return all_metrics


### LOADING / SAVING
def _saveto_iteration(saveto: Path | str, i: int) -> Path:
    """
    For a given path, create a variant in subfolder `i`.
    The subfolder is recursively created if it does not exist yet.

    Example:
        _saveto_iteration("C:/path/to/myfile.keras", 5) -> Path("C:/path/to/5/myfile.keras")
    """
    saveto = Path(saveto)
    folder = saveto.parent / str(i)
    makedirs(folder, exist_ok=True)
    saveto_i = folder / saveto.name
    return saveto_i


def save_models(models: Iterable[BasePNN], saveto: Path | str, **kwargs) -> None:
    """
    Save multiple trained PNNs to file.
    Separate folders are used (/created if necessary) for each.
    """
    for i, model in enumerate(models):
        saveto_i = _saveto_iteration(saveto, i)
        model.save(saveto_i, **kwargs)


def load_model_iteration(pnn_type: type, i: int, scenario: c.Parameter | str,
                         *, saveto: Optional[Path | str]=c.model_path) -> BasePNN:
    """
    From a given save folder, load the i'th model for a given scenario.
    Convenience function for dealing with multi-model save folders.
    Example:
        load_model_iteration(BNN_DC, 16, c.prisma_gen) will load the model at c.model_path/16/bnn_dc_prisma_gen.keras
    """
    saveto = Path(saveto)
    saveto_i = saveto / str(i)
    filename = saveto_i / f"{pnn_type.name}_{scenario}.keras"
    return pnn_type.load(filename)


def save_estimates(y_true: np.ndarray, estimates: Iterable[tuple[np.ndarray]], saveto: Path | str, **kwargs) -> None:
    """
    Save multiple PNN estimates to file.
    Separate folders are used (/created if necessary) for each.
    """
    for i, est in enumerate(estimates):
        saveto_i = _saveto_iteration(saveto, i)
        save_model_outputs(y_true, *est, saveto_i, **kwargs)


### ASSESSMENT
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, total_var: np.ndarray, *, columns: Iterable[str]=c.iops_names) -> pd.DataFrame:
    """
    Calculate the mean absolute percentage error (MAPE) and other metrics between true and predicted values.

    Args:
    - y_true: Actual values (numpy array).
    - y_pred: Predicted values (numpy array).
    - total_var: Total predicted variance (numpy array).

    Returns:
    - DataFrame of metrics (MAPD, MAD, sspb, mdsa) for the predictions.
    """
    # Ensure y_true and y_pred are DataFrames
    y_true = pd.DataFrame(y_true, columns=columns)
    y_pred = pd.DataFrame(y_pred, columns=columns)
    total_var = pd.DataFrame(total_var, columns=columns)
    y_pred_std = np.sqrt(total_var)

    # Calculate metrics
    metrics_combined = {"MdSA": m.mdsa(y_true, y_pred),
                        "SSPB": m.sspb(y_true, y_pred),
                        "MAD": m.MAD(y_true, y_pred),
                        "MAPE": m.mape(y_true, y_pred),
                        "r_squared": m.r_squared(y_true, y_pred),
                        "log_r_squared": m.log_r_squared(y_true, y_pred),
                        "coverage": m.coverage(y_true, y_pred, y_pred_std),
                        "MA": m.miscalibration_area(y_true, y_pred, y_pred_std),
                        }

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
    aleatoric_fraction = aleatoric_variance / (aleatoric_variance + epistemic_variance) * 100
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
