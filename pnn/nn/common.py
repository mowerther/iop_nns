"""
Functions etc. to be shared between network architectures, e.g. loss functions.
"""
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from .pnn_base import BasePNN
from .. import constants as c, metrics as m


### APPLICATION
def train_and_evaluate_models(model_class: type, X_train: np.ndarray, y_train_scaled: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, scaler_y: MinMaxScaler, *,
                              n_models: int=10, mdsa_columns: Iterable[str]=c.iops_443, **predict_kwargs) -> tuple[BasePNN, pd.DataFrame]:
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
        model = model_class.build_and_train(X_train, y_train_scaled)
        all_models.append(model)
        print(f"Model {label}: Finished training.")

        # Assess model
        mean_preds, total_var, aleatoric_var, epistemic_var = model.predict_with_uncertainty(X_test, scaler_y, **predict_kwargs)
        print(f"Model {label}: Finished prediction.")

        metrics_df = calculate_metrics(y_test, mean_preds, total_var)
        all_metrics.append(metrics_df)
        print(f"Model {label}: Calculated performance metrics.")

        # Select best model so far
        evaluation_mdsa = metrics_df.loc[mdsa_columns, "MdSA"].mean()  # Average the MdSA of the specified variables
        if evaluation_mdsa < best_mdsa:
            best_mdsa = evaluation_mdsa
            best_overall_model = model
            print(f"Model {label} is the new best model (mean MdSA: {evaluation_mdsa:.0f}%).")

        print("\n\n")

    all_metrics = pd.concat({i+1: df for i, df in enumerate(all_metrics)}, names=["model", "variable"])

    return best_overall_model, all_metrics



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
