"""
Various plotting functions
"""
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.style.use("default")

from .constants import iops, iops_main, network_types, split_types, uncertainty_colors, uncertainty_types, save_path
from .metrics import metrics_display


### CONSTANTS
model_colors = {
    "mdn": "#FF5733",
    "bnn_dropconnect": "#3357FF",
    "bnn_mcd": "#33FF57",
    "ensemble": "#F933FF",
    "rnn": "#FFC733",}


### FUNCTIONS
def plot_log_binned_statistics_line(binned: pd.DataFrame, variable: str, ax: plt.Axes, *,
                                    uncertainty_keys: Iterable[str]=uncertainty_types.keys(), **kwargs) -> None:
    """
    Given a DataFrame containing log-binned statistics, plot the total/aleatoric/epistemic uncertainties for one variable.
    Plots a line for the mean uncertainty and a shaded area for the standard deviation.
    If no ax is provided, a new figure is created.
    """
    # Set up keys
    mean, std = f"{variable}_mean", f"{variable}_std"

    # Loop over uncertainty types and plot each
    for unc, label in uncertainty_types.items():
        df = binned.loc[unc]
        color = uncertainty_colors[unc]

        df.plot.line(ax=ax, y=mean, color=color, label=label, **kwargs)
        ax.fill_between(df.index, df[mean] - df[std], df[mean] + df[std], color=color, alpha=0.1)

    # Labels
    ax.set_xlabel(variable)
    ax.grid(True, ls="--")


def plot_log_binned_statistics(binned: Iterable[pd.DataFrame], *,
                               saveto: Path | str=save_path/"uncertainty_line.png") -> None:
    """
    Plot some number of DataFrames containing log-binned statistics.
    """
    # If only one DataFrame is provided, wrap it in a list
    if isinstance(binned, pd.DataFrame):
        binned = [binned]

    # Generate figure
    fig, axs = plt.subplots(nrows=len(binned), ncols=len(iops), sharex=True, figsize=(15, 25), layout="constrained", squeeze=False)

    # Plot lines
    for ax_row, (label, df) in zip(axs, binned.items()):
        for ax, var in zip(ax_row, iops):
            plot_log_binned_statistics_line(df, var, ax=ax, legend=False)

        ax_row[0].set_ylabel(label)

    # Settings
    axs[0, 0].set_xscale("log")
    for ax in axs.ravel():
        ax.set_ylim(ymin=0)

    fig.suptitle("")
    fig.supxlabel("In situ value", fontweight="bold")
    fig.supylabel("Mean uncertainty [%]", fontweight="bold")

    plt.savefig(saveto)
    plt.close()


def plot_performance_metrics_lollipop(metrics_results: dict[str, pd.DataFrame], *,
                                      groups: dict[str, str]=iops_main, metrics_to_plot: dict[str, str]=metrics_display, models_to_plot: dict[str, str]=network_types, splits: dict[str, str]=split_types,
                                      saveto: Path | str=save_path/"performance_lolliplot_vertical.png") -> None:
    """
    Plot some number of DataFrames containing performance metric statistics.
    """
    # Constants
    bar_width = 0.15
    opacity = 0.8

    # Separating the results for the scenarios
    metrics_results_split = {label: {key: val for key, val in metrics_results.items() if f"_{label}" in key} for label in splits}

    # Generate figure ; rows are metrics, columns are split types
    n_groups = len(groups)
    n_metrics = len(metrics_to_plot)
    n_models = len(models_to_plot)
    n_splits = len(splits)
    fig, axs = plt.subplots(nrows=n_metrics, ncols=n_splits, figsize=(14, 8), sharex=True, sharey="row", squeeze=False)

    # Plot results; must be done in a loop because there is no Pandas lollipop function
    for ax_row, metric_label in zip(axs, metrics_to_plot):
        for ax, (split_label, metrics_split) in zip(ax_row, metrics_results_split.items()):
            for model_idx, (network_type, df) in enumerate(zip(models_to_plot, metrics_split.values())):
                color = model_colors.get(network_type, "gray")
                label = models_to_plot.get(network_type, "model")

                values = df.loc[metric_label][groups.keys()]
                locations = np.arange(n_groups) - (bar_width * (n_models - 1) / 2) + model_idx * bar_width

                ax.scatter(locations, values, color=color, label=label, s=50, zorder=3)  # Draw points
                ax.vlines(locations, 0, values, colors='grey', lw=1, alpha=0.7)

            ax.grid(True, axis="y", linestyle="--", linewidth=0.5, color="black", alpha=0.4)

    # Label variables
    axs[0, 0].set_xticks(np.arange(n_groups))
    axs[0, 0].set_xticklabels(groups.values())

    # Label y-axes
    for ax, ylabel in zip(axs[:, 0], metrics_to_plot.values()):
        ax.set_ylabel(ylabel, fontsize=12)
    fig.align_ylabels()

    # y-axis limits; currently hardcoded
    axs[0, 0].set_ylim(ymin=0)
    axs[2, 0].set_ylim(0, 1)

    maxbias = np.abs(axs[1, 0].get_ylim()).max()
    axs[1, 0].set_ylim(-maxbias, maxbias)

    # Titles
    for ax, title in zip(axs[0], splits.values()):
        ax.set_title(title)

    # Plot legend outside the subplots
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(saveto, dpi=200, bbox_inches="tight")
    plt.close()
