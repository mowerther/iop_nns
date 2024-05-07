"""
Various plotting functions
"""
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from matplotlib import pyplot as plt
plt.style.use("default")

from .constants import iops, uncertainty_colors, uncertainty_types, save_path


### CONSTANTS
model_colors = {
    'mdn': '#FF5733',
    'mcd': '#33FF57',
    'dc': '#3357FF',
    'ens': '#F933FF',
    'rnn': '#FFC733',
    }


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
                               saveto: Optional[Path | str]=save_path/"uncertainty_line.png") -> None:
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

    # Settings
    axs[0, 0].set_xscale("log")
    for ax in axs.ravel():
        ax.set_ylim(ymin=0)

    fig.suptitle("")
    fig.supxlabel("In situ value", fontweight="bold")
    fig.supylabel("Mean uncertainty [%]", fontweight="bold")

    plt.savefig(saveto)
    plt.close()
