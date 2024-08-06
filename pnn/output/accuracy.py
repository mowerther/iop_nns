"""
Plots that show the accuracy of estimates.
"""
from itertools import product
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import ticker

from .. import constants as c
from .. import metrics as m
from .common import IOP_LIMS, IOP_SCALE, _plot_grouped_values, add_legend_below_figure, label_topleft, saveto_append_tag


### SCATTER PLOTS
## Performance (matchups) - scatter plot, per algorithm/scenario combination
def plot_performance_scatter(df: pd.DataFrame, *,
                             columns: Iterable[c.Parameter]=c.iops, rows: Iterable[c.Parameter]=[c.total_unc_pct, c.ale_frac],
                             title: Optional[str]=None,
                             saveto: Path | str="scatterplot.pdf") -> None:
    """
    Plot one DataFrame with y, y_hat, with total uncertainty (top) or aleatoric fraction (bottom) as colour.
    """
    # Create figure
    fig, axs = plt.subplots(nrows=len(rows), ncols=len(columns), sharex=True, sharey=True, figsize=(20, 10), squeeze=False, layout="constrained")

    # Plot data per row
    for ax_row, uncertainty in zip(axs, rows):
        for ax, iop in zip(ax_row, columns):
            # Plot data per panel
            im = ax.scatter(df.loc[c.y_true, iop], df.loc[c.y_pred, iop], c=df.loc[uncertainty, iop],
                            rasterized=True, alpha=0.7, cmap=uncertainty.cmap, vmin=uncertainty.vmin, vmax=uncertainty.vmax)

        # Color bar per row
        cb = fig.colorbar(im, ax=ax_row[-1], label=uncertainty.label)
        cb.locator = ticker.MaxNLocator(nbins=6)
        cb.update_ticks()

    # Matchup plot settings
    for ax in axs.ravel():
        # ax.set_aspect("equal")
        ax.axline((0, 0), slope=1, color="black")
        ax.grid(True, color="black", alpha=0.5, linestyle="--")

    # Metrics
    for ax, iop in zip(axs[0], columns):
        # Calculate
        y, y_hat = df.loc[c.y_true, iop], df.loc[c.y_pred, iop]
        r_square = f"$R^2 = {m.log_r_squared(y, y_hat):.2f}$"
        sspb = f"SSPB = ${m.sspb(y, y_hat):+.1f}$%"
        other_metrics = [f"{func.__name__} = {func(y, y_hat):.1f}%" for func in [m.mdsa, m.mape]]

        # Format
        metrics_text = "\n".join([r_square, sspb, *other_metrics])
        ax.text(0.95, 0.03, metrics_text, transform=ax.transAxes, horizontalalignment="right", verticalalignment="bottom", color="black", size=9, bbox={"facecolor": "white", "edgecolor": "black"})

    # Plot settings
    axs[0, 0].set_xscale(IOP_SCALE)
    axs[0, 0].set_yscale(IOP_SCALE)
    axs[0, 0].set_xlim(*IOP_LIMS)
    axs[0, 0].set_ylim(*IOP_LIMS)

    # Labels
    for ax, iop in zip(axs[0], columns):
        ax.set_title(iop.label)
    fig.supxlabel("In-situ (actual)", fontsize="x-large", fontweight="bold")
    fig.supylabel("Model estimate", x=-0.02, fontsize="x-large", fontweight="bold")
    fig.suptitle(title, fontsize="x-large")

    # Save result
    plt.savefig(saveto, bbox_inches="tight")
    plt.close()


## Loop over scenarios and networks, and make a scatter plot for each combination
_scatter_levels = ["scenario", "network"]
def plot_performance_scatter_multi(results: pd.DataFrame, *,
                                   scenarios: Iterable[c.Parameter]=c.scenarios_123,
                                   saveto: Path | str=c.supplementary_path/"scatterplot.pdf", tag: Optional[str]=None,
                                   **kwargs) -> None:
    """
    Plot many DataFrames with y, y_hat, with total uncertainty (top) or aleatoric fraction (bottom) as colour.
    """
    saveto = saveto_append_tag(saveto, tag)

    # Loop over results and plot each network/scenario combination in a separate figure
    for network, scenario in product(c.networks, scenarios):
        saveto_here = saveto.with_stem(f"{saveto.stem}_{network}-{scenario}")
        df = results.loc[:, scenario, network]
        plot_performance_scatter(df, title=f"{network.label} {scenario.label}", saveto=saveto_here, **kwargs)


### ACCURACY METRICS
## Combined plot
_accuracy_metrics = [c.mdsa, c.sspb, c.log_r_squared]
def plot_accuracy_metrics(data: pd.DataFrame, *,
                          scenarios: Iterable[c.Parameter]=c.scenarios_123,
                          saveto: Path | str=c.output_path/"accuracy_metrics.pdf", tag: Optional[str]=None) -> None:
    """
    Generate a boxplot of accuracy metrics contained in a DataFrame.
    """
    # Generate figure ; rows are metrics, columns are scenarios
    n_rows = len(_accuracy_metrics)
    n_cols = len(scenarios)
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4*n_cols, 2.3*n_rows), sharex=True, sharey="row", squeeze=False)

    # Plot
    _plot_grouped_values(axs, data, colparameters=scenarios, groups=c.iops, groupmembers=c.networks, rowparameters=_accuracy_metrics)

    # Plot legend outside the subplots
    add_legend_below_figure(fig, c.networks)

    plt.tight_layout()
    saveto = saveto_append_tag(saveto, tag)
    plt.savefig(saveto, bbox_inches="tight")
    plt.close()


## Plot a single metric (default: MdSA, aph)
def plot_mdsa(data: pd.DataFrame, *,
              metric: c.Parameter=c.mdsa, scenarios: Iterable[c.Parameter]=c.scenarios_123, variables: Iterable[c.Parameter]=[c.aph_443, c.aph_675],
              saveto: Path | str=c.output_path/"mdsa.pdf", tag: Optional[str]=None,
              **kwargs) -> None:
    """
    Generate an MdSA boxplot.
    """
    # Generate figure ; rows are IOPs
    n_rows = len(variables)
    n_groups = len(scenarios)
    fig, axs = plt.subplots(nrows=n_rows, ncols=1, figsize=(1.6*n_groups, 2.3*n_rows), sharex=True, sharey=True, squeeze=False)

    # Reorder for boxplot function
    data = data.stack().unstack(level="variable")  # Transpose
    data = data.reorder_levels([-1, "network", "model", "scenario"])

    # Plot
    _plot_grouped_values(axs, data, colparameters=[metric], groups=scenarios, groupmembers=c.networks, rowparameters=variables, apply_titles=False, ylim_quantile=0.015, **kwargs)

    # Adjust axis labels
    fig.supylabel(metric.label, fontweight="bold", fontsize=12)
    for ax, var in zip(axs.ravel(), variables):
        ax.set_ylabel(None)
        label_topleft(ax, var.label)

    # Plot legend outside the subplots
    add_legend_below_figure(fig, c.networks)

    plt.tight_layout()
    saveto = saveto_append_tag(saveto, tag)
    plt.savefig(saveto, bbox_inches="tight")
    plt.close()