"""
Plots that show the accuracy of estimates.
"""
from functools import partial
from itertools import product
from pathlib import Path
from sys import stdout
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import ticker

from .. import constants as c
from .. import metrics as m
from .common import IOP_LIMS, IOP_LIMS_PRISMA, IOP_SCALE, _plot_grouped_values, add_legend_below_figure, label_topleft, saveto_append_tag, title_type_for_scenarios
from .common import _dataframe_to_string, _select_metric, print_metric_range


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
    fig.supxlabel("In-situ (actual)", fontsize="x-large")
    fig.supylabel("Model estimate", x=-0.02, fontsize="x-large")
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


## PRISMA scatter plot - accuracy for one IOP, for each network and scenario (combining 2a-2b, 3a-3b)
_markers = ["o", "x"]
@plt.rc_context({"axes.labelweight": "bold", "axes.labelsize": "large", "axes.titleweight": "bold"})
def plot_prisma_scatter(results: pd.DataFrame, variable: c.Parameter, *,
                        uncertainty: c.Parameter=c.total_unc_pct,
                        metrics: dict[c.Parameter, Callable]={c.mdsa: m.mdsa, c.sspb: m.sspb},
                        fig: Optional[plt.Figure]=None,
                        saveto: Optional[Path | str]=None,
                        **kwargs) -> None:
    """
    For one IOP, make a scatter plot for every PRISMA subscenario.
    If `fig` is specified, draw into an existing figure (useful for subfigures).
    """
    # Setup
    norm = plt.Normalize(vmin=uncertainty.vmin, vmax=uncertainty.vmax)
    NEWFIG = (fig is None)

    # Create figure
    nrows = len(c.networks)
    ncols = len(c.scenarios_prisma_scatter)
    if NEWFIG:
        fig = plt.figure(figsize=(9.2, 16), layout="constrained")
    axs = fig.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, gridspec_kw={"hspace": 0.05}, squeeze=False)

    # Loop and plot
    for ax_row, network in zip(axs, c.networks):
        # Scatter plot
        for ax, scenarios in zip(ax_row, c.scenarios_prisma_scatter):
            # Setup
            metrics_scenario = []

            for sub, marker in zip(scenarios, _markers):
                # Select/process data
                df = results.loc[:, sub, network][variable]
                y, y_hat, color = df.loc[c.y_true], df.loc[c.y_pred], df.loc[uncertainty]
                metrics_sub = "\n".join(f"{metric.name}: {func(y, y_hat):.0f} {metric.unit}" for metric, func in metrics.items())
                metrics_scenario.append(metrics_sub)

                # Plot
                ax.scatter(y, y_hat, c=color, norm=norm, cmap=uncertainty.cmap, marker=marker, alpha=0.9, zorder=3)

            # Metrics in textbox
            if len(scenarios) == 2:  # Add labels in L2/ACOLITE case
                metrics_scenario = [f"{sub.label} ({marker})\n{metrics_sub}" for sub, marker, metrics_sub in zip(c._scenarios_prisma_sub, _markers, metrics_scenario)]
            metrics_scenario = "\n\n".join(metrics_scenario)
            ax.text(0.95, 0.05, metrics_scenario, transform=ax.transAxes, horizontalalignment="right", verticalalignment="bottom", color="black", size=9, bbox={"facecolor": "white", "edgecolor": "black", "boxstyle": "round"}, zorder=2)

            # Panel settings
            ax.set_aspect("equal")
            ax.axline((0, 0), slope=1, color="black")

        # Labels
        ax_row[len(ax_row)//2].set_title(network.label)

    # Panel settings
    axs[0, 0].set_xscale(IOP_SCALE)
    axs[0, 0].set_yscale(IOP_SCALE)
    axs[0, 0].set_xlim(*IOP_LIMS_PRISMA)
    axs[0, 0].set_ylim(*IOP_LIMS_PRISMA)

    # Labels
    axs[-1, axs.shape[1]//2].set_xlabel(f"In situ {variable.label}")
    fig.supylabel(f"Estimated {variable.label}")
    for ax, scenario in zip(axs[0], c.scenarios_prisma_overview):
        ax.set_title(f"{scenario.label}\n{ax.get_title()}")

    # Colour bar
    sm = plt.cm.ScalarMappable(cmap=uncertainty.cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axs, location="bottom", fraction=0.01, pad=0.02, extend=uncertainty.extend_cbar)
    cb.set_label(uncertainty.label, fontweight="bold")
    cb.locator = ticker.MaxNLocator(nbins=5)
    cb.update_ticks()

    if NEWFIG:
        if saveto:
            plt.savefig(saveto, bbox_inches="tight")
        else:
            plt.show()
        plt.close()


def plot_prisma_scatter_multi(results: pd.DataFrame, *,
                              variables: Iterable[c.Parameter]=[c.aph_443, c.aph_675],
                              saveto: Path | str=c.output_path/"prisma_scatter.pdf",
                              **kwargs) -> None:
    """
    For one IOP, make a scatter plot for every PRISMA subscenario.
    Option 1: Combine all into one figure (note: can get very wide).
    Option 2: One file for each.
    """
    # Plot in one figure
    nvars = len(variables)
    fig = plt.figure(figsize=(9.2*nvars, 16), layout="constrained")
    subfigs = fig.subfigures(ncols=nvars, wspace=0.04)

    for sfig, variable in zip(subfigs, variables):
        plot_prisma_scatter(results, variable, fig=sfig, **kwargs)

    plt.savefig(saveto)
    plt.close()

    # Plot separately
    for variable in variables:
        saveto_here = saveto.with_stem(f"{saveto.stem}_{variable}")
        plot_prisma_scatter(results, variable, saveto=saveto_here, **kwargs)


### ACCURACY METRICS
## Combined plot
_accuracy_metrics = [c.mdsa, c.sspb, c.log_r_squared]
def plot_accuracy_metrics(data: pd.DataFrame, *,
                          scenarios: Iterable[c.Parameter]=c.scenarios_123,
                          saveto: Path | str=c.output_path/"accuracy_metrics.pdf", tag: Optional[str]=None) -> None:
    """
    Generate a boxplot of accuracy metrics contained in a DataFrame.
    """
    # Setup
    title_type = title_type_for_scenarios(scenarios)

    # Generate figure ; rows are metrics, columns are scenarios
    n_rows = len(_accuracy_metrics)
    n_cols = len(scenarios)
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True, sharey="row", figsize=(3.2*n_cols, 1.8*n_rows), gridspec_kw={"hspace": 0.007, "wspace": 0.007}, layout="constrained", squeeze=False)

    # Plot
    _plot_grouped_values(axs, data, colparameters=scenarios, groups=c.iops, groupmembers=c.networks, rowparameters=_accuracy_metrics, apply_titles=title_type)

    # Plot legend outside the subplots
    add_legend_below_figure(fig, c.networks)

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
    fig, axs = plt.subplots(nrows=n_rows, ncols=1, figsize=(1.6*n_groups, 1.8*n_rows), sharex=True, sharey=True, squeeze=False)

    # Reorder for boxplot function
    data = data.stack().unstack(level="variable")  # Transpose
    data = data.reorder_levels([-1, "network", "model", "scenario"])

    # Plot
    _plot_grouped_values(axs, data, colparameters=[metric], groups=scenarios, groupmembers=c.networks, rowparameters=variables, apply_titles=False, ylim_quantile=0.015, **kwargs)

    # Adjust axis labels
    fig.supylabel(metric.label)
    for ax, var in zip(axs.ravel(), variables):
        ax.set_ylabel(None)
        label_topleft(ax, var.label)

    # Plot legend outside the subplots
    add_legend_below_figure(fig, c.networks)

    plt.tight_layout()
    saveto = saveto_append_tag(saveto, tag)
    plt.savefig(saveto, bbox_inches="tight")
    plt.close()


### PRINT METRICS
## Compare metrics across scenarios, architectures, etc.
print_mdsa_range = partial(print_metric_range, metric=c.mdsa)

## Print difference between normal and recalibrated models
def print_metric(data: pd.DataFrame, metric: c.Parameter, *,
                 variables: Iterable[c.Parameter]=c.iops,
                 saveto: Path | str=stdout) -> None:
    """
    Print the values for one metric.
    Optionally save to a file.
    """
    data = _select_metric(data, metric, columns=variables)
    print()
    print(f"{metric.label}:")
    print(_dataframe_to_string(data), file=saveto)
    print()


def print_metric_difference(metrics_all: pd.DataFrame, metrics_all_recal: pd.DataFrame, metrics_median: pd.DataFrame, metrics_median_recal: pd.DataFrame, metric: c.Parameter, *,
                            variables: Iterable[c.Parameter]=c.iops,
                            saveto: Path | str=stdout) -> None:
    """
    For one metric, print the difference between the without-recalibration and with-recalibration cases, in absolute terms and relative to the standard deviation between model instances.
    """
    # Select this metric
    metrics_all, metrics_all_recal, metrics_median, metrics_median_recal = [_select_metric(df, metric, columns=variables) for df in (metrics_all, metrics_all_recal, metrics_median, metrics_median_recal)]

    # Absolute difference
    metrics_diff = metrics_median_recal - metrics_median
    print()
    print(f"Difference in {metric.label}:")
    print(_dataframe_to_string(metrics_diff), file=saveto)
    print()

    # Relative difference
    metrics_std, metrics_std_recal = [df.groupby(c.scenario_network).std() for df in (metrics_all, metrics_all_recal)]
    metrics_diff_relative = metrics_diff / metrics_std * 100
    print()
    print(f"Difference in {metric.label} -- as % of the standard deviation:")
    print(_dataframe_to_string(metrics_diff_relative), file=saveto)
    print()

print_mdsa = partial(print_metric, metric=c.mdsa)
print_mdsa_difference = partial(print_metric_difference, metric=c.mdsa)
