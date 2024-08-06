"""
Plots that show the uncertainty of estimates.
"""
from functools import partial
import itertools
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.special import erf

from matplotlib import pyplot as plt
from matplotlib import transforms

from .. import constants as c
from .common import IOP_SCALE, _heatmap, _plot_grouped_values, add_legend_below_figure, saveto_append_tag


### IOP VALUE VS. UNCERTAINTY (BINNED)
## Individual panels
def plot_log_binned_statistics_line(binned: pd.DataFrame, ax: plt.Axes, *,
                                    uncertainties: Iterable[str]=c.relative_uncertainties, **kwargs) -> None:
    """
    Given a DataFrame containing log-binned statistics, plot the total/aleatoric/epistemic uncertainties for one variable.
    Plots a line for the average uncertainty and a shaded area for typical deviations.
    """
    # Loop over uncertainty types and plot each
    for unc in uncertainties:
        df = binned.loc[unc]
        color = unc.color

        df.plot.line(ax=ax, y="median", color=color, label=unc.label, **kwargs)
        ax.fill_between(df.index, df["median"] - df["mad"], df["median"] + df["mad"], color=color, alpha=0.1)

    # Labels
    ax.grid(True, ls="--")


## Combined, for all combinations of network and scenario
def plot_log_binned_statistics(binned: pd.DataFrame, *,
                               scenarios: Iterable[c.Parameter]=c.scenarios_123,
                               saveto: Path | str=c.supplementary_path/"uncertainty_line.pdf", tag: Optional[str]=None) -> None:
    """
    Plot log-binned statistics from a main DataFrame.
    """
    # Create figure
    nrows = len(scenarios)*len(c.networks)
    ncols = len(c.iops)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(2.5*ncols, 1.67*nrows), layout="constrained", squeeze=False)

    # Plot lines
    for ax_row, (network, scenario) in zip(axs, itertools.product(c.networks, scenarios)):
        for ax, var in zip(ax_row, c.iops):
            df = binned.loc[scenario, network][var]
            plot_log_binned_statistics_line(df, ax=ax, legend=False)

    # Settings
    axs[0, 0].set_xscale(IOP_SCALE)
    for ax in axs.ravel():
        ax.set_ylim(ymin=0)
    for ax, var in zip(axs[-1], c.iops):
        ax.set_xlabel(var.label)
    for ax, (network, scenario) in zip(axs[:, 0], itertools.product(c.networks, scenarios)):
        ax.set_ylabel(f"{network.label}\n{scenario.label}")

    # Labels
    fig.supxlabel("In situ value", fontweight="bold")
    fig.supylabel("Median uncertainty [%]", fontweight="bold")
    fig.align_ylabels()

    saveto = saveto_append_tag(saveto, tag)
    plt.savefig(saveto)
    plt.close()


### AVERAGE UNCERTAINTY HEATMAP
_heatmap_uncertainties = [c.total_unc_pct, c.ale_frac]
# Single
def plot_uncertainty_heatmap(data: pd.DataFrame, *,
                             uncertainties: Iterable[c.Parameter]=_heatmap_uncertainties, variables: Iterable[c.Parameter]=c.iops, scenarios: Iterable[c.Parameter]=c.scenarios_123,
                             saveto: Path | str=c.output_path/"uncertainty_heatmap.pdf", tag: Optional[str]=None) -> None:
    """
    Plot a heatmap showing the average uncertainty and aleatoric fraction for each combination of network, IOP, and splitting method.
    """
    # Create figure
    nrows = len(uncertainties)
    ncols = len(scenarios)
    width, height = 3.7*ncols, 3*nrows
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(width, height), gridspec_kw={"wspace": -1, "hspace": 0}, layout="constrained", squeeze=False)

    # Plot data
    _heatmap(axs, data, rowparameters=uncertainties, colparameters=scenarios, datarowparameters=c.networks, datacolparameters=variables)

    # Labels
    fig.supxlabel("IOPs", fontweight="bold")
    fig.supylabel("Models\n", fontweight="bold")

    saveto = saveto_append_tag(saveto, tag)
    plt.savefig(saveto)
    plt.close()


# With recalibration
def plot_uncertainty_heatmap_with_recal(data: pd.DataFrame, data_recal: pd.DataFrame, *,
                                        uncertainties: Iterable[c.Parameter]=_heatmap_uncertainties, variables: Iterable[c.Parameter]=c.iops, scenarios: Iterable[c.Parameter]=c.scenarios_123,
                                        saveto: Path | str=c.output_path/"uncertainty_heatmap_recal.pdf", tag: Optional[str]=None) -> None:
    """
    Plot a heatmap showing the average uncertainty and aleatoric fraction for each combination of network, IOP, and splitting method.
    Includes an extra row for recalibrated uncertainties.
    """
    # Find recalibrated uncertainties
    uncertainties_recal = [param for param in uncertainties if param is not c.ale_frac]

    # Create figure
    nrows_data = len(uncertainties)
    nrows_recal = len(uncertainties_recal)
    nrows = nrows_data + nrows_recal
    ncols = len(scenarios)
    width, height = 3.33*ncols, 2.2*nrows
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(width, height), gridspec_kw={"wspace": -1, "hspace": 0}, layout="constrained", squeeze=False)

    # Plot data
    _heatmap(axs[:nrows_data], data, rowparameters=uncertainties, colparameters=scenarios, datarowparameters=c.networks, datacolparameters=variables)
    _heatmap(axs[nrows_data:], data_recal, rowparameters=uncertainties_recal, colparameters=scenarios, datarowparameters=c.networks, datacolparameters=variables, apply_titles=False, colorbar_tag="\n(Recalibrated)")

    # Labels
    fig.supxlabel("IOPs", fontweight="bold")
    fig.supylabel("Models", fontweight="bold")

    saveto = saveto_append_tag(saveto, tag)
    plt.savefig(saveto)
    plt.close()



### COVERAGE
## Helper function: Lines at k=1, 2, ...
k_to_percentage = lambda k: 100*erf(k/np.sqrt(2))

def add_coverage_k_lines(*axs: Iterable[plt.Axes], klim: int=3) -> None:
    """
    Add horizontal lines at k=1, k=2, ... coverage.
    """
    for k in range(1, klim+1):
        percentage = k_to_percentage(k)

        # Plot lines
        for ax in axs:
            ax.axhline(percentage, color="black", linestyle="--", linewidth=1, zorder=4)

        # Add text to last panel
        ax = axs[-1]
        trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)  # X in plot coordinates, Y in data coordinates
        ax.text(1.02, percentage, f"$k = {k}$", transform=trans, horizontalalignment="left", verticalalignment="center")

## Plot coverage per IOP, network, scenario
def plot_coverage(data: pd.DataFrame, *,
                  scenarios: Iterable[c.Parameter]=c.scenarios_123,
                  saveto: Path | str=c.output_path/"uncertainty_coverage.pdf", tag: Optional[str]=None) -> None:
    """
    Box plot showing the coverage factor (pre-calculated).
    """
    # Generate figure ; columns are split types
    n_rows = 1
    n_cols = len(scenarios)
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3.5*n_cols, 4), sharex=True, sharey=True, squeeze=False)

    # Plot
    _plot_grouped_values(axs, data, colparameters=scenarios, rowparameters=[c.coverage], groups=c.iops, groupmembers=c.networks)

    # k lines
    for ax_row in axs:
        add_coverage_k_lines(*ax_row)

    # Plot legend outside the subplots
    add_legend_below_figure(fig, c.networks)

    plt.tight_layout()
    saveto = saveto_append_tag(saveto, tag)
    plt.savefig(saveto, bbox_inches="tight")
    plt.close()

## Plot coverage per IOP, network, scenario -- with recalibrated data
def plot_coverage_with_recal(data: pd.DataFrame, data_recal: pd.DataFrame, *,
                             scenarios: Iterable[c.Parameter]=c.scenarios_123,
                             saveto: Path | str=c.output_path/"uncertainty_coverage_recal.pdf", tag: Optional[str]=None) -> None:
    """
    Box plot showing the coverage factor (pre-calculated).
    """
    # Generate figure ; columns are split types
    nrows = 2
    ncols = len(scenarios)
    fig = plt.figure(figsize=(3.5*ncols, 7), layout="constrained")
    subfigs = fig.subfigures(nrows=nrows, hspace=0.1)
    labels = ["Without recalibration", "With recalibration"]

    # Plot
    for sfig, df, title in zip(subfigs, [data, data_recal], labels):
        axs = sfig.subplots(nrows=1, ncols=ncols, sharex=True, sharey=True, squeeze=False)
        _plot_grouped_values(axs, df, colparameters=scenarios, rowparameters=[c.coverage], groups=c.iops, groupmembers=c.networks)

        # k lines
        add_coverage_k_lines(*axs[0])

        # Recal labels
        sfig.suptitle(title, fontweight="bold", fontsize=12)

    # Plot legend outside the subplots
    add_legend_below_figure(fig, c.networks)

    saveto = saveto_append_tag(saveto, tag)
    plt.savefig(saveto, bbox_inches="tight")
    plt.close()


### MISCALIBRATION AREA
## Table
def table_miscalibration_area(df: pd.DataFrame, *,
                              scenarios: Iterable[c.Parameter]=c.scenarios_123,
                              saveto: Path | str=c.output_path/"miscalibration_area.csv", tag: Optional[str]=None) -> None:
    """
    Reorder the miscalibration area table and save it to file.
    To do: fully automate.
    """
    areas = df[c.miscalibration_area].unstack("variable")[c.iops]
    areas = areas.reorder_levels(["network", "scenario"])
    areas.sort_index(inplace=True)
    areas.sort_index(key=lambda x: x.map({model: i for i, model in enumerate(c.networks)}))
    areas.rename(index={model.name: model.label for model in c.networks}, level="network", inplace=True)
    areas.rename(index={scenario.name: scenario.label for scenario in scenarios}, level="scenario", inplace=True)

    saveto = saveto_append_tag(saveto, tag)
    areas.to_csv(saveto)


## Plot
def miscalibration_area_heatmap(data: pd.DataFrame, data_recal: pd.DataFrame, *,
                                metric: c.Parameter=c.miscalibration_area, scenarios: Iterable[c.Parameter]=c.scenarios_123, variables: Iterable[c.Parameter]=c.iops,
                                precision: int=2, diff_range: float=0.4,
                                saveto: Path | str=c.output_path/"miscalibration_area.pdf", tag: Optional[str]=None) -> None:
    """
    Generate a heatmap showing the miscalibration area without and with recalibration, as well as their difference.
    Can be used for a different variable by changing `metric`
    """
    labels = ["Without recalibration", "With recalibration", "Calibration difference"]

    # Reorder data
    new_order = [-1, "network", "scenario"]
    data, data_recal = [df.stack().unstack("variable").reorder_levels(new_order) for df in (data, data_recal)]

    # Calculate miscalibration area difference
    metric_diff = c.Parameter(f"{metric.name}_diff", f"{metric.label} difference", cmap=c.cmap_difference, symmetric=True, vmin=-diff_range, vmax=diff_range)
    data_diff = data_recal.loc[metric] - data.loc[metric]
    data_diff = pd.concat({metric_diff.name: data_diff})

    # Create figure
    nrows = len(c.networks)
    ncols = 3
    nrows_data = len(scenarios)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(13, 14), gridspec_kw={"hspace": 0.25, "wspace": 0.02})

    # Plot main metric  --  note that axs is being transposed, so rows <--> columns
    plot_heatmap = partial(_heatmap, colparameters=c.networks, datarowparameters=scenarios, datacolparameters=variables, colorbar_location="bottom", precision=precision)
    plot_heatmap(axs[np.newaxis, :, 0], data, rowparameters=[metric], apply_titles=False)
    plot_heatmap(axs[np.newaxis, :, 1], data_recal, rowparameters=[metric], apply_titles=True)
    plot_heatmap(axs[np.newaxis, :, 2], data_diff, rowparameters=[metric_diff], apply_titles=False)

    # Labels
    for ax, label in zip(axs[0], labels):
        ax.set_title(f"{label}\n{ax.get_title()}")
    for ax in axs.ravel():  # To do: replace with rcParams
        title = ax.get_title()
        if title:
            ax.set_title(title, fontweight="bold")

    saveto = saveto_append_tag(saveto, tag)
    plt.savefig(saveto, bbox_inches="tight")
    plt.close()



### CALIBRATION CURVES
## Plot one
def _plot_calibration_single(ax: plt.Axes, data: pd.Series, **kwargs) -> None:
    """
    Plot a single calibration curve.
    Accounts for the desired order (expected on the y axis, observed on the x axis).
    """
    ax.plot(data.array, data.index, **kwargs)


## Plot all
# Base
def _plot_calibration_curves_base(axs: np.ndarray[plt.Axes], calibration_curves: pd.DataFrame,
                                  rows: Iterable[c.Parameter], columns: Iterable[c.Parameter], groupmembers: Iterable[c.Parameter]) -> None:
    """
    Plot calibration curves (expected vs. observed).

    Base function that plots into existing panels. Should not be used by itself.
    """
    # Plot all curves
    for ax_row, row_key in zip(axs, rows):
        for ax, col_key in zip(ax_row, columns):
            # Select data
            df = calibration_curves.loc[row_key][col_key]

            # Plot data
            for key in groupmembers:
                _plot_calibration_single(ax, df.loc[key], c=key.color, lw=3)

            # Plot diagonal
            ax.axline((0, 0), slope=1, c="black")

    # Limits (in a loop because there is no guarantee that the axs share x or y)
    for ax in axs.ravel():
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.locator_params(axis="both", nbins=5)  # Creates one spurious xtick that I have no idea how to deal with, but probably no one will notice

    # Labels
    for ax, title in zip(axs[0], columns):
        ax.set_title(title.label)

    for ax, label in zip(axs[:, 0], rows):
        ax.set_ylabel(label.label_2lines)

    for ax in axs[-1]:
        ax.set_xlabel(None)

    fig = axs[0, 0].figure  # Assume all axs are in the same figure
    fig.supxlabel("Observed proportion in interval", fontweight="bold")
    fig.supylabel("Expected proportion in interval", fontweight="bold")
    fig.align_ylabels()


# Applied
def plot_calibration_curves(calibration_curves: pd.DataFrame, *,
                            rows: Iterable[c.Parameter]=c.scenarios_123, columns: Iterable[c.Parameter]=c.iops, groupmembers: Iterable[c.Parameter]=c.networks,
                            saveto: Path | str=c.output_path/"calibration_curves.pdf", tag: Optional[str]=None) -> None:
    """
    Plot calibration curves (expected vs. observed).
    """
    # Create figure
    fig, axs = plt.subplots(nrows=len(rows), ncols=len(columns), sharex=True, sharey=True, figsize=(2*len(columns), 2*len(rows)), layout="constrained", squeeze=False)

    # Loop and plot
    _plot_calibration_curves_base(axs, calibration_curves, rows, columns, groupmembers)

    # Plot legend outside the subplots
    add_legend_below_figure(fig, groupmembers)

    saveto = saveto_append_tag(saveto, tag)
    plt.savefig(saveto, bbox_inches="tight")
    plt.close()


# With recalibration
def plot_calibration_curves_with_recal(calibration_curves: pd.DataFrame, calibration_curves_recal: pd.DataFrame, *,
                                       rows: Iterable[c.Parameter]=c.scenarios_123, columns: Iterable[c.Parameter]=c.iops, groupmembers: Iterable[c.Parameter]=c.networks,
                                       saveto: Path | str=c.output_path/"calibration_curves_recal.pdf", tag: Optional[str]=None) -> None:
    """
    Plot calibration curves (expected vs. observed).
    """
    # Create figure
    fig = plt.figure(figsize=(2*len(columns), 4*len(rows)), layout="constrained")
    subfigs = fig.subfigures(nrows=2, hspace=1.0/fig.get_figheight())
    labels = ["Without recalibration", "With recalibration"]

    nrows = len(rows)
    ncols = len(columns)

    # Plot
    for sfig, df, title in zip(subfigs, [calibration_curves, calibration_curves_recal], labels):
        axs = sfig.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, squeeze=False)
        _plot_calibration_curves_base(axs, df, rows, columns, groupmembers)

        # Recal labels
        sfig.suptitle(title, fontweight="bold", fontsize=12)

    # Plot legend outside the subplots
    add_legend_below_figure(fig, c.networks)

    saveto = saveto_append_tag(saveto, tag)
    plt.savefig(saveto, bbox_inches="tight")
    plt.close()
