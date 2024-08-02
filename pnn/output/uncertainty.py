"""
Plots that show the uncertainty of estimates.
"""
import itertools
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.special import erf

from matplotlib import pyplot as plt
from matplotlib import ticker, transforms
from matplotlib.colors import Normalize

from .. import constants as c
from .common import IOP_SCALE, _plot_grouped_values, add_legend_below_figure, saveto_append_tag


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
    # Generate figure
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
def plot_uncertainty_heatmap(results_agg: pd.DataFrame, *,
                             uncertainties: Iterable[c.Parameter]=[c.total_unc_pct, c.ale_frac], variables: Iterable[c.Parameter]=c.iops, scenarios: Iterable[c.Parameter]=c.scenarios_123,
                             saveto: Path | str=c.output_path/"uncertainty_heatmap.pdf", tag: Optional[str]=None) -> None:
    """
    Plot a heatmap showing the average uncertainty and aleatoric fraction for each combination of network, IOP, and splitting method.
    """
    # Generate figure
    nrows = len(uncertainties)
    ncols = len(scenarios)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(3.7*ncols, 3*nrows), gridspec_kw={"wspace": -1, "hspace": 0}, layout="constrained", squeeze=False)

    # Plot data
    for ax_row, unc in zip(axs, uncertainties):
        # Plot each panel per row
        for ax, scenario in zip(ax_row, scenarios):
            # Select relevant data
            df = results_agg.loc[unc, scenario][variables]

            # Plot image
            im = ax.imshow(df, cmap=unc.cmap, vmin=unc.vmin, vmax=unc.vmax)

            # Plot individual values
            norm = Normalize(unc.vmin, unc.vmax)
            for i, x in enumerate(c.networks):
                for j, y in enumerate(variables):
                    # Ensure text is visible
                    value = df.iloc[i, j]
                    facecolor = unc.cmap(norm(value))
                    facecolor_brightness = 0.3 * facecolor[0] + 0.6 * facecolor[1] + 0.1 * facecolor[2]
                    textcolor = "w" if facecolor_brightness < 0.7 else "k"

                    # Show text
                    ax.text(j, i, f"{value:.0f}", ha="center", va="center", color=textcolor)

            # Panel settings
            ax.grid(False)

        # Color bar per row
        cb = fig.colorbar(im, ax=ax_row, fraction=0.1, pad=0.01, shrink=1, extend="max" if unc is c.total_unc_pct else "neither")
        cb.set_label(label=unc.label, weight="bold")
        cb.locator = ticker.MaxNLocator(nbins=6)
        cb.update_ticks()

    # Labels
    fig.supxlabel("IOPs", fontweight="bold")
    fig.supylabel("Models", fontweight="bold")

    wrap_labels = lambda parameters: list(zip(*enumerate(p.label for p in parameters)))  # (0, 1, ...) (label0, label1, ...)
    wrap_labels2 = lambda parameters: list(zip(*enumerate(p.label_2lines for p in parameters)))  # (0, 1, ...) (label0, label1, ...)
    for ax in axs[-1]:
        ax.set_xticks(*wrap_labels2(variables))

    for ax in axs[:, 0]:
        ax.set_yticks(*wrap_labels(c.networks))

    for ax, scenario in zip(axs[0], scenarios):
        ax.set_title(scenario.label_2lines)

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
    # Generate figure ; rows are metrics, columns are split types
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
    # Generate figure ; rows are metrics, columns are split types
    n_rows = 2
    n_cols = len(scenarios)
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3.5*n_cols, 7), sharex=True, sharey=True, squeeze=False, gridspec_kw={"hspace": 0.4, "wspace": 0.05})

    # Plot
    for j, df in enumerate([data, data_recal]):
        _plot_grouped_values(axs[[j]], df, colparameters=scenarios, rowparameters=[c.coverage], groups=c.iops, groupmembers=c.networks)

    # k lines
    for ax_row in axs:
        add_coverage_k_lines(*ax_row)

    # Recal labels
    fig.subplots_adjust(top=0.95, bottom=0.08)
    labels = ["Without recalibration", "With recalibration"]
    label_positions = [0.985, 0.48]  # Hand-tuned
    for label, position in zip(labels, label_positions):
        fig.text(0.5, position, label, fontsize=12, fontweight="bold", ha="center")

    # Plot legend outside the subplots
    add_legend_below_figure(fig, c.networks)

    saveto = saveto_append_tag(saveto, tag)
    plt.savefig(saveto, bbox_inches="tight")
    plt.close()


### MISCALIBRATION AREA
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


### CALIBRATION CURVES
## Plot one
def _plot_calibration_single(ax: plt.Axes, data: pd.Series, **kwargs) -> None:
    """
    Plot a single calibration curve.
    Accounts for the desired order (expected on the y axis, observed on the x axis).
    """
    ax.plot(data.array, data.index, **kwargs)


## Plot all
def plot_calibration_curves(calibration_curves: pd.DataFrame, *,
                            rows: Iterable[c.Parameter]=c.scenarios_123, columns: Iterable[c.Parameter]=c.iops, groupmembers: Iterable[c.Parameter]=c.networks,
                            saveto: Path | str=c.output_path/"calibration_curves.pdf", tag: Optional[str]=None) -> None:
    """
    Plot calibration curves (expected vs. observed).
    """
    # Create figure
    fig, axs = plt.subplots(nrows=len(rows), ncols=len(columns), sharex=True, sharey=True, figsize=(2*len(columns), 2*len(rows)), layout="constrained", squeeze=False)

    # Loop and plot
    for ax_row, row_key in zip(axs, rows):
        for ax, col_key in zip(ax_row, columns):
            # Select data
            df = calibration_curves.loc[row_key][col_key]

            # Plot data
            for key in groupmembers:
                _plot_calibration_single(ax, df.loc[key], c=key.color, lw=3)

            # Plot diagonal
            ax.axline((0, 0), slope=1, c="black")
            ax.grid(True, ls="--", c="black", alpha=0.5)

    # Limits
    axs[0, 0].set_xlim(0, 1)
    axs[0, 0].set_ylim(0, 1)
    for ax in axs.ravel():
        ax.set_aspect("equal")
    axs[0, 0].locator_params(axis="both", nbins=5)  # Creates one spurious xtick that I have no idea how to deal with, but probably no one will notice

    # Plot legend outside the subplots
    add_legend_below_figure(fig, groupmembers)

    # Labels
    for ax, title in zip(axs[0], columns):
        ax.set_title(title.label)

    for ax, label in zip(axs[:, 0], rows):
        ax.set_ylabel(label.label_2lines)

    for ax in axs[-1]:
        ax.set_xlabel(None)

    fig.supxlabel("Observed proportion in interval", fontweight="bold")
    fig.supylabel("Expected proportion in interval", fontweight="bold")
    fig.align_ylabels()

    saveto = saveto_append_tag(saveto, tag)
    plt.savefig(saveto, bbox_inches="tight")
    plt.close()
