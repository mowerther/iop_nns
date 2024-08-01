"""
Functions relating to outputs, such as plots and tables.
"""
import itertools
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.special import erf

from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.colors import Normalize

from .. import metrics
from .. import constants as c

### FUNCTIONS
## Uncertainty statistics - line plot
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

def plot_log_binned_statistics(binned: pd.DataFrame, *,
                               saveto: Path | str=c.supplementary_path/"uncertainty_line.pdf") -> None:
    """
    Plot log-binned statistics from a main DataFrame.
    """
    # Generate figure
    fig, axs = plt.subplots(nrows=len(c.scenarios_123)*len(c.networks), ncols=len(c.iops), sharex=True, figsize=(15, 25), layout="constrained", squeeze=False)

    # Plot lines
    for ax_row, (network, split) in zip(axs, itertools.product(c.networks, c.scenarios_123)):
        for ax, var in zip(ax_row, c.iops):
            df = binned.loc[split, network][var]
            plot_log_binned_statistics_line(df, ax=ax, legend=False)

    # Settings
    axs[0, 0].set_xscale("log")
    for ax in axs.ravel():
        ax.set_ylim(ymin=0)
    for ax, var in zip(axs[-1], c.iops):
        ax.set_xlabel(var.label)
    for ax, (network, split) in zip(axs[:, 0], itertools.product(c.networks, c.scenarios_123)):
        ax.set_ylabel(f"{network.label}\n{split.label}")

    # Labels
    fig.suptitle("")
    fig.supxlabel("In situ value", fontweight="bold")
    fig.supylabel("Median uncertainty [%]", fontweight="bold")
    fig.align_ylabels()

    plt.savefig(saveto)
    plt.close()


## Uncertainty statistics - heatmap
_heatmap_metrics = [c.total_unc_pct, c.ale_frac]
def plot_uncertainty_heatmap(results_agg: pd.DataFrame, *,
                             variables: Iterable[c.Parameter]=c.iops,
                             saveto: Path | str=c.output_path/"uncertainty_heatmap.pdf") -> None:
    """
    Plot a heatmap showing the average uncertainty and aleatoric fraction for each combination of network, IOP, and splitting method.
    """
    # Generate figure
    fig, axs = plt.subplots(nrows=2, ncols=len(c.scenarios_123), sharex=True, sharey=True, figsize=(11, 6), gridspec_kw={"wspace": -1, "hspace": 0}, layout="constrained", squeeze=False)

    # Plot data
    for ax_row, unc in zip(axs, _heatmap_metrics):
        # Plot each panel per row
        for ax, split in zip(ax_row, c.scenarios_123):
            # Select relevant data
            df = results_agg.loc[unc, split]
            df = df[variables]

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

        # Color bar per row
        cb = fig.colorbar(im, ax=ax_row, fraction=0.1, pad=0.01, shrink=1, extend="max" if unc is c.total_unc_pct else "neither")
        cb.set_label(label=unc.label, weight="bold")
        cb.locator = ticker.MaxNLocator(nbins=6)
        cb.update_ticks()

    # Labels
    fig.supxlabel("IOPs", fontweight="bold")
    fig.supylabel("Models", fontweight="bold")

    wrap_labels = lambda parameters: list(zip(*enumerate(p.label for p in parameters)))  # (0, 1, ...) (label0, label1, ...)
    for ax in axs[-1]:
        ax.set_xticks(*wrap_labels(variables), rotation=45, ha="right")

    for ax in axs[:, 0]:
        ax.set_yticks(*wrap_labels(c.networks))

    for ax, split in zip(axs[0], c.scenarios_123):
        ax.set_title(split.label, fontweight="bold")

    plt.savefig(saveto)
    plt.close()


## Uncertainty metrics - bar plot
k_to_percentage = lambda k: 100*erf(k/np.sqrt(2))
def add_coverage_k_lines(*axs: Iterable[plt.Axes], klim: int=3) -> None:
    """
    Add horizontal lines at k=1, k=2, ... coverage.
    """
    for k in range(1, klim+1):
        percentage = k_to_percentage(k)

        # Plot lines
        for ax in axs:
            ax.axhline(percentage, color="black", linestyle="--", zorder=4)

        # Add text to last panel
        ax = axs[-1]
        ax.text(1.01, percentage/100, f"$k = {k}$", transform=ax.transAxes, horizontalalignment="left", verticalalignment="center")

def plot_coverage(data: pd.DataFrame, *,
                  groups: Iterable[c.Parameter]=c.iops, groupmembers: Iterable[c.Parameter]=c.networks, scenarios: Iterable[c.Parameter]=c.scenarios_123,
                  saveto: Path | str=c.output_path/"uncertainty_coverage.pdf") -> None:
    """
    Bar plot showing the coverage factor (pre-calculated).
    """
    # Constants
    bar_width = 0.15

    # Generate figure ; rows are metrics, columns are split types
    n_groups = len(groups)
    n_members = len(groupmembers)
    n_rows = 1
    n_cols = len(scenarios)
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(11, 4), sharex=True, sharey="row", squeeze=False)

    # Plot results
    for ax_row in axs:
        for ax, scenario in zip(ax_row, scenarios):
            for member_idx, member in enumerate(groupmembers):
                # Select data
                df = data.loc[scenario, member, groups]
                values = df[c.coverage]

                color = member.color
                label = member.label

                locations = np.arange(n_groups) - (bar_width * (n_members - 1) / 2) + member_idx * bar_width

                ax.bar(locations, values, color=color, label=label, width=bar_width, zorder=3)  # Draw points

            ax.grid(True, axis="y", linestyle="--", linewidth=0.5, color="black", alpha=0.4)

        add_coverage_k_lines(*ax_row)

    # Label variables
    axs[0, 0].set_xticks(np.arange(n_groups))
    axs[0, 0].set_xticklabels([p.label_2lines for p in groups])

    # Label y-axes
    axs[0, 0].set_ylabel(c.coverage.label, fontsize=12)
    fig.align_ylabels()

    # y-axis limits
    axs[0, 0].set_ylim(c.coverage.vmin, c.coverage.vmax)

    # Titles
    for ax, scenario in zip(axs[0], scenarios):
        ax.set_title(scenario.label)

    # Plot legend outside the subplots
    _add_legend_below_figure(fig, groupmembers)

    plt.tight_layout()
    plt.savefig(saveto, bbox_inches="tight")
    plt.close()


## Uncertainty metrics - miscalibration area
def table_miscalibration_area(df: pd.DataFrame, *,
                              scenarios: Iterable[c.Parameter]=c.scenarios_123,
                              saveto: Path | str=c.output_path/"miscalibration_area.csv") -> None:
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

    areas.to_csv(saveto)


## Uncertainty metrics - calibration curves
def _plot_calibration_single(ax: plt.Axes, data: pd.Series, **kwargs) -> None:
    """
    Plot a single calibration curve.
    Accounts for the desired order (expected on the y axis, observed on the x axis).
    """
    ax.plot(data.array, data.index, **kwargs)


def plot_calibration_curves(calibration_curves: pd.DataFrame, *,
                            rows: Iterable[c.Parameter]=c.scenarios_123, columns: Iterable[c.Parameter]=c.iops, groupmembers: Iterable[c.Parameter]=c.networks,
                            saveto: Path | str=c.output_path/"calibration_curves.pdf") -> None:
    """
    Plot calibration curves (expected vs. observed).
    """
    # Create figure
    fig, axs = plt.subplots(nrows=len(rows), ncols=len(columns), sharex=True, sharey=True, figsize=(12, 6), layout="constrained", squeeze=False)

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
    _add_legend_below_figure(fig, groupmembers)

    # Labels
    for ax, title in zip(axs[0], columns):
        ax.set_title(title.label)

    for ax, label in zip(axs[:, 0], rows):
        ax.set_ylabel(label.label)

    for ax in axs[-1]:
        ax.set_xlabel(None)

    fig.supxlabel("Observed proportion in interval", fontweight="bold")
    fig.supylabel("Expected proportion in interval", fontweight="bold")
    fig.align_ylabels()

    plt.savefig(saveto, bbox_inches="tight")
    plt.close()
