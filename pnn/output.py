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
plt.style.use("default")
from matplotlib import ticker, patches
from matplotlib.colors import Normalize, to_rgba

from . import metrics
from . import constants as c


### HELPER FUNCTIONS
def _add_legend_below_figure(fig: plt.Figure, items: Iterable[c.Parameter], **kwargs) -> None:
    """
    Add a legend below the subplots, with a patch for each item.
    """
    legend_content = [patches.Patch(color=key.color, label=key.label, **kwargs) for key in items]
    fig.legend(handles=legend_content, loc="upper center", bbox_to_anchor=(0.5, 0), ncols=len(items), framealpha=1, edgecolor="black")


### FUNCTIONS
## Input data - full
def plot_full_dataset(df: pd.DataFrame, *,
                      variables: Iterable[c.Parameter]=c.iops,
                      title: Optional[str]=None,
                      saveto: Path | str=c.output_path/"full_dataset.pdf") -> None:
    """
    Plot the full input dataset, with separate panels for each IOP.
    """
    # Constants
    lims = (1e-5, 1e1)
    bins = np.logspace(np.log10(lims[0]), np.log10(lims[1]), 50)
    scale = "log"
    ncols = 2
    nrows = len(variables) // ncols

    # Create figure
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(12, 6), squeeze=False, layout="constrained")

    # Plot data per row
    for ax, var in zip(axs.ravel(), variables):
        # Plot data
        data = df[var]
        ax.hist(data, bins=bins, color=var.color, histtype="stepfilled")

        # Panel settings
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.text(0.05, 0.90, var.label, transform=ax.transAxes, horizontalalignment="left", verticalalignment="top", fontsize=12, color="black", bbox={"facecolor": "white", "edgecolor": "black", "boxstyle": "round"})


    # Panel settings
    axs[0, 0].set_xscale(scale)
    axs[0, 0].set_xlim(*lims)

    # Labels
    fig.supxlabel("In situ value", fontweight="bold")
    fig.supylabel("Frequency", fontweight="bold")

    # Save result
    plt.savefig(saveto, dpi=200, bbox_inches="tight")
    plt.close()


## Input data - Random/WD/OOD splits
traincolor, testcolor = "black", "C1"
def plot_data_splits(train_sets: Iterable[pd.DataFrame], test_sets: Iterable[pd.DataFrame], *,
                     variables: Iterable[c.Parameter]=c.iops_443, scenarios: Iterable[c.Parameter]=c.scenarios_123,
                     title: Optional[str]=None,
                     saveto: Path | str=c.output_path/"scenarios.pdf") -> None:
    """
    Plot the data splits (random/within-distribution/out-of-distribution).
    Datasets are expected in alternating train/test order, e.g. train-random, test-random, train-wd, test-wd, train-ood, test-ood.
    One column per variable (default: IOPs at 443 nm).
    """
    # Constants
    lims = (1e-5, 1e1)
    bins = np.logspace(np.log10(lims[0]), np.log10(lims[1]), 35)
    scale = "log"
    ncols = len(variables)
    nrows = len(scenarios)

    # Checks
    assert len(train_sets) == len(test_sets) == len(scenarios), f"Mismatch between number of scenarios ({len(scenarios)}), number of training datasets ({len(train_sets)}), and number of test datasets ({len(test_sets)})."

    # Create figure
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(12, 7), squeeze=False, layout="constrained", gridspec_kw={"hspace": 0.15})

    # Plot data per row
    for ax_row, df_train, df_test in zip(axs, train_sets, test_sets):
        for ax, var in zip(ax_row, variables):
            # Plot data
            ax.hist(df_train[var], bins=bins, color=traincolor, histtype="step")
            ax.hist(df_train[var], bins=bins, color=traincolor, histtype="stepfilled", alpha=0.5)
            ax.hist(df_test[var], bins=bins, color=testcolor, histtype="step")
            ax.hist(df_test[var], bins=bins, color=testcolor, histtype="stepfilled", alpha=0.5)

            # Panel settings
            ax.grid(True, linestyle="--", alpha=0.5)


    # Panel settings
    axs[0, 0].set_xscale(scale)
    axs[0, 0].set_xlim(*lims)

    # Legend
    trainpatch = patches.Patch(facecolor=to_rgba(traincolor, 0.5), edgecolor=traincolor, label="Train")
    testpatch = patches.Patch(facecolor=to_rgba(testcolor, 0.5), edgecolor=testcolor, label="Test")
    axs[0, 0].legend(handles=[trainpatch, testpatch], loc="upper left", framealpha=1, edgecolor="black")

    # Labels
    for scenario, ax in zip(scenarios, axs[:, ncols//2]):
        ax.set_title(scenario.label)
    for var, ax in zip(variables, axs[-1]):
        ax.set_xlabel(var.label)

    fig.supxlabel("In situ value", fontweight="bold")
    fig.supylabel("Frequency", fontweight="bold")

    # Save result
    plt.savefig(saveto, dpi=200, bbox_inches="tight")
    plt.close()



## Performance (matchups) - scatter plot, per algorithm/scenario combination
def plot_performance_scatter_single(df: pd.DataFrame, *,
                                    columns: Iterable[c.Parameter]=c.iops, rows: Iterable[c.Parameter]=[c.total_unc_pct, c.ale_frac],
                                    title: Optional[str]=None,
                                    saveto: Path | str="scatterplot.pdf") -> None:
    """
    Plot one DataFrame with y, y_hat, with total uncertainty (top) or aleatoric fraction (bottom) as colour.
    """
    # Constants
    lims = (1e-4, 1e1)
    scale = "log"

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
        r_square = f"$R^2 = {metrics.log_r_squared(y, y_hat):.2f}$"
        sspb = f"SSPB = ${metrics.sspb(y, y_hat):+.1f}$%"
        other_metrics = [f"{func.__name__} = {func(y, y_hat):.1f}%" for func in [metrics.mdsa, metrics.mape]]

        # Format
        metrics_text = "\n".join([r_square, sspb, *other_metrics])
        ax.text(0.95, 0.03, metrics_text, transform=ax.transAxes, horizontalalignment="right", verticalalignment="bottom", color="black", size=9, bbox={"facecolor": "white", "edgecolor": "black"})

    # Plot settings
    axs[0, 0].set_xscale(scale)
    axs[0, 0].set_yscale(scale)
    axs[0, 0].set_xlim(*lims)
    axs[0, 0].set_ylim(*lims)

    # Labels
    for ax, iop in zip(axs[0], columns):
        ax.set_title(iop.label)
    fig.supxlabel("In-situ (actual)", fontsize="x-large", fontweight="bold")
    fig.supylabel("Model estimate", x=-0.02, fontsize="x-large", fontweight="bold")
    fig.suptitle(title, fontsize="x-large")

    # Save result
    plt.savefig(saveto, dpi=200, bbox_inches="tight")
    plt.close()

_scatter_levels = ["split", "network"]
def plot_performance_scatter(results: pd.DataFrame, *,
                             saveto: Path | str=c.supplementary_path/"scatterplot.pdf", **kwargs) -> None:
    """
    Plot many DataFrames with y, y_hat, with total uncertainty (top) or aleatoric fraction (bottom) as colour.
    """
    saveto = Path(saveto)

    # Loop over results and plot each network/split combination in a separate figure
    for (splitkey, networkkey), df in results.groupby(level=_scatter_levels):
        # Set up labels
        saveto_here = saveto.with_stem(f"{saveto.stem}_{networkkey}-{splitkey}")
        split, network = c.scenarios_123_fromkey[splitkey], c.networks_fromkey[networkkey]

        # Plot
        plot_performance_scatter_single(df, title=f"{network.label} {split.label}", saveto=saveto_here, **kwargs)


## Performance metrics - lollipop plot
_lollipop_metrics = [c.mdsa, c.sspb, c.r_squared]
def plot_performance_metrics_lollipop(data: pd.DataFrame, *,
                                      groups: Iterable[c.Parameter]=c.iops, groupmembers: Iterable[c.Parameter]=c.networks, metrics: Iterable[c.Parameter]=_lollipop_metrics, splits: Iterable[c.Parameter]=c.scenarios_123,
                                      saveto: Path | str=c.output_path/"performance_lolliplot_vertical.pdf") -> None:
    """
    Plot some number of DataFrames containing performance metric statistics.
    """
    # Constants
    bar_width = 0.15

    # Generate figure ; rows are metrics, columns are split types
    n_groups = len(groups)
    n_members = len(groupmembers)
    n_rows = len(metrics)
    n_cols = len(splits)
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 7), sharex=True, sharey="row", squeeze=False)

    # Plot results; must be done in a loop because there is no Pandas lollipop function
    for ax_row, metric in zip(axs, metrics):
        for ax, split in zip(ax_row, splits):
            for member_idx, member in enumerate(groupmembers):
                # Select data
                df = data.loc[split, member, metric]
                values = df[groups]

                color = member.color
                label = member.label

                locations = np.arange(n_groups) - (bar_width * (n_members - 1) / 2) + member_idx * bar_width

                ax.scatter(locations, values, color=color, label=label, s=50, zorder=3, edgecolor="black")  # Draw points
                ax.vlines(locations, 0, values, colors='grey', lw=1, alpha=0.7)

            ax.grid(True, axis="y", linestyle="--", linewidth=0.5, color="black", alpha=0.4)

    # Label variables
    axs[0, 0].set_xticks(np.arange(n_groups))
    axs[0, 0].set_xticklabels([p.label_2lines for p in groups])

    # Label y-axes
    for ax, metric in zip(axs[:, 0], metrics):
        ax.set_ylabel(metric.label, fontsize=12)
    fig.align_ylabels()

    # y-axis limits; currently hardcoded
    for ax, metric in zip(axs[:, 0], metrics):
        if metric.symmetric:
            maxvalue = np.abs(ax.get_ylim()).max()
            ax.set_ylim(-maxvalue, maxvalue)
        else:
            ax.set_ylim(metric.vmin, metric.vmax)

    # Titles
    for ax, split in zip(axs[0], splits):
        ax.set_title(split.label)

    # Plot legend outside the subplots
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), framealpha=1, edgecolor="black")

    plt.tight_layout()
    plt.savefig(saveto, dpi=200, bbox_inches="tight")
    plt.close()


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
                  groups: Iterable[c.Parameter]=c.iops, groupmembers: Iterable[c.Parameter]=c.networks, splits: Iterable[c.Parameter]=c.scenarios_123,
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
    n_cols = len(splits)
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(11, 4), sharex=True, sharey="row", squeeze=False)

    # Plot results
    for ax_row in axs:
        for ax, split in zip(ax_row, splits):
            for member_idx, member in enumerate(groupmembers):
                # Select data
                df = data.loc[split, member, c.coverage]
                values = df[groups]

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
    for ax, split in zip(axs[0], splits):
        ax.set_title(split.label)

    # Plot legend outside the subplots
    _add_legend_below_figure(fig, groupmembers)

    plt.tight_layout()
    plt.savefig(saveto, bbox_inches="tight")
    plt.close()


## Uncertainty metrics - miscalibration area
def table_miscalibration_area(df: pd.DataFrame, *, saveto: Path | str=c.output_path/"miscalibration_area.csv") -> None:
    """
    Reorder the miscalibration area table and save it to file.
    To do: fully automate.
    """
    areas = df.loc[:, :, "miscalibration area"]
    areas = areas.reorder_levels(["network", "split"])
    areas.sort_index(inplace=True)
    areas.sort_index(key=lambda x: x.map({model: i for i, model in enumerate(c.networks)}))
    areas.rename(index={model.name: model.label for model in c.networks}, level="network", inplace=True)
    areas.rename(index={split.name: split.label for split in c.splits}, level="split", inplace=True)

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
