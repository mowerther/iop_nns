"""
Plots that show the uncertainty of estimates.
"""
from functools import partial
import itertools
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd
from scipy.special import erf

from matplotlib import pyplot as plt
from matplotlib import transforms

from .. import constants as c
from .common import IOP_SCALE, _dataframe_to_string, _heatmap, _plot_grouped_values, add_legend_below_figure, saveto_append_tag, title_type_for_scenarios
from .common import dash, median_with_confidence_interval, print_metric_range


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
    fig.supxlabel("In situ value")
    fig.supylabel("Median uncertainty [%]")
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
    # Setup
    title_type = title_type_for_scenarios(scenarios)

    # Create figure
    nrows = len(uncertainties)
    ncols = len(scenarios)
    width, height = 3.7*ncols, 3*nrows
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(width, height), gridspec_kw={"wspace": -1, "hspace": 0}, layout="constrained", squeeze=False)

    # Plot data
    _heatmap(axs, data, rowparameters=uncertainties, colparameters=scenarios, datarowparameters=c.networks, datacolparameters=variables, apply_titles=title_type)

    # Labels
    fig.supxlabel("IOPs")
    fig.supylabel("Models\n")

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
    # Setup
    title_type = title_type_for_scenarios(scenarios)

    # Find recalibrated uncertainties
    uncertainties_recal = [param for param in uncertainties if param is not c.ale_frac]

    # Create figure
    nrows_data = len(uncertainties)
    nrows_recal = len(uncertainties_recal)
    nrows = nrows_data + nrows_recal
    ncols = len(scenarios)
    ncols_data = len(variables)
    width = max(0.5*ncols*ncols_data, 9.6)
    height = 2.2*nrows
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(width, height), gridspec_kw={"wspace": -1, "hspace": 0}, layout="constrained", squeeze=False)

    # Plot data
    _heatmap(axs[:nrows_data], data, rowparameters=uncertainties, colparameters=scenarios, datarowparameters=c.networks, datacolparameters=variables, apply_titles=title_type)
    _heatmap(axs[nrows_data:], data_recal, rowparameters=uncertainties_recal, colparameters=scenarios, datarowparameters=c.networks, datacolparameters=variables, apply_titles=False, colorbar_tag="\n(Recalibrated)")

    # Labels
    fig.supxlabel("IOPs")
    fig.supylabel("Models")

    saveto = saveto_append_tag(saveto, tag)
    plt.savefig(saveto)
    plt.close()



### COVERAGE
## Helper function: Lines at k=1, 2, ...
k_to_percentage = lambda k: 100*erf(k/np.sqrt(2))

def add_coverage_k_lines(*axs: Iterable[plt.Axes], klim: int=2) -> None:
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
    # Setup
    title_type = title_type_for_scenarios(scenarios)

    # Generate figure ; columns are split types
    n_rows = 1
    n_cols = len(scenarios)
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3.5*n_cols, 4), sharex=True, sharey=True, squeeze=False)

    # Plot
    _plot_grouped_values(axs, data, colparameters=scenarios, rowparameters=[c.coverage], groups=c.iops, groupmembers=c.networks, apply_titles=title_type)

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
                             scenarios: Iterable[c.Parameter]=c.scenarios_123, groups: Iterable[c.Parameter]=c.iops,
                             saveto: Path | str=c.output_path/"uncertainty_coverage_recal.pdf", tag: Optional[str]=None) -> None:
    """
    Box plot showing the coverage factor (pre-calculated).
    """
    # Setup
    title_type = title_type_for_scenarios(scenarios)

    # Generate figure ; columns are split types
    nrows = 2
    ncols = len(scenarios)
    ngroups = len(groups)
    width = max(0.5*ngroups*ncols, 8.5)  # Dynamic width, with a minimum value
    fig = plt.figure(figsize=(width, 5.7), layout="constrained")
    subfigs = fig.subfigures(nrows=nrows, hspace=0.1)
    labels = ["Without recalibration", "With recalibration"]

    # Plot
    for sfig, df, title in zip(subfigs, [data, data_recal], labels):
        axs = sfig.subplots(nrows=1, ncols=ncols, sharex=True, sharey=True, squeeze=False)
        _plot_grouped_values(axs, df, colparameters=scenarios, rowparameters=[c.coverage], groups=groups, groupmembers=c.networks, apply_titles=title_type)

        # k lines
        add_coverage_k_lines(*axs[0])

        # Recal labels
        sfig.suptitle(title)

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
@plt.rc_context({"axes.titleweight": "bold"})
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
    metric_diff = c.Parameter(f"{metric.name}_diff", f"{metric.label}\ndifference", cmap=c.cmap_difference, symmetric=True, vmin=-diff_range, vmax=diff_range)
    data_diff = data_recal.loc[metric] - data.loc[metric]
    data_diff = pd.concat({metric_diff.name: data_diff})

    # Create figure
    nrows = len(c.networks)
    nrows_data = len(scenarios)
    ncols = 3
    ncols_data = len(variables)
    width = max(0.72*ncols*ncols_data, 8)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(width, 14), gridspec_kw={"hspace": 0.25, "wspace": 0.02})

    # Plot main metric  --  note that axs is being transposed, so rows <--> columns
    plot_heatmap = partial(_heatmap, colparameters=c.networks, datarowparameters=scenarios, datacolparameters=variables, colorbar_location="bottom", precision=precision, colorbar_tick_rotation=45)
    plot_heatmap(axs[np.newaxis, :, 0], data, rowparameters=[metric], apply_titles=False)
    plot_heatmap(axs[np.newaxis, :, 1], data_recal, rowparameters=[metric], apply_titles=True)
    plot_heatmap(axs[np.newaxis, :, 2], data_diff, rowparameters=[metric_diff], apply_titles=False)

    # Labels
    for ax, label in zip(axs[0], labels):
        ax.set_title(f"{label}\n{ax.get_title()}")

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
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.tick_params(axis="x", rotation=90)

    # Labels
    for ax, title in zip(axs[0], columns):
        ax.set_title(title.label)

    for ax, label in zip(axs[:, 0], rows):
        ax.set_ylabel(label.label_2lines)

    for ax in axs[-1]:
        ax.set_xlabel(None)

    fig = axs[0, 0].figure  # Assume all axs are in the same figure
    fig.supxlabel("Observed proportion in interval")
    fig.supylabel("Expected proportion in interval")
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
    # Setup: choose horizontal or vertical stacking
    STACK_HORIZONTALLY = (len(columns) < 3)
    figsize = (4*len(columns), 1.5*len(rows)) if STACK_HORIZONTALLY else (2*len(columns), 4*len(rows))
    subfig_layout = (1, 2) if STACK_HORIZONTALLY else (2, 1)

    # Create figure
    fig = plt.figure(figsize=figsize, layout="constrained")
    subfigs = fig.subfigures(*subfig_layout, hspace=1.0/fig.get_figheight(), wspace=1.0/fig.get_figwidth())
    labels = ["Without recalibration", "With recalibration"]

    nrows = len(rows)
    ncols = len(columns)

    # Plot
    for sfig, df, title in zip(subfigs, [calibration_curves, calibration_curves_recal], labels):
        axs = sfig.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, squeeze=False)
        _plot_calibration_curves_base(axs, df, rows, columns, groupmembers)

        # Recal labels
        sfig.suptitle(title, fontweight="bold", fontsize=12)

    # Remove duplicate labels for horizontal plot
    if STACK_HORIZONTALLY:
        fig.supxlabel(subfigs[0].get_supxlabel())
        fig.supylabel(subfigs[0].get_supylabel())
        for sfig in subfigs:
            sfig.supxlabel(None)
            sfig.supylabel(None)

    # Plot legend outside the subplots
    add_legend_below_figure(fig, c.networks)

    saveto = saveto_append_tag(saveto, tag)
    plt.savefig(saveto, bbox_inches="tight")
    plt.close()


### PRINT STATISTICS
## Ratios between scenarios
def compare_uncertainty_scenarios_123(data: pd.DataFrame, *,
                                      uncertainty: c.Parameter=c.total_unc_pct) -> None:
    """
    Compare the predicted uncertainty across the three scenarios.
    `data` should be averaged uncertainties.
    """
    data = data.loc[uncertainty]
    ratios = data.loc[[c.wd, c.ood]] / data.loc[c.random_split]
    print(dash)
    print(f"Median {uncertainty.label}, ratio between scenario and {c.random_split}:")
    print(_dataframe_to_string(ratios))
    print()

## Coverage range
print_coverage_range = partial(print_metric_range, metric=c.coverage)

## Improvement from recalibration
def calculate_number_improved(differences: pd.DataFrame, group_func: Callable, *, improve_is_increase=False) -> pd.DataFrame:
    """
    Calculate the percentage of improved values.
    If `improve_is_increase` is True (default: False), then an increase in the statistic is treated as an improvement; if False, a decrease.
        Use False for MdSA, Miscalibration area, etc. Use True for R², etc.
    """
    number_total = group_func(differences).count()
    number_decreased = group_func(differences < 0).sum()  # sum adds up the Trues
    number_increased = group_func(differences > 0).sum()  # sum adds up the Trues

    number_improved = number_increased if improve_is_increase else number_decreased  # Get inverse if increase is desired
    percentage_improved = 100 * number_improved / number_total

    percentage_improved.name = "% improved"

    return number_total, number_improved, percentage_improved

def recalibration_improvement(metrics: pd.DataFrame, metrics_recal: pd.DataFrame, *,
                              statistic: c.Parameter=c.miscalibration_area, improve_is_increase=False,
                              variables: Iterable[c.Parameter]=c.iops) -> None:
    """
    Compare the metrics before and after recalibration and see if a certain statistic (default: miscalibration area) increased or decreased.
    If `improve_is_increase` is True (default: False), then an increase in the statistic is treated as an improvement; if False, a decrease.
        Use False for MdSA, Miscalibration area, etc. Use True for R², etc.
    """
    print()
    print(dash)
    _group = partial(pd.DataFrame.groupby, by=c.scenario_network, sort=False)

    diff = metrics_recal[statistic] - metrics[statistic]
    diff = diff.unstack()[variables]

    # Median change
    median_diff = _group(diff).median()
    print(f"Median change in {statistic} with recalibration:")
    print(_dataframe_to_string(median_diff))
    print()

    # Fraction improved - per scenario/network/variable
    number_total, number_improved, percentage_improved = calculate_number_improved(diff, _group, improve_is_increase=improve_is_increase)
    print(f"Percentage of models where {statistic} improved with recalibration:")
    print(_dataframe_to_string(percentage_improved))

    # Fraction increased: group by individual levels
    for level in ["scenario", "network"]:
        # Median change
        diff_level = diff.stack().groupby(level, sort=False)
        diff_level = median_with_confidence_interval(diff_level)
        print(f"Median change in {statistic} with recalibration:")
        print(_dataframe_to_string(diff_level, precision=3))
        print()

        # Percentage improved
        _aggregate = lambda df: df.groupby(level, sort=False).sum().sum(axis=1)
        percentage_improved_level = 100 * _aggregate(number_improved) / _aggregate(number_total)
        print(f"Percentage of models where {statistic} improved with recalibration:")
        print(_dataframe_to_string(percentage_improved_level))
        print()

    print(dash)


recalibration_change_in_mdsa = partial(recalibration_improvement, statistic=c.mdsa, improve_is_increase=False)


## Threshold for applying recalibration
def bin_by_column(data: pd.DataFrame, column: c.Parameter, *,
                  vmin: Optional[float]=None, vmax: Optional[float]=None, binsize: float=0.01, N: Optional[int]=None, log=False) -> Callable:
    """
    Generate a function for grouping data into bins, as determined by one column (e.g. an IOP or a metric).
    If N is specified, use a linear/logarithmic spacing.
    If vmin and vmax are not specified, use the information from the Parameter object.
    """
    # Determine min/max if none were provided
    if vmin is None:
        vmin = column.vmin
    if vmax is None:
        vmax = column.vmax

    # Log-transform if desired
    if log:
        vmin, vmax = np.log10(vmin), np.log10(vmax)

    # Create bins
    if N is None:
        func = partial(np.arange, step=binsize)
    else:
        func = partial(np.linspace, num=N)
    bins = func(vmin, vmax)
    if log:
        bins = 10**bins

    # Create cut and groupby function
    cut = pd.cut(data[column], bins)
    group = partial(pd.DataFrame.groupby, by=cut, observed=False, sort=True)

    return bins, cut, group

def plot_recalibration_MA(metrics: pd.DataFrame, metrics_recal: pd.DataFrame, *,
                          binsize: float=0.01,
                          saveto: str | Path=c.output_path/"miscalibration_scatter.pdf", tag: Optional[str]=None) -> None:
    """
    Based on the without- and with-recalibration metrics, plot the miscalibration improvement as a function of original miscalibration area.
    """
    # Setup: data
    bins, cut, group = bin_by_column(metrics, c.miscalibration_area, vmax=1, binsize=binsize)
    diff = metrics_recal[c.miscalibration_area] - metrics[c.miscalibration_area]
    diff_binned = group(diff)
    diff_median_ci = median_with_confidence_interval(diff_binned)

    # Setup: figure
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), layout="constrained")

    # Plot data
    x = diff_median_ci.index.categories.mid
    ax.scatter(metrics[c.miscalibration_area], diff, s=1, color="C2", rasterized=True, label="Individual\ncomparisons")
    ax.plot(x, diff_median_ci["median"], linewidth=3, label="Binned median", color="black")
    ax.fill_between(x, diff_median_ci["ci_lower"], diff_median_ci["ci_upper"], alpha=0.7, label="Binned CI", color="grey", edgecolor="black")
    ax.axhline(0, color="black", linewidth=2)

    # Labels
    ax.set_xlabel(f"{c.miscalibration_area.label}\n(without recalibration)")
    ax.set_ylabel(f"Difference in {c.miscalibration_area.label.lower()}")
    ax.set_xlim(0, metrics[c.miscalibration_area].max())
    ax.set_aspect("equal")
    ax.legend(loc="upper right", scatterpoints=5)

    # Save
    saveto = saveto_append_tag(saveto, tag)
    plt.savefig(saveto, bbox_inches="tight")
    plt.close()


def recalibration_MA_threshold(metrics: pd.DataFrame, metrics_recal: pd.DataFrame, *, binsize: float=0.01) -> None:
    """
    Based on the without- and with-recalibration metrics, determine the threshold miscalibration area at which recalibration becomes useful.
    """
    # Setup
    *_, group = bin_by_column(metrics, c.miscalibration_area, vmax=1, binsize=binsize)

    # Calculate statistics
    diff = metrics_recal[c.miscalibration_area] - metrics[c.miscalibration_area]
    diff_binned = group(diff)
    diff_median_ci = median_with_confidence_interval(diff_binned)

    # Fraction improved
    number_total, number_improved, percentage_improved = calculate_number_improved(diff, group, improve_is_increase=False)

    # Combine results
    results = diff_median_ci.copy()
    results[percentage_improved.name] = percentage_improved

    # Find threshold indices
    median_improved = (results["median"] < 0).idxmax()  # Index of first occurence of the maximum (True)
    ci_upper_improved = (results["ci_upper"] < 0).idxmax()
    percentage_over_50 = (results[percentage_improved.name] > 50).idxmax()  # Essentially equivalent to median_improved
    percentage_over_75 = (results[percentage_improved.name] > 75).idxmax()
    threshold = ci_upper_improved.left  # Lower limit of interval

    print()
    print(dash)
    print(f"{c.miscalibration_area.label} -- First interval in which")
    print(f"The median is < 0: {median_improved}")
    print(f"The upper CI is < 0: {ci_upper_improved}")
    print(f"More than 50% are improved: {percentage_over_50}")
    print(f"More than 75% are improved: {percentage_over_75}")
    print(f"Recommended threshold: {threshold}")
    print()

    # Coverage above and below threshold  -- note: this needs to be improved because it currently assumes a 1-to-1 relationship between without- and with-recalibration models, which is not correct
    coverage_below = median_with_confidence_interval(metrics_recal.loc[metrics[c.miscalibration_area] <= threshold][[c.coverage]])
    coverage_above = median_with_confidence_interval(metrics_recal.loc[metrics[c.miscalibration_area] >  threshold][[c.coverage]])
    print(f"Coverage below threshold:\n{coverage_below}")
    print()
    print(f"Coverage above threshold:\n{coverage_above}")

    print(dash)
