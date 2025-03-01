"""
Anything that needs to be shared between modules.
"""
from functools import partial
from pathlib import Path
from string import ascii_lowercase
from typing import Iterable, Optional
import itertools

import numpy as np
import pandas as pd
from pandas.core.groupby import GroupBy

from matplotlib import pyplot as plt
from matplotlib import colors, patches, ticker
import matplotlib as mpl

from .. import constants as c


### UNIVERSAL RCPARAMS
mpl.rcParams.update({
                    "axes.grid": True,
                    "figure.dpi": 300,
                    "figure.labelweight": "bold",
                    "figure.titleweight": "bold",
                    "grid.linestyle": "--",
                    "grid.alpha": 0.5,
                    "legend.edgecolor": "black",
                    "legend.framealpha": 1,
                    })


### OTHER CONSTANTS
IOP_LIMS = (1e-5, 1e1)
IOP_LIMS_PRISMA = (1e-2, 1e1)
IOP_SCALE = "log"
IOP_TICKS = 10**np.arange(np.log10(IOP_LIMS[0]), np.log10(IOP_LIMS[1])+0.01)  # 10^-5, 10^-4, ..., 10^1
dash = "----------"  # nmixx-approved


### HELPER FUNCTIONS
def add_legend_below_figure(fig: plt.Figure, items: Iterable[c.Parameter], **kwargs) -> None:
    """
    Add a legend below the subplots, with a patch for each item.
    """
    legend_content = [patches.Patch(color=key.color, label=key.label, **kwargs) for key in items]
    fig.legend(handles=legend_content, loc="upper center", bbox_to_anchor=(0.5, 0), ncols=len(items))


def get_axes_size(ax: plt.Axes) -> tuple[float, float]:
    """
    Get the size (in inches) of an Axes object.
    """
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    return bbox.width, bbox.height


def _label_in_box(ax: plt.Axes, text: str, x: float, y: float, **kwargs) -> None:
    """
    Add a label to the top left corner of a panel.
    """
    ax.text(x, y, text, transform=ax.transAxes, fontsize=12, color="black", bbox={"facecolor": "white", "edgecolor": "black", "boxstyle": "round"}, **kwargs)

label_topleft = partial(_label_in_box, x=0.05, y=0.90, horizontalalignment="left", verticalalignment="top")
label_bottomright = partial(_label_in_box, x=0.95, y=0.10, horizontalalignment="right", verticalalignment="bottom")

def _ax_or_ravel(ax: plt.Axes | np.ndarray) -> list[plt.Axes]:
    """
    Flatten an array of axes into a list or pack a single ax into a list.
    """
    if isinstance(ax, plt.Axes):
        return [ax, ]
    elif isinstance(ax, np.ndarray):
        return ax.ravel().tolist()
    else:
        return ax

def label_axes_sequentially(*axs: plt.Axes | np.ndarray, labels=ascii_lowercase) -> None:
    """
    Label any number of axes sequentially (a, b, c, ...).
    Split out any iterables of Axes, so that the function can take individual Axes and arrays thereof.
    """
    axs_split = [_ax_or_ravel(ax) for ax in axs]
    axs_split = list(itertools.chain.from_iterable(axs_split))

    assert (n_axs := len(axs_split)) <= (n_label := len(labels)), f"Too many axes ({n_axs}) for number of labels ({n_label})"

    for ax, s in zip(axs_split, labels):
        label_bottomright(ax, s)


def pick_textcolor(cmap: colors.Colormap, value: float) -> str:
    """
    For a given value and colour map, pick the appropriate text colour (white or black) that is most visible.
    """
    facecolor = cmap(value)
    facecolor_brightness = 0.3 * facecolor[0] + 0.6 * facecolor[1] + 0.1 * facecolor[2]
    textcolor = "w" if facecolor_brightness < 0.7 else "k"
    return textcolor


def saveto_append_tag(saveto: Path | str, tag: Optional[str]=None) -> Path:
    """
    Add a tag to the end of a path. Does nothing if `tag` is None.
    """
    saveto = Path(saveto)
    if tag is not None:
        saveto = saveto.with_stem(f"{saveto.stem}_{tag}")
    return saveto


def title_type_for_scenarios(scenarios: Iterable[c.Parameter]) -> int:
    """
    Pick the title type (to be used in _apply_titles) for scenarios.
    """
    return 1 if scenarios is c.scenarios_insitu else 2


### HIDDEN HELPER FUNCTIONS
## Plotting
_label_types = {1: "label", 2: "label_2lines"}
def _apply_titles(axs: Iterable[plt.Axes], parameters: Iterable[c.Parameter] | c.Parameter, label_type: int=2) -> None:
    """
    Apply no titles, 1-line label titles, or 2-line label titles.
    """
    # Do nothing if no titles are desired
    if not label_type:
        return

    # Set up iterable if only one parameter was provided
    if not isinstance(parameters, Iterable):
        parameters = [parameters] * len(axs)

    # Determine attribute
    label_type = _label_types[label_type]

    # Apply
    for ax, param in zip(axs, parameters):
        ax.set_title(getattr(param, label_type))


## Text output
def _format_float(number: float, *, precision: int=1) -> str:
    return f"{number:.{precision}f}"

def _dataframe_to_string(data: pd.DataFrame, *, precision: int=1, **kwargs) -> str:
    float_format = partial(_format_float, precision=precision)
    return data.to_string(float_format=float_format, **kwargs)

def _select_metric(data: pd.DataFrame, metric: c.Parameter, columns: Iterable[c.Parameter]=c.iops) -> pd.DataFrame:
    return data[metric].unstack()[columns]


### PLOTTING FUNCTIONS
## Grouped metrics - useful for boxplots, lollipops, etc.
def _plot_grouped_values(axs: np.ndarray[plt.Axes], data: pd.DataFrame,
                         colparameters: Iterable[c.Parameter], rowparameters: Iterable[c.Parameter],
                         groups: Iterable[c.Parameter], groupmembers: Iterable[c.Parameter],
                         *,
                         spacing: float=0.15, ylim_quantile: float=0.03, apply_titles: int=2) -> None:
    """
    For a DataFrame with at least 3 index levels and 1 column level, plot them as follows:
        - `colparameters`, level=0  --  one column of panels for each
        - `groupmembers`, level=1  --  parameters within each group
        - `groups`, level=3  --  groups within each panel
        - `rowparameters`, columns of DataFrame  --  one row of panels for each

    This function must be called on an existing Axes array, which is modified in-place.
    Do not use this directly; rather, build a function around it that handles figure creation and saving, etc.

    To do:
        - Free order of parameters
        - Generalise for different plotting functions, e.g. plt.Axes.boxplot, plot_lollipop, etc.
    """
    # Setup
    fig = axs[0, 0].figure  # We assume all axes are in the same figure
    n_groups = len(groups)
    n_members = len(groupmembers)


    # Plot values; must be done in a loop because there is no direct function in Pandas
    for ax_row, rowparam in zip(axs, rowparameters):
        for ax, colparam in zip(ax_row, colparameters):
            for member_idx, member in enumerate(groupmembers):
                # Select data
                df = data.loc[colparam, member][rowparam].unstack()
                df = df[groups]  # Re-order

                locations = np.arange(n_groups) - (spacing * (n_members - 1) / 2) + member_idx * spacing

                ax.boxplot(df, positions=locations, whis=(0, 100), widths=spacing*0.9, zorder=3, patch_artist=True,
                           boxprops={"facecolor": member.color, "edgecolor": member.color},
                           capprops={"color": member.color},
                           medianprops={"color": "black"},
                           whiskerprops={"color": member.color},
                           )

            # Panel settings
            ax.grid(False, axis="x")
            ax.axhline(0, color="black", linewidth=1, zorder=1)

        # y-axis limits
        ymin = data[rowparam].quantile(ylim_quantile) if rowparam.vmin is None else rowparam.vmin
        ymax = data[rowparam].quantile(1-ylim_quantile) if rowparam.vmax is None else rowparam.vmax
        if rowparam.symmetric:
            maxvalue = np.max(np.abs([ymin, ymax]))
            ymin, ymax = -maxvalue, maxvalue
        ax_row[0].set_ylim(ymin, ymax)

    # Label variables
    axs[0, 0].set_xticks(np.arange(n_groups))
    axs[0, 0].set_xticklabels([p.label_2lines for p in groups])

    xmin = -spacing * (n_members + 1) / 2
    xmax = n_groups - 1 + spacing * (n_members + 1) / 2
    axs[0, 0].set_xlim(xmin, xmax)

    # Label y-axes
    for ax, rowparam in zip(axs[:, 0], rowparameters):
        ax.set_ylabel(rowparam.label_2lines)
    fig.align_ylabels()

    # Titles
    _apply_titles(axs[0], colparameters, apply_titles)


## Heatmap, e.g. for median uncertainty
def _heatmap(axs: np.ndarray[plt.Axes], data: pd.DataFrame,
             rowparameters: Iterable[c.Parameter], colparameters: Iterable[c.Parameter],
             datarowparameters: Iterable[c.Parameter], datacolparameters: Iterable[c.Parameter],
             *,
             apply_titles: int=2, remove_ticks=True, precision: int=0,
             colorbar_location: str="right", colorbar_pad: Optional[float]=None, colorbar_shrink: Optional[float]=None, colorbar_tag: Optional[str]="", colorbar_tick_rotation: Optional[float]=0) -> None:
    """
    Plot a heatmap with text.
    For a DataFrame with at least 2 index levels and 1 column level, plot them as follows:
        - `rowparameters`, level=0  --  one row of panels for each
        - `colparameters`, level=1  --  one column of panels for each
        - `datarowparameters`, level=2  --  rows in the heatmap
        - `datacolparameters`, columns of DataFrame  -- columns in the heatmap

    `apply_titles` will add titles to the top row in the Axes array.
    `cbar_label_tag` will be appended to the colour bar labels; this can be used e.g. to add "\n(Recalibrated)".

    This function must be called on an existing Axes array, which is modified in-place.
    Do not use this directly; rather, build a function around it that handles figure creation and saving, etc.
    """
    # Setup
    fig = axs[0, 0].figure  # We assume all axes are in the same figure
    data = data.loc[rowparameters, colparameters, datarowparameters][datacolparameters]  # Remove extraneous variables

    # Color bar settings
    VERTICAL = (colorbar_location in ("left", "right"))
    if colorbar_pad is None:  # More padding if at the bottom
        colorbar_pad = 0.01 if VERTICAL else 0.05
    if colorbar_shrink is None:  # Reduce size if at the bottom, to prevent text collisions
        colorbar_shrink = 1. if VERTICAL else 0.95

    # Plot data
    for ax_row, rowparam in zip(axs, rowparameters):
        # Color bar settings
        df_row = data.loc[rowparam]
        vmin = rowparam.vmin if rowparam.vmin is not None else df_row.min().min()
        vmax = rowparam.vmax if rowparam.vmax is not None else df_row.max().max()
        if rowparam.symmetric and (rowparam.vmin is None or rowparam.vmax is None):
            maxvalue = max([-vmin, vmax])
            vmin, vmax = -maxvalue, maxvalue
        norm = colors.Normalize(vmin, vmax)

        for ax, colparam in zip(ax_row, colparameters):
            # Select relevant data
            df = data.loc[rowparam, colparam, datarowparameters][datacolparameters]

            # Plot image
            im = ax.imshow(df, cmap=rowparam.cmap, vmin=vmin, vmax=vmax, aspect="auto")

            # Plot individual values
            for i, x in enumerate(datarowparameters):
                for j, y in enumerate(datacolparameters):
                    # Ensure text is visible
                    value = df.iloc[i, j]
                    textcolor = pick_textcolor(rowparam.cmap, norm(value))

                    # Show text
                    ax.text(j, i, f"${value:.{precision}f}$", ha="center", va="center", color=textcolor)

            # Panel settings
            ax.grid(False)

        # Color bar per row
        cb = fig.colorbar(im, ax=ax_row, location=colorbar_location, fraction=0.1, pad=colorbar_pad, shrink=colorbar_shrink, extend=rowparam.extend_cbar)
        cb.set_label(label=f"{rowparam.label}{colorbar_tag}", weight="bold")
        cb.locator = ticker.MaxNLocator(nbins=5)
        cb.ax.tick_params(rotation=colorbar_tick_rotation)
        cb.update_ticks()

    # Labels
    wrap_labels = lambda parameters: list(zip(*enumerate(p.label_2lines for p in parameters)))  # (0, 1, ...) (label0, label1, ...)
    for ax in axs[-1]:
        ax.set_xticks(*wrap_labels(datacolparameters))

    for ax in axs[:, 0]:
        ax.set_yticks(*wrap_labels(datarowparameters))

    _apply_titles(axs[0], colparameters, apply_titles)

    # Remove ticks
    if remove_ticks:
        for ax in axs.ravel():
            # Check for unlabelled xticks
            if not ax.xaxis.get_majorticklabels():
                ax.tick_params(axis="x", which="both", bottom=False)
            # Check for unlabelled yticks
            if not ax.yaxis.get_majorticklabels():
                ax.tick_params(axis="y", which="both", left=False)


### TEXT OUTPUT
## Compare metrics across scenarios, architectures, etc.
def median_with_confidence_interval(df: pd.DataFrame | GroupBy, **kwargs) -> pd.DataFrame:
    """
    Calculate the median with lower and upper bounds for a 1-sigma confidence interval.
    """
    stats = {"median": df.median(**kwargs),
             "ci_lower": df.quantile(c.k1_lower, **kwargs),
             "ci_upper": df.quantile(c.k1_upper, **kwargs)}
    stats = pd.DataFrame(stats)
    return stats

_relative_std = lambda df: 100 * df.std() / df.median()  # df can also be a groupby
_metric_statistics = {"Minimum": GroupBy.min, "Median": GroupBy.median, "Maximum": GroupBy.max,
                      "Standard deviation relative to the median [%]": _relative_std}
def print_metric_range(metrics_all: pd.DataFrame, metric: c.Parameter, *,
                       scenarios: Iterable[c.Parameter]=c.scenarios_insitu, variables: Iterable[c.Parameter]=c.iops) -> None:
    """
    For one metric, print its median, maximum etc. across the N different model instances.
    """
    # Setup
    data = _select_metric(metrics_all, metric)[variables]
    data_over_instances = data.groupby(c.scenario_network, sort=False)

    print(dash)
    print(f"Statistics for {metric.label}")

    # Statistics over instances
    for label, func in _metric_statistics.items():
        stats_over_instances = func(data_over_instances)
        print()
        print(f"{label} {metric.label}:")
        print(_dataframe_to_string(stats_over_instances))

    # Median by scenario/network/IOP
    for level in ["scenario", "network", "variable"]:
        metrics_by_level = data.stack().groupby(level, sort=False)  # .stack() puts the variables back into the index
        median_and_ci = median_with_confidence_interval(metrics_by_level)
        print()
        print(f"Median {metric.label}, with CI, by {level}:")
        print(_dataframe_to_string(median_and_ci))

    print(dash)
    print()
