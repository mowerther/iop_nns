"""
Anything that needs to be shared between modules.
"""
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors, patches, ticker
import matplotlib as mpl

from .. import constants as c


### UNIVERSAL RCPARAMS
mpl.rcParams.update({
                    "figure.dpi": 300,
                    "axes.grid": True,
                    "grid.linestyle": "--",
                    "grid.alpha": 0.5,
                    })


### OTHER CONSTANTS
IOP_LIMS = (1e-5, 1e1)
IOP_SCALE = "log"
IOP_TICKS = 10**np.arange(np.log10(IOP_LIMS[0]), np.log10(IOP_LIMS[1])+0.01)  # 10^-5, 10^-4, ..., 10^1


### HELPER FUNCTIONS
def add_legend_below_figure(fig: plt.Figure, items: Iterable[c.Parameter], **kwargs) -> None:
    """
    Add a legend below the subplots, with a patch for each item.
    """
    legend_content = [patches.Patch(color=key.color, label=key.label, **kwargs) for key in items]
    fig.legend(handles=legend_content, loc="upper center", bbox_to_anchor=(0.5, 0), ncols=len(items), framealpha=1, edgecolor="black")


def get_axes_size(ax: plt.Axes) -> tuple[float, float]:
    """
    Get the size (in inches) of an Axes object.
    """
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    return bbox.width, bbox.height


def label_topleft(ax: plt.Axes, text: str, **kwargs) -> None:
    """
    Add a label to the top left corner of a panel.
    """
    ax.text(0.05, 0.90, text, transform=ax.transAxes, horizontalalignment="left", verticalalignment="top", fontsize=12, color="black", bbox={"facecolor": "white", "edgecolor": "black", "boxstyle": "round"})

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


### PLOTTING FUNCTIONS
## Grouped metrics - useful for boxplots, lollipops, etc.
def _plot_grouped_values(axs: np.ndarray[plt.Axes], data: pd.DataFrame,
                         colparameters: Iterable[c.Parameter], rowparameters: Iterable[c.Parameter],
                         groups: Iterable[c.Parameter], groupmembers: Iterable[c.Parameter],
                         *,
                         spacing: float=0.15, ylim_quantile: float=0.03, apply_titles=True) -> None:
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
    # Check inputs
    assert 1


    # Metadata
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

                bplot = ax.boxplot(df, positions=locations, whis=(0, 100), widths=spacing*0.9, zorder=3, medianprops={"color": "black"}, patch_artist=True)
                for patch in bplot["boxes"]:
                    patch.set_facecolor(member.color)

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
        ax.set_ylabel(rowparam.label_2lines, fontsize=12)
    fig.align_ylabels()

    # Titles
    if apply_titles:
        for ax, colparam in zip(axs[0], colparameters):
            ax.set_title(colparam.label_2lines)


## Heatmap, e.g. for median uncertainty
def _heatmap(axs: np.ndarray[plt.Axes], data: pd.DataFrame,
             rowparameters: Iterable[c.Parameter], colparameters: Iterable[c.Parameter],
             datarowparameters: Iterable[c.Parameter], datacolparameters: Iterable[c.Parameter],
             *,
             apply_titles=True, remove_ticks=True, precision: int=0,
             colorbar_location: str="right", colorbar_pad: Optional[float]=None, colorbar_shrink: Optional[float]=None, colorbar_tag: Optional[str]="") -> None:
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
        cb.update_ticks()

    # Labels
    wrap_labels = lambda parameters: list(zip(*enumerate(p.label_2lines for p in parameters)))  # (0, 1, ...) (label0, label1, ...)
    for ax in axs[-1]:
        ax.set_xticks(*wrap_labels(datacolparameters))

    for ax in axs[:, 0]:
        ax.set_yticks(*wrap_labels(datarowparameters))

    if apply_titles:
        for ax, colparam in zip(axs[0], colparameters):
            ax.set_title(colparam.label_2lines)

    # Remove ticks
    if remove_ticks:
        for ax in axs.ravel():
            # Check for unlabelled xticks
            if not ax.xaxis.get_majorticklabels():
                ax.tick_params(axis="x", which="both", bottom=False)
            # Check for unlabelled yticks
            if not ax.yaxis.get_majorticklabels():
                ax.tick_params(axis="y", which="both", left=False)
