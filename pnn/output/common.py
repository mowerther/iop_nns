"""
Anything that needs to be shared between modules.
"""
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import patches
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


### HELPER FUNCTIONS
def add_legend_below_figure(fig: plt.Figure, items: Iterable[c.Parameter], **kwargs) -> None:
    """
    Add a legend below the subplots, with a patch for each item.
    """
    legend_content = [patches.Patch(color=key.color, label=key.label, **kwargs) for key in items]
    fig.legend(handles=legend_content, loc="upper center", bbox_to_anchor=(0.5, 0), ncols=len(items), framealpha=1, edgecolor="black")


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
        - `groups`, level=2  --  groups within each panel
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
