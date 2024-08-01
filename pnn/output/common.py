"""
Anything that needs to be shared between modules.
"""
from typing import Iterable

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
def _add_legend_below_figure(fig: plt.Figure, items: Iterable[c.Parameter], **kwargs) -> None:
    """
    Add a legend below the subplots, with a patch for each item.
    """
    legend_content = [patches.Patch(color=key.color, label=key.label, **kwargs) for key in items]
    fig.legend(handles=legend_content, loc="upper center", bbox_to_anchor=(0.5, 0), ncols=len(items), framealpha=1, edgecolor="black")
