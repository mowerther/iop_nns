"""
Plot input data.
"""
from functools import partial
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.colors import to_rgba

from .. import constants as c
from ..data import DataScenario
from .common import IOP_LIMS, IOP_SCALE, IOP_TICKS, label_topleft

### CONSTANTS
TRAIN_COLOR = "black"
TEST_COLOR = "C1"


### HELPER FUNCTIONS
iop_bins = lambda n: np.logspace(np.log10(IOP_LIMS[0]), np.log10(IOP_LIMS[-1]), n)


### PLOT FULL INPUT DATASETS PER IOP
def plot_full_dataset(df: pd.DataFrame, *,
                      variables: Iterable[c.Parameter]=c.iops,
                      title: Optional[str]=None,
                      saveto: Path | str=c.output_path/"full_dataset.pdf") -> None:
    """
    Plot the full input dataset, with separate panels for each IOP.
    """
    # Constants
    bins = iop_bins(50)
    ncols = 2
    nrows = len(variables) // ncols

    # Create figure
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(10, 4), squeeze=False, layout="constrained")

    # Plot data per row
    for ax, var in zip(axs.ravel(), variables):
        # Plot data
        data = df[var]
        ax.hist(data, bins=bins, color=var.color, histtype="stepfilled")

        # Panel settings
        label_topleft(ax, var.label)

    # Panel settings
    axs[0, 0].set_xscale(IOP_SCALE)
    axs[0, 0].set_xlim(*IOP_LIMS)
    axs[0, 0].set_xticks(IOP_TICKS)

    # Labels
    fig.supxlabel("In situ value")
    fig.supylabel("Frequency")

    # Save result
    plt.savefig(saveto, bbox_inches="tight")
    plt.close()



## PLOT TRAIN/TEST SPLITS FOR DIFFERENT SCENARIOS
def plot_scenarios(*data_scenarios: Iterable[DataScenario],
                   variables: Iterable[c.Parameter]=c.iops_443, scenarios: Iterable[c.Parameter]=c.scenarios_insitu,
                   title: Optional[str]=None,
                   saveto: Path | str=c.output_path/"scenarios.pdf") -> None:
    """
    Plot the data splits (e.g. random / within-distribution / out-of-distribution).
    One column per variable (default: IOPs at 443 nm).
    """
    # Constants
    bins = iop_bins(35)
    ncols = len(variables)
    nrows = len(scenarios)

    # Checks
    assert len(data_scenarios) == len(scenarios), f"Mismatch between number of scenarios ({len(scenarios)}) and number of data scenarios provided ({len(data_scenarios)})."

    # Create figure
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(10, 5), squeeze=False, layout="constrained", gridspec_kw={"hspace": 0.12})

    # Plot data per row
    for ax_row, data_scenario in zip(axs, data_scenarios):
        df_train = data_scenario.train_data
        df_test = list(data_scenario.test_scenarios_and_data.values())[0]

        for ax, var in zip(ax_row, variables):
            # Plot data
            ax.hist(df_train[var], bins=bins, color=TRAIN_COLOR, histtype="step")
            ax.hist(df_train[var], bins=bins, color=TRAIN_COLOR, histtype="stepfilled", alpha=0.5)
            ax.hist(df_test[var], bins=bins, color=TEST_COLOR, histtype="step")
            ax.hist(df_test[var], bins=bins, color=TEST_COLOR, histtype="stepfilled", alpha=0.5)

    # Panel settings
    axs[0, 0].set_xscale(IOP_SCALE)
    axs[0, 0].set_xlim(*IOP_LIMS)
    axs[0, 0].set_xticks(IOP_TICKS)

    # Legend
    trainpatch = patches.Patch(facecolor=to_rgba(TRAIN_COLOR, 0.5), edgecolor=TRAIN_COLOR, label="Train")
    testpatch = patches.Patch(facecolor=to_rgba(TEST_COLOR, 0.5), edgecolor=TEST_COLOR, label="Test")
    axs[0, 0].legend(handles=[trainpatch, testpatch], loc="upper left")

    # Labels
    for scenario, ax in zip(scenarios, axs[:, ncols//2]):
        ax.set_title(scenario.label)
    for var, ax in zip(variables, axs[-1]):
        ax.set_xlabel(var.label)

    fig.supxlabel("In situ value")
    fig.supylabel("Frequency")

    # Save result
    plt.savefig(saveto, bbox_inches="tight")
    plt.close()


## PLOT TRAIN/TEST SPLITS FOR PRISMA SCENARIOS
plot_prisma_scenarios = partial(plot_scenarios, scenarios=c.scenarios_prisma_overview, saveto=c.output_path/"scenarios_prisma.pdf")
