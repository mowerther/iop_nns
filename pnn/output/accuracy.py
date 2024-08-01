"""
Plots that show the accuracy of estimates.
"""
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import ticker

from .. import constants as c
from .. import metrics as m
from .common import IOP_LIMS, IOP_SCALE


### SCATTER PLOTS
## Performance (matchups) - scatter plot, per algorithm/scenario combination
def plot_performance_scatter(df: pd.DataFrame, *,
                             columns: Iterable[c.Parameter]=c.iops, rows: Iterable[c.Parameter]=[c.total_unc_pct, c.ale_frac],
                             title: Optional[str]=None,
                             saveto: Path | str="scatterplot.pdf") -> None:
    """
    Plot one DataFrame with y, y_hat, with total uncertainty (top) or aleatoric fraction (bottom) as colour.
    """
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
        r_square = f"$R^2 = {m.log_r_squared(y, y_hat):.2f}$"
        sspb = f"SSPB = ${m.sspb(y, y_hat):+.1f}$%"
        other_metrics = [f"{func.__name__} = {func(y, y_hat):.1f}%" for func in [m.mdsa, m.mape]]

        # Format
        metrics_text = "\n".join([r_square, sspb, *other_metrics])
        ax.text(0.95, 0.03, metrics_text, transform=ax.transAxes, horizontalalignment="right", verticalalignment="bottom", color="black", size=9, bbox={"facecolor": "white", "edgecolor": "black"})

    # Plot settings
    axs[0, 0].set_xscale(IOP_SCALE)
    axs[0, 0].set_yscale(IOP_SCALE)
    axs[0, 0].set_xlim(*IOP_LIMS)
    axs[0, 0].set_ylim(*IOP_LIMS)

    # Labels
    for ax, iop in zip(axs[0], columns):
        ax.set_title(iop.label)
    fig.supxlabel("In-situ (actual)", fontsize="x-large", fontweight="bold")
    fig.supylabel("Model estimate", x=-0.02, fontsize="x-large", fontweight="bold")
    fig.suptitle(title, fontsize="x-large")

    # Save result
    plt.savefig(saveto, bbox_inches="tight")
    plt.close()


## Loop over scenarios and networks, and make a scatter plot for each combination
_scatter_levels = ["scenario", "network"]
def plot_performance_scatter_multi(results: pd.DataFrame, *,
                                   saveto: Path | str=c.supplementary_path/"scatterplot.pdf", **kwargs) -> None:
    """
    Plot many DataFrames with y, y_hat, with total uncertainty (top) or aleatoric fraction (bottom) as colour.
    To do: simplify by looping over pre-defined networks and scenarios, rather than finding them in the DataFrame.
    """
    saveto = Path(saveto)

    # Loop over results and plot each network/scenario combination in a separate figure
    for (scenariokey, networkkey), df in results.groupby(level=_scatter_levels):
        # Set up labels
        saveto_here = saveto.with_stem(f"{saveto.stem}_{networkkey}-{scenariokey}")
        scenario, network = c.scenarios_123_fromkey[scenariokey], c.networks_fromkey[networkkey]

        # Plot
        plot_performance_scatter(df, title=f"{network.label} {scenario.label}", saveto=saveto_here, **kwargs)


### ACCURACY METRICS
## Combined plot
_lollipop_metrics = [c.mdsa, c.sspb, c.log_r_squared]
def plot_accuracy_metrics(data: pd.DataFrame, *,
                             groups: Iterable[c.Parameter]=c.iops, groupmembers: Iterable[c.Parameter]=c.networks, metrics: Iterable[c.Parameter]=_lollipop_metrics, scenarios: Iterable[c.Parameter]=c.scenarios_123,
                             saveto: Path | str=c.output_path/"accuracy_metrics.pdf") -> None:
    """
    Plot some number of DataFrames containing performance metric statistics.
    """
    # Constants
    bar_width = 0.15

    # Generate figure ; rows are metrics, columns are split types
    n_groups = len(groups)
    n_members = len(groupmembers)
    n_rows = len(metrics)
    n_cols = len(scenarios)
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 7), sharex=True, sharey="row", squeeze=False)

    # Plot results; must be done in a loop because there is no Pandas lollipop function
    for ax_row, metric in zip(axs, metrics):
        for ax, scenario in zip(ax_row, scenarios):
            for member_idx, member in enumerate(groupmembers):
                # Select data
                df = data.loc[scenario, member, groups]
                values = df[metric]

                locations = np.arange(n_groups) - (bar_width * (n_members - 1) / 2) + member_idx * bar_width

                ax.scatter(locations, values, color=member.color, label=member.label, s=50, zorder=3, edgecolor="black")  # Draw points
                ax.vlines(locations, 0, values, colors="grey", lw=1, alpha=0.7)

            # Panel settings
            ax.grid(False, axis="x")
            ax.axhline(0, color="black", linewidth=1, zorder=1)

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
    for ax, scenario in zip(axs[0], scenarios):
        ax.set_title(scenario.label)

    # Plot legend outside the subplots
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), framealpha=1, edgecolor="black")

    plt.tight_layout()
    plt.savefig(saveto, bbox_inches="tight")
    plt.close()


# # Figure 5 - boxplot

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# data_path = r"C:\SwitchDrive\Data\pnn_model_estimates"

# model_file_mapping = {
#     'BNN MCD': 'bnn_mcd',
#     'BNN DC': 'bnn_dc',
#     'MDN': 'mdn',
#     'ENS NN': 'ens_nn',
#     'RNN': 'rnn'
# }

# models = ['BNN MCD', 'BNN DC', 'MDN', 'ENS NN', 'RNN']
# scenarios = ['random', 'wd', 'ood']
# metrics = ['MdSA', 'SSPB', 'log_r_squared']

# colors = {'BNN MCD': '#6699CC', 'BNN DC': '#997700', 'MDN': '#994455', 'ENS NN': '#EE99AA', 'RNN': '#EECC66'}

# def read_csv_file(model, scenario):
#     filename = f"{model_file_mapping[model]}_{scenario}_split_metrics.csv"
#     filepath = os.path.join(data_path, filename)
#     df = pd.read_csv(filepath)
#     df['model'] = model
#     df['scenario'] = scenario
#     return df

# # read all CSV files and combine into a single DataFrame
# all_data = pd.concat([read_csv_file(model, scenario)
#                       for model in models
#                       for scenario in scenarios])

# # prep for plotting
# plot_data = all_data.melt(id_vars=['model', 'scenario', 'variable'],
#                           value_vars=metrics,
#                           var_name='metric', value_name='value')

# # Set up the plot
# fig, axes = plt.subplots(3, 3, figsize=(16, 10))
# plt.subplots_adjust(wspace=0.05, hspace=0.2)

# # format x-axis labels
# def format_xlabel(label):
#     if label.startswith('a'):
#         parts = label.split('_')
#         return f"$a_{{{parts[0][1:]}}}({parts[1]})$"
#     return label

# # create the boxplots
# for i, metric in enumerate(metrics):
#     for j, scenario in enumerate(scenarios):
#         ax = axes[i, j]
#         data = plot_data[(plot_data['metric'] == metric) & (plot_data['scenario'] == scenario)]

#         sns.boxplot(x='variable', y='value', hue='model', data=data, ax=ax, palette=colors,
#                     whis=[0, 100], width=0.6, hue_order=models)

#         # title only for the first row
#         if i == 0:
#             title = {"random": "Random split", "wd": "Within-distribution split", "ood": "Out-of-distribution split"}
#             ax.set_title(title[scenario], fontsize=12)

#         # y-label only for the first column
#         if j == 0:
#             if metric == 'MdSA':
#                 ax.set_ylabel('MdSA [%]', fontsize=10, fontweight='bold')
#             elif metric == 'SSPB':
#                 ax.set_ylabel('SSPB [%]', fontsize=10, fontweight='bold')
#             elif metric == 'log_r_squared':
#                 ax.set_ylabel('$R^2$', fontsize=10, fontweight='bold')
#         else:
#             ax.set_ylabel('')
#             ax.set_yticklabels([])

#         #  x-axis labels
#         x_labels = [format_xlabel(label.get_text()) for label in ax.get_xticklabels()]
#         ax.set_xticks(range(len(x_labels)))

#         # x-labels only for the last row, show ticks for all
#         if i == 2:
#             ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
#         else:
#             ax.set_xticklabels([])
#         ax.set_xlabel('')

#         # legend
#         if i == 1 and j == 2:  # only the middle-right plot
#             handles, labels = ax.get_legend_handles_labels()
#             ax.legend(handles, labels, title='Model', bbox_to_anchor=(1.05, 0.7),
#                       loc='upper left', fontsize=8)
#         else:
#             ax.get_legend().remove()

#         # y-axis limits based on metric
#         if metric == 'MdSA':
#             ax.set_ylim(0, 300)
#         elif metric == 'SSPB':
#             ax.set_ylim(-150, 100)
#         elif metric == 'log_r_squared':
#             ax.set_ylim(-1.5, 1)

#         ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# plt.tight_layout()
# plt.savefig('C:/SwitchDrive/Data/Plots/model_performance_boxplots.pdf', bbox_inches='tight')
# plt.show()
