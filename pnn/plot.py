"""
Various plotting functions.
"""
import itertools
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.style.use("default")
from matplotlib import ticker
from cmcrameri.cm import managua

from .constants import iops, iops_main, network_types, split_types, uncertainty_types, save_path, supplementary_path
from . import io, metrics


### CONSTANTS
cmap_uniform = plt.cm.cividis.resampled(10)
cmap_aleatoric_fraction = managua.resampled(10)

model_colors = {
    "mdn": "#FF5733",
    "bnn_dropconnect": "#3357FF",
    "bnn_mcd": "#33FF57",
    "ensemble": "#F933FF",
    "rnn": "#FFC733",}

uncertainty_colors = {"ale_unc_pct": cmap_aleatoric_fraction.colors[-3],
                      "epi_unc_pct": cmap_aleatoric_fraction.colors[2],
                      "total_unc_pct": "black",}


### FUNCTIONS
## Performance (matchups) - scatter plot, per algorithm/scenario combination
# -> for appendix, just performance, not (fractional) uncertainties ?
scatterplot_metrics = {"total_unc_pct": "Total uncertainty [%]", "ale_frac": "Aleatoric fraction"}
def plot_performance_scatter_single(df: pd.DataFrame, *,
                                    columns: dict[str, str]=iops, rows: dict[str, str]=scatterplot_metrics,
                                    title: Optional[str]=None,
                                    saveto: Path | str="scatterplot.png") -> None:
    """
    Plot one DataFrame with y, y_hat, with total uncertainty (top) or aleatoric fraction (bottom) as colour.
    """
    # Constants
    rowkwargs = {"total_unc_pct": dict(vmin=0, vmax=20, cmap=cmap_uniform),
                 "ale_frac": dict(vmin=0, vmax=1, cmap=cmap_aleatoric_fraction),}
    lims = (1e-4, 1e1)
    scale = "log"

    # Create figure
    fig, axs = plt.subplots(nrows=len(rows), ncols=len(columns), sharex=True, sharey=True, figsize=(20, 10), squeeze=False, layout="constrained")

    # Plot data per row
    for ax_row, (ckey, clabel), kwargs in zip(axs, rows.items(), rowkwargs.values()):
        # Plot data per panel
        for ax, variable in zip(ax_row, columns):
            im = ax.scatter(df.loc["y_true", variable], df.loc["y_pred", variable], c=df.loc[ckey, variable], alpha=0.7, **kwargs)

        # Color bar per row
        cb = fig.colorbar(im, ax=ax_row[-1], label=clabel)
        cb.locator = ticker.MaxNLocator(nbins=6)
        cb.update_ticks()

    # Matchup plot settings
    for ax in axs.ravel():
        # ax.set_aspect("equal")
        ax.axline((0, 0), slope=1, color="black")
        ax.grid(True, color="black", alpha=0.5, linestyle="--")

    # Regression
    # slope, intercept, r_value, p_value, std_err = linregress(np.log(x_values), np.log(y_values))

    # # Set x_reg to span the entire x-axis range of the plot
    # # x_reg = np.linspace(*ax.get_xlim(), 500)
    # # y_reg = np.exp(intercept + slope * np.log(x_reg))
    # limits = [min(ax.get_xlim()[0], ax.get_ylim()[0]), 10, 10]
    # ax.plot(limits, limits, ls='--', color='black')

    # if row in [0,1,2,3,4,5]:
    #     x_reg = np.logspace(-3, 1, 500)  # Generates 500 points between 10^-3 and 10^1
    #     y_reg = np.exp(intercept + slope * np.log(x_reg))

    #     ax.plot(x_reg, y_reg, color='grey', label=f'R²={r_value**2:.2f}')
    #     ax.legend(loc='upper left')

    # Metrics
    for ax, variable in zip(axs[0], columns):
        # Calculate
        y, y_hat = df.loc["y_true", variable], df.loc["y_pred", variable]
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
    for ax, label in zip(axs[0], columns.values()):
        ax.set_title(label)
    fig.supxlabel("In-situ (actual)", fontsize="x-large", fontweight="bold")
    fig.supylabel("Model estimate", x=-0.02, fontsize="x-large", fontweight="bold")
    fig.suptitle(title, fontsize="x-large")

    # Save result
    plt.savefig(saveto, dpi=200, bbox_inches="tight")
    plt.close()

_scatter_levels = ["split", "network"]
def plot_performance_scatter(results: pd.DataFrame, *,
                             saveto: Path | str=supplementary_path/"scatterplot.png", **kwargs) -> None:
    """
    Plot many DataFrames with y, y_hat, with total uncertainty (top) or aleatoric fraction (bottom) as colour.
    """
    saveto = Path(saveto)

    # Loop over results and plot each network/split combination in a separate figure
    for (split, network), df in results.groupby(level=_scatter_levels):
        # Set up labels
        saveto_here = saveto.with_stem(f"{saveto.stem}_{network}-{split}")
        splitlabel, networklabel = split_types[split], network_types[network]

        # Plot
        plot_performance_scatter_single(df, title=f"{networklabel} {splitlabel}", saveto=saveto_here, **kwargs)


## Performance metrics - lollipop plot
_lollipop_metrics = metrics_display = {"mdsa": "MDSA [%]", "sspb": "Bias [%]", "r_squared": r"$R^2$"}
def plot_performance_metrics_lollipop(metrics_results: pd.DataFrame, *,
                                      groups: dict[str, str]=iops_main, metrics_to_plot: dict[str, str]=_lollipop_metrics, models_to_plot: dict[str, str]=network_types, splits: dict[str, str]=split_types,
                                      saveto: Path | str=save_path/"performance_lolliplot_vertical.png") -> None:
    """
    Plot some number of DataFrames containing performance metric statistics.
    """
    # Constants
    bar_width = 0.15

    # Generate figure ; rows are metrics, columns are split types
    n_groups = len(groups)
    n_metrics = len(metrics_to_plot)
    n_models = len(models_to_plot)
    n_splits = len(splits)
    fig, axs = plt.subplots(nrows=n_metrics, ncols=n_splits, figsize=(14, 8), sharex=True, sharey="row", squeeze=False)

    # Plot results; must be done in a loop because there is no Pandas lollipop function
    for ax_row, metric_label in zip(axs, metrics_to_plot):
        for ax, split_label in zip(ax_row, splits):
            for model_idx, network_type in enumerate(models_to_plot):
                # Select data
                df = metrics_results.loc[split_label, network_type, metric_label]
                values = df[groups.keys()]

                color = model_colors.get(network_type, "gray")
                label = models_to_plot.get(network_type, "model")

                locations = np.arange(n_groups) - (bar_width * (n_models - 1) / 2) + model_idx * bar_width

                ax.scatter(locations, values, color=color, label=label, s=50, zorder=3)  # Draw points
                ax.vlines(locations, 0, values, colors='grey', lw=1, alpha=0.7)

            ax.grid(True, axis="y", linestyle="--", linewidth=0.5, color="black", alpha=0.4)

    # Label variables
    axs[0, 0].set_xticks(np.arange(n_groups))
    axs[0, 0].set_xticklabels(groups.values())

    # Label y-axes
    for ax, ylabel in zip(axs[:, 0], metrics_to_plot.values()):
        ax.set_ylabel(ylabel, fontsize=12)
    fig.align_ylabels()

    # y-axis limits; currently hardcoded
    axs[0, 0].set_ylim(ymin=0)
    axs[2, 0].set_ylim(ymax=1)

    maxbias = np.abs(axs[1, 0].get_ylim()).max()
    axs[1, 0].set_ylim(-maxbias, maxbias)

    # Titles
    for ax, title in zip(axs[0], splits.values()):
        ax.set_title(title)

    # Plot legend outside the subplots
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(saveto, dpi=200, bbox_inches="tight")
    plt.close()


## Uncertainty statistics - line plot
def plot_log_binned_statistics_line(binned: pd.DataFrame, ax: plt.Axes, *,
                                    uncertainty_keys: Iterable[str]=uncertainty_types.keys(), **kwargs) -> None:
    """
    Given a DataFrame containing log-binned statistics, plot the total/aleatoric/epistemic uncertainties for one variable.
    Plots a line for the mean uncertainty and a shaded area for the standard deviation.
    If no ax is provided, a new figure is created.
    """
    # Loop over uncertainty types and plot each
    for unc, label in uncertainty_types.items():
        df = binned.loc[unc]
        color = uncertainty_colors[unc]

        df.plot.line(ax=ax, y="mean", color=color, label=label, **kwargs)
        ax.fill_between(df.index, df["mean"] - df["std"], df["mean"] + df["std"], color=color, alpha=0.1)

    # Labels
    ax.grid(True, ls="--")

def plot_log_binned_statistics(binned: pd.DataFrame, *,
                               saveto: Path | str=supplementary_path/"uncertainty_line.png") -> None:
    """
    Plot log-binned statistics from a main DataFrame.
    """
    # Generate figure
    fig, axs = plt.subplots(nrows=len(split_types)*len(network_types), ncols=len(iops), sharex=True, figsize=(15, 25), layout="constrained", squeeze=False)

    # Plot lines
    for ax_row, (network, split) in zip(axs, itertools.product(network_types, split_types)):
        for ax, var in zip(ax_row, iops):
            df = binned.loc[split, network][var]
            plot_log_binned_statistics_line(df, ax=ax, legend=False)

    # Settings
    axs[0, 0].set_xscale("log")
    for ax in axs.ravel():
        ax.set_ylim(ymin=0)
    for ax, var in zip(axs[-1], iops.values()):
        ax.set_xlabel(var)
    for ax, (network, split) in zip(axs[:, 0], itertools.product(network_types.values(), split_types.values())):
        ax.set_ylabel(f"{network}\n{split}")

    # Labels
    fig.suptitle("")
    fig.supxlabel("In situ value", fontweight="bold")
    fig.supylabel("Mean uncertainty [%]", fontweight="bold")
    fig.align_ylabels()

    plt.savefig(saveto)
    plt.close()


## Uncertainty statistics - heatmap
heatmap_metrics = {"total_unc_pct": "Total uncertainty [%]", "ale_frac": "Aleatoric fraction"}
def uncertainty_heatmap(results_agg: pd.DataFrame, *,
                        variables=iops_main,
                        saveto: Path | str=save_path/"uncertainty_heatmap.png") -> None:
    """
    Plot a heatmap showing the average uncertainty and aleatoric fraction for each combination of network, IOP, and splitting method.
    """
    # Constants
    rowkwargs = {"total_unc_pct": dict(vmin=0, vmax=20, cmap=cmap_uniform),
                 "ale_frac": dict(vmin=0, vmax=1, cmap=cmap_aleatoric_fraction),}

    # Generate figure
    fig, axs = plt.subplots(nrows=2, ncols=len(split_types), sharex=True, sharey=True, figsize=(11, 9), gridspec_kw={"wspace": -1, "hspace": 0}, layout="constrained", squeeze=False)

    # Plot data
    for ax_row, (unc, unc_label), kwargs in zip(axs, heatmap_metrics.items(), rowkwargs.values()):
        # Plot each panel per row
        for ax, split in zip(ax_row, split_types):
            # Select relevant data
            df = results_agg.loc[unc, split]
            df = df[variables.keys()]

            # Plot image
            im = ax.imshow(df, **kwargs)

            # Plot individual values
            for i, x in enumerate(network_types):
                for j, y in enumerate(variables):
                    ax.text(j, i, f"{df.iloc[i, j]:.2f}", ha="center", va="center", color="w")

        # Color bar per row
        cb = fig.colorbar(im, ax=ax_row, fraction=0.1, pad=0.01, shrink=1)
        cb.set_label(label=unc_label, weight="bold")
        cb.locator = ticker.MaxNLocator(nbins=6)
        cb.update_ticks()

    # Labels
    fig.supxlabel("IOPs", fontweight="bold")
    fig.supylabel("Models", fontweight="bold")

    wrap_labels = lambda labels: list(zip(*enumerate(labels)))  # (0, 1, ...) (label0, label1, ...)
    for ax in axs[-1]:
        ax.set_xticks(*wrap_labels(variables.values()), rotation=45, ha="right")

    for ax in axs[:, 0]:
        ax.set_yticks(*wrap_labels(network_types.values()))

    for ax, label in zip(axs[0], split_types.values()):
        ax.set_title(label, fontweight="bold")

    plt.savefig(saveto)
    plt.close()


# ### sharpness and coverage factor plots
# # -> turn into normal barplots?

# import matplotlib.pyplot as plt
# import numpy as np

# # Variables and their display titles
# #variables = ['aCDOM_443', 'aCDOM_675', 'aNAP_443', 'aNAP_675', 'aph_443', 'aph_675']
# #display_titles = ['a$_{CDOM}$ 443', 'a$_{CDOM}$ 675', 'a$_{NAP}$ 443', 'a$_{NAP}$ 675', 'a$_{ph}$ 443', 'a$_{ph}$ 675']

# variables = ['aCDOM_443', 'aCDOM_675', 'aph_443', 'aph_675']
# display_titles = ['a$_{CDOM}$ 443', 'a$_{CDOM}$ 675', 'a$_{ph}$ 443', 'a$_{ph}$ 675']

# models_and_colors = {
#     'mdn': 'blue',
#     'dc': 'green',
#     'mcd': 'red',
#     'ens': 'purple',
#     'rnn': 'black'
# }
# new_model_labels = ['MDN', 'BNN DC', 'BNN MCD', 'ENS NN', 'RNN']

# fig, axs = plt.subplots(1, 4, figsize=(15, 4))

# bar_width = 0.35
# index = np.arange(len(models_and_colors))

# for i, var in enumerate(variables):
#     ax = axs[i]
#     ax.set_title(display_titles[i])

#     wd_scores = []
#     ood_scores = []
#     for model_key in models_and_colors.keys():
#         wd_label = f'{model_key}_wd_sharpness'
#         ood_label = f'{model_key}_ood_sharpness'

#         wd_score = sharpness_results_dfs[wd_label].loc[0, var] if wd_label in sharpness_results_dfs else 0
#         ood_score = sharpness_results_dfs[ood_label].loc[0, var] if ood_label in sharpness_results_dfs else 0
#         wd_scores.append(wd_score)
#         ood_scores.append(ood_score)

#     wd_bars = ax.bar(index, wd_scores, bar_width, label='Within-distribution', color='#469990')
#     ood_bars = ax.bar(index + bar_width, ood_scores, bar_width, label='Out-of-distribution', color='#f4bf75')

#     ax.set_xticks(index + bar_width / 2)
#     ax.set_xticklabels(new_model_labels, rotation=45)
#     ax.set_ylim(0, 0.3)
#     ax.grid(True, axis='y', ls='--',c='black', alpha=0.5)

# fig.supylabel('Sharpness',x=-0.005,y=0.55, fontweight='bold',fontsize=14)
# fig.supxlabel('Models', fontweight='bold',fontsize=14)

# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 0.6))

# plt.tight_layout()

# save_path = '/content/drive/My Drive/iop_ml/plots/'
# plt.savefig(save_path + 'sharpness_results_wd_ood.png',dpi=200,bbox_inches='tight')

# plt.show()

# ### Coverage factor matrix

# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# from matplotlib.colors import Normalize
# from matplotlib.cm import ScalarMappable

# new_model_labels = ['MDN', 'BNN DC', 'BNN MCD', 'ENS NN', 'RNN']

# # Separate the coverage factors results into 'wd' and 'ood' - but should also have random split for completeness?
# wd_coverage_factors = {k: v for k, v in coverage_factors_results.items() if 'wd_coverage_factors' in k}
# ood_coverage_factors = {k: v for k, v in coverage_factors_results.items() if 'ood_coverage_factors' in k}

# def prepare_data_for_heatmap(coverage_dict):
#     data = pd.DataFrame()
#     for key, df in coverage_dict.items():
#         model_name = key.split('_')[0]
#         df_transposed = df.T
#         df_transposed.columns = [model_name]
#         data = pd.concat([data, df_transposed], axis=1)
#     data = data.T
#     return data

# wd_data = prepare_data_for_heatmap(wd_coverage_factors)
# ood_data = prepare_data_for_heatmap(ood_coverage_factors)

# # Exclude 'aNAP_443' and 'aNAP_675' from both wd_data and ood_data
# wd_data = wd_data[['aCDOM_443', 'aCDOM_675', 'aph_443', 'aph_675']]
# ood_data = ood_data[['aCDOM_443', 'aCDOM_675', 'aph_443', 'aph_675']]

# display_titles = ['a$_{CDOM}$ 443', 'a$_{CDOM}$ 675', 'a$_{ph}$ 443', 'a$_{ph}$ 675']
# wd_data.columns = display_titles
# ood_data.columns = display_titles

# # Normalize the color range for both heatmaps - 0 - 100% coverage
# norm = Normalize(vmin=0, vmax=100)

# fig, axs = plt.subplots(1, 2, figsize=(11, 5))
# cmap = plt.get_cmap('viridis')

# def plot_heatmap(data, ax, title, model_labels=None, norm=None):
#     highest_in_columns = data.max()
#     im = ax.imshow(data, cmap=cm.batlow, norm=norm)
#     ax.set_xticks(np.arange(data.shape[1]))
#     ax.set_yticks(np.arange(data.shape[0]))
#     ax.set_xticklabels(data.columns)
#     ax.set_yticklabels(model_labels)
#     ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

#     # Loop over data dimensions and create text annotations.
#     for i in range(data.shape[0]):
#         for j in range(data.shape[1]):
#             if data.iloc[i, j] == highest_in_columns[j]:
#                 text_color = 'red'  # Red for the highest number in the column
#             else:
#                 text_color = 'black'
#             ax.text(j, i, format(data.iloc[i, j], '.1f'),
#                     ha="center", va="center", color=text_color, fontsize=10)

#     ax.set_title(title)

# plot_heatmap(wd_data, axs[0], 'Within-distribution', model_labels=new_model_labels, norm=norm)
# plot_heatmap(ood_data, axs[1], 'Out-of-distribution', model_labels=new_model_labels, norm=norm)

# # Create an axis for the colorbar
# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.8, 0.25, 0.02, 0.5])

# # Create a colorbar in the created axis
# sm = ScalarMappable(cmap=cm.batlow, norm=norm)
# sm.set_array([])
# cbar = fig.colorbar(sm, cax=cbar_ax, ticks=np.arange(0, 101, 20))
# cbar.set_label('Coverage factor [%]', fontsize=12)

# fig.supxlabel('IOPs', x=0.45,y=-0.09, fontweight='bold',fontsize=14)
# fig.supylabel('Models',x=0.04, fontweight='bold',fontsize=14)

# save_path = '/content/drive/My Drive/iop_ml/plots/'
# plt.savefig(save_path + 'coverage_factors_matrix_4_variables.png',dpi=200,bbox_inches='tight')

# plt.show()

# #####
# ## Calibration plots
# #####

# ### Need to add the random split results as a calibration row

# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.lines as mlines
# import pandas as pd

# # Define the new model labels for the legend
# new_model_labels = ['MDN', 'BNN DC', 'BNN MCD', 'ENS NN', 'RNN']

# # Assuming you have all the necessary data loaded and the function from uct is imported
# # pip install uncertainty-toolbox

# # Define colors for each model for distinction
# model_colors = {
#     'mdn': 'red',
#     'mcd': 'green',
#     'dc': 'blue',
#     'ens': 'orange',
#     'rnn': 'cyan'
# }

# column_names = ['aCDOM_443', 'aCDOM_675', 'aph_443', 'aph_675']
# display_titles = ['a$_{CDOM}$ 443', 'a$_{CDOM}$ 675', 'a$_{ph}$ 443', 'a$_{ph}$ 675']

# # Assuming the original plot_calibration function is imported from uct
# # Define a DataFrame to hold miscalibration areas for 'wd' and 'ood'
# miscalibration_areas_wd = pd.DataFrame(columns=column_names, index=new_model_labels)
# miscalibration_areas_ood = pd.DataFrame(columns=column_names, index=new_model_labels)

# # Define a function to calculate miscalibration area
# def calculate_miscalibration_area(exp_proportions, obs_proportions):
#     # Ensure exp_proportions and obs_proportions are numpy arrays
#     exp_proportions = np.array(exp_proportions)
#     obs_proportions = np.array(obs_proportions)
#     return np.abs(exp_proportions - obs_proportions).mean()

# # Custom plot_calibration function with removed fill_between and updated labels
# def custom_plot_calibration(y_true, y_std, y_pred, model_label, color, ax=None):
#     # Call the original plot_calibration function with swapped y_pred and y_true
#     ax = uct.viz.plot_calibration(y_pred, y_std, y_true, ax=ax)

#     # Set the color and label of the last line (our calibration curve)
#     ax.get_lines()[-1].set_color(color)
#     ax.get_lines()[-1].set_label(model_label)

#     # Remove the miscalibration area text box
#     for text in ax.texts:
#         if "Miscalibration area" in text.get_text():
#             text.remove()

#     for line in ax.lines:
#         if line.get_label() == 'Ideal':
#             line.remove()
#     for coll in ax.collections:
#         coll.remove()

#     ax.plot([0, 1], [0, 1], 'k--', label='Ideal')

#     ax.set_xlabel("Observed Proportion in Interval")
#     ax.set_ylabel("Predicted Proportion in Interval")

#     return ax

# # Create the figure with the specified size
# fig = plt.figure(figsize=(14, 5))

# # Adjust your plotting function to the new layout and axis labels
# def plot_calibration_curves(fig, results):
#     # Creating 2 rows for 'WD' and 'OOD', and 6 columns for the variables
#     for i, column in enumerate(column_names):
#         for j, data_key in enumerate(['_wd', '_ood']):
#             ax = fig.add_subplot(2, 6, i + j * 6 + 1)
#             for model_idx, model_name in enumerate(['mdn', 'dc', 'mcd', 'ens', 'rnn']):
#                 model_key = f"{model_name}{data_key}"
#                 if model_key in results:
#                     model_results = results[model_key]
#                     y_pred = model_results['y_pred'][column].values
#                     y_std = model_results['std_pred'][column].values
#                     y_true = model_results['y_true'][column].values
#                     custom_plot_calibration(y_pred, y_std, y_true, new_model_labels[model_idx], model_colors[model_name], ax=ax)

#             ax.set_title(display_titles[i])
#             ax.set_xlabel("")
#             ax.set_ylabel("")

# plot_calibration_curves(fig, results)

# # Add a global legend outside of the subplots
# handles = [mlines.Line2D([], [], color=color, label=label) for label, color in zip(new_model_labels, model_colors.values())]
# fig.legend(handles=handles, loc='lower center', ncol=len(new_model_labels), bbox_to_anchor=(0.35, -0.06))

# fig.supxlabel("Estimated proportion in interval", x=0.35,fontsize=12)
# fig.supylabel("Actual proportion in interval",x=0.01, fontsize=12)

# fig.text(0.35, 1.01, 'Within-distribution', ha='center', va='center', fontsize=14, fontweight='bold', transform=fig.transFigure)
# fig.text(0.35, 0.52, 'Out-of-distribution', ha='center', va='center', fontsize=14, fontweight='bold', transform=fig.transFigure)

# plt.tight_layout(h_pad=5)

# save_path = '/content/drive/My Drive/iop_ml/plots/'
# plt.savefig(save_path + 'calibration_curves_presentation.png',dpi=200,bbox_inches='tight')
# # Show the plot
# plt.show()

# #####
# # plot to exemplify sharpness, coverage factor, calibration confidence in the methods section - not a result plot

# #### this could use actual data from a model! has to be the case for the paper if we use these plots

# def make_plots(pred_mean, pred_std, y, scenario_title, plot_save_str="row"):
#     savefig=False
#     plt.rcParams['text.usetex'] = False
#     plt.rcParams['font.family'] = 'sans-serif'

#     """Make set of plots."""
#     ylims = [-3, 3]
#     n_subset = 50

#     fig, axs = plt.subplots(1, 2, figsize=(14, 8))

#     # Make ordered intervals plot
#     uct.plot_intervals_ordered(
#         pred_mean, pred_std, y, n_subset=n_subset, ylims=ylims, ax=axs[0]
#     )

#     # Make calibration plot
#     uct.plot_calibration(pred_mean, pred_std, y, ax=axs[1])
#     axs[1].plot([0, 1], [0, 1], color='black', linestyle='--')  # Calibration line

#     axs[0].set_title('Ordered estimation intervals', fontsize=14)
#     axs[0].set_xlabel('Index (ordered by observed value)', fontsize=12)
#     axs[0].set_ylabel('Estimated values and intervals', fontsize=12)

#     axs[1].set_title('Average calibration', fontsize=14)
#     axs[1].set_xlabel('Estimated proportion in interval', fontsize=14)
#     axs[1].set_ylabel('Observed proportion in interval', fontsize=14)

#     # for text_obj in axs[1].texts:
#     #     if "Miscalibration area" in text_obj.get_text():
#     #         text_obj.set_fontsize('large')
#     #         break

#     fig.suptitle(scenario_title, y=0.85, fontsize=16, fontweight='bold')
#     fig.subplots_adjust(wspace=0.25, top=0.85)

#     if savefig:
#         save_path = '/content/drive/My Drive/iop_ml/plots_scenarios/'
#         os.makedirs(save_path, exist_ok=True)
#         full_save_path = os.path.join(save_path, f"{plot_save_str}_save.png")
#         plt.savefig(full_save_path, dpi=200, bbox_inches='tight')
#         print(f"Plot saved to {full_save_path}")

#     plt.show()


# np.random.seed(11)

# # Generate synthetic predictive uncertainty results
# n_obs = 650
# f, std, y, x = uct.synthetic_sine_heteroscedastic(n_obs)

# pred_mean_list = [f]

# pred_std_list = [
#     std * 0.5,  # overconfident
#     std * 2.0,  # underconfident
#     std,  # correct
# ]

# scenario_titles = ['Overconfident', 'Underconfident', 'Well-calibrated']

# idx_counter = 0
# for i, pred_mean in enumerate(pred_mean_list):
#     for j, pred_std in enumerate(pred_std_list):
#         scenario_title = scenario_titles[j]  # Get the scenario title
#         mace = uct.mean_absolute_calibration_error(pred_mean, pred_std, y)
#         rmsce = uct.root_mean_squared_calibration_error(pred_mean, pred_std, y)

#         idx_counter += 1
#         print(f"Scenario {idx_counter}: MACE: {mace:.4f}, RMSCE: {rmsce:.4f}, MA: {ma:.4f}")
#         make_plots(pred_mean, pred_std, y, scenario_title, f"row_{idx_counter}")
#         print('saved')


# #### not a result plot: but the scenario distributions
# ### -> render the same as the IOP distribution plots of aph443, anap443, acdom443
# ## -> see scenario_datasets folder for the .csv files

# random_train = pd.read_csv('robust_train_random.csv').rename(columns={'org_aNAP_443':'org_anap_443', 'org_aph_443':'org_aph_443', 'org_aCDOM_443':'org_acdom_443'})
# random_test = pd.read_csv('robust_test_random.csv').rename(columns={'org_aNAP_443':'org_anap_443', 'org_aph_443':'org_aph_443', 'org_aCDOM_443':'org_acdom_443'})

# train_set_wd = pd.read_csv('wd_train_set.csv')
# test_set_wd = pd.read_csv('wd_test_set.csv')

# train_set_oos = pd.read_csv('ood_train_set.csv')
# test_set_oos = pd.read_csv('ood_test_set.csv')

# clipped_variables = ['org_aph_443', 'org_anap_443', 'org_acdom_443']

# random_train_clipped = random_train[clipped_variables].clip(lower=0.00001)
# random_test_clipped = random_test[clipped_variables].clip(lower=0.00001)

# # Update the original dataframes with the clipped values
# random_train.update(random_train_clipped)
# random_test.update(random_test_clipped)

# # Remove rows where any of the clipped variables have values <= 0.000001
# random_train_r = random_train[~(random_train[clipped_variables] <= 0.000001).any(axis=1)]
# random_test_r = random_test[~(random_test[clipped_variables] <= 0.000001).any(axis=1)]

# import seaborn as sns
# import matplotlib.pyplot as plt

# columns = ['org_aph_443', 'org_anap_443', 'org_acdom_443']

# def plot_pdf_curves_only(train_set_random, test_set_random, train_set, test_set, train_set_oos, test_set_oos, columns, figsize=(14, 8)):
#     fig, axes = plt.subplots(3, len(columns), figsize=figsize, constrained_layout=True)

#     x_labels = [r'a$_{ph}$443 [m$^{-1}$]', r'a$_{nap}$443 [m$^{-1}$]', r'a$_{cdom}$443 [m$^{-1}$]']

#     custom_titles = {
#         (0, 1): 'Random split',
#         (1, 1): 'Within-distribution split',
#         (2, 1): 'Out-of-distribution split'
#     }

#     datasets = [
#         (train_set_random, test_set_random, "Random split datasets"),
#         (train_set, test_set, "Within-distribution (WD) datasets"),
#         (train_set_oos, test_set_oos, "Out-of-distribution (OOS) datasets")
#     ]

#     x_limits = [(-1, 6), (-1, 6), (-1, 6)]
#     y_limits = [(0, 4), (0, 5), (0, 2.5)]

#     for row, (train_data, test_data, title) in enumerate(datasets):
#         for idx, col in enumerate(columns):
#             sns.kdeplot(train_data[col], shade=True, bw_adjust=0.5, ax=axes[row, idx], label="Train", color="black",alpha=0.7)
#             sns.kdeplot(test_data[col], shade=True, bw_adjust=0.5, ax=axes[row, idx], label="Test", color="orange",alpha=0.7)
#             #xes[row, idx].set_xscale('log')
#             axes[row, idx].set_xlim(*x_limits[idx])
#             axes[row, idx].set_ylim(*y_limits[idx])
#             axes[row,idx].set_ylabel("")
#             axes[row, idx].set_xlabel(x_labels[idx] if row == 2 else '', fontsize=14)
#             if (row, idx) in custom_titles:
#                 axes[row, idx].set_title(custom_titles[(row, idx)],fontweight='bold')
#             else:
#                 axes[row, idx].set_title('')
#             axes[row, idx].grid(True, which='both', linestyle='--', alpha=0.6)

#     fig.supylabel('Density', x=-0.03,y=0.55, fontsize=12, fontweight='bold')
#     fig.supxlabel('IOP', fontsize=12, fontweight='bold')

#     handles, labels = axes[0, 0].get_legend_handles_labels()
#     fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.05, 0.57), ncol=1, frameon=True, fontsize=14)

#     plt.savefig(r'C:\SwitchDrive\Data\Update_4\plots\pdf_all_data_split.png',dpi=200,bbox_inches='tight')
#     plt.show()

# # Usage example, assuming the same datasets and columns are defined
# plot_pdf_curves_only(random_train_r, random_test_r, train_set_wd, test_set_wd, train_set_oos, test_set_oos, columns)
