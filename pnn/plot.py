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

from . import metrics
from . import constants as c


### FUNCTIONS
## Performance (matchups) - scatter plot, per algorithm/scenario combination
# -> for appendix, just performance, not (fractional) uncertainties ?
scatterplot_metrics = {"total_unc_pct": "Total uncertainty [%]", "ale_frac": "Aleatoric fraction"}
def plot_performance_scatter_single(df: pd.DataFrame, *,
                                    columns: Iterable[c.Parameter]=c.iops, rows: Iterable[c.Parameter]=[c.total_unc_pct, c.ale_frac],
                                    title: Optional[str]=None,
                                    saveto: Path | str="scatterplot.png") -> None:
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
        # Plot data per panel
        for ax, iop in zip(ax_row, columns):
            im = ax.scatter(df.loc["y_true", iop], df.loc["y_pred", iop], c=df.loc[uncertainty, iop],
                            alpha=0.7, cmap=uncertainty.cmap, vmin=uncertainty.vmin, vmax=uncertainty.vmax)

        # Color bar per row
        cb = fig.colorbar(im, ax=ax_row[-1], label=uncertainty.label)
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
    for ax, iop in zip(axs[0], columns):
        # Calculate
        y, y_hat = df.loc["y_true", iop], df.loc["y_pred", iop]
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
                             saveto: Path | str=c.supplementary_path/"scatterplot.png", **kwargs) -> None:
    """
    Plot many DataFrames with y, y_hat, with total uncertainty (top) or aleatoric fraction (bottom) as colour.
    """
    saveto = Path(saveto)

    # Loop over results and plot each network/split combination in a separate figure
    for (splitkey, networkkey), df in results.groupby(level=_scatter_levels):
        # Set up labels
        saveto_here = saveto.with_stem(f"{saveto.stem}_{networkkey}-{splitkey}")
        split, network = c.splits_fromkey[splitkey], c.networks_fromkey[networkkey]

        # Plot
        plot_performance_scatter_single(df, title=f"{network.label} {split.label}", saveto=saveto_here, **kwargs)


## Performance metrics - lollipop plot
_lollipop_metrics = [c.mdsa, c.sspb, c.r_squared]
def plot_performance_metrics_lollipop(data: pd.DataFrame, *,
                                      groups: Iterable[c.Parameter]=c.iops_main, groupmembers: Iterable[c.Parameter]=c.networks, metrics: Iterable[c.Parameter]=_lollipop_metrics, splits: Iterable[c.Parameter]=c.splits,
                                      saveto: Path | str=c.save_path/"performance_lolliplot_vertical.png") -> None:
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
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(14, 8), sharex=True, sharey="row", squeeze=False)

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
    axs[0, 0].set_xticklabels([p.label for p in groups])

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
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))

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
                               saveto: Path | str=c.supplementary_path/"uncertainty_line.png") -> None:
    """
    Plot log-binned statistics from a main DataFrame.
    """
    # Generate figure
    fig, axs = plt.subplots(nrows=len(c.splits)*len(c.networks), ncols=len(c.iops), sharex=True, figsize=(15, 25), layout="constrained", squeeze=False)

    # Plot lines
    for ax_row, (network, split) in zip(axs, itertools.product(c.networks, c.splits)):
        for ax, var in zip(ax_row, c.iops):
            df = binned.loc[split, network][var]
            plot_log_binned_statistics_line(df, ax=ax, legend=False)

    # Settings
    axs[0, 0].set_xscale("log")
    for ax in axs.ravel():
        ax.set_ylim(ymin=0)
    for ax, var in zip(axs[-1], c.iops):
        ax.set_xlabel(var.label)
    for ax, (network, split) in zip(axs[:, 0], itertools.product(c.networks, c.splits)):
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
def uncertainty_heatmap(results_agg: pd.DataFrame, *,
                        variables: Iterable[c.Parameter]=c.iops_main,
                        saveto: Path | str=c.save_path/"uncertainty_heatmap.png") -> None:
    """
    Plot a heatmap showing the average uncertainty and aleatoric fraction for each combination of network, IOP, and splitting method.
    """
    # Generate figure
    fig, axs = plt.subplots(nrows=2, ncols=len(c.splits), sharex=True, sharey=True, figsize=(11, 9), gridspec_kw={"wspace": -1, "hspace": 0}, layout="constrained", squeeze=False)

    # Plot data
    for ax_row, unc in zip(axs, _heatmap_metrics):
        # Plot each panel per row
        for ax, split in zip(ax_row, c.splits):
            # Select relevant data
            df = results_agg.loc[unc, split]
            df = df[variables]

            # Plot image
            im = ax.imshow(df, cmap=unc.cmap, vmin=unc.vmin, vmax=unc.vmax)

            # Plot individual values
            for i, x in enumerate(c.networks):
                for j, y in enumerate(variables):
                    ax.text(j, i, f"{df.iloc[i, j]:.2f}", ha="center", va="center", color="w")

        # Color bar per row
        cb = fig.colorbar(im, ax=ax_row, fraction=0.1, pad=0.01, shrink=1)
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

    for ax, split in zip(axs[0], c.splits):
        ax.set_title(split.label, fontweight="bold")

    plt.savefig(saveto)
    plt.close()


## Uncertainty metrics - bar plot
_bar_metrics = [c.sharpness, c.coverage]
def plot_uncertainty_metrics_bar(data: pd.DataFrame, *,
                                 groups: Iterable[c.Parameter]=c.iops_main, groupmembers: Iterable[c.Parameter]=c.networks, metrics: Iterable[c.Parameter]=_bar_metrics, splits: Iterable[c.Parameter]=c.splits,
                                 saveto: Path | str=c.save_path/"uncertainty_metrics_bar.png") -> None:
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
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(14, 8), sharex=True, sharey="row", squeeze=False)

    # Plot results
    for ax_row, metric in zip(axs, metrics):
        for ax, split in zip(ax_row, splits):
            for member_idx, member in enumerate(groupmembers):
                # Select data
                df = data.loc[split, member, metric]
                values = df[groups]

                color = member.color
                label = member.label

                locations = np.arange(n_groups) - (bar_width * (n_members - 1) / 2) + member_idx * bar_width

                ax.bar(locations, values, color=color, label=label, width=bar_width, zorder=3)  # Draw points

            ax.grid(True, axis="y", linestyle="--", linewidth=0.5, color="black", alpha=0.4)

    # Label variables
    axs[0, 0].set_xticks(np.arange(n_groups))
    axs[0, 0].set_xticklabels([p.label for p in groups])

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
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(saveto, dpi=200, bbox_inches="tight")
    plt.close()


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
