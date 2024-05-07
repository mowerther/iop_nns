import functools
import itertools
from pathlib import Path
from typing import Iterable, Optional
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter

from pnn import metrics

### CONSTANTS
plt.style.use("default")
pred_path = Path("pnn_model_estimates/")
save_path = Path("manuscript_figures/")

variables = ("aCDOM_443", "aCDOM_675", "aNAP_443", "aNAP_675", "aph_443", "aph_675")
uncertainty_labels = ("Total", "Aleatoric", "Epistemic")
colors = ("black", "blue", "orange")

# Define model colors for distinction
model_colors = {
    'mdn': '#FF5733',
    'mcd': '#33FF57',
    'dc': '#3357FF',
    'ens': '#F933FF',
    'rnn': '#FFC733',
    }

### LOADING / PROCESSING DATA
def filter_df(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """
    Reorganise a single dataframe to have Instance as its index, only contain data columns, and (optionally) have a suffix added to all columns.
    """
    df_filtered = df.loc[df["Category"] == category]
    return df_filtered.set_index("Instance") \
                      .drop(columns=["Category"])


def reorganise_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorganise an input dataframe so that "Category" and "Instance" become a multi-index.
    """
    return df.set_index(["Category", "Instance"])


def calculate_percentage_uncertainty(df: pd.DataFrame, *,
                                     reference_key: str="pred_scaled_for_unc", uncertainty_keys: Iterable[str]=("total_unc", "ale_unc", "epi_unc")) -> pd.DataFrame:
    """
    Calculates the percentage uncertainty (total, aleatoric, epistemic) relative to the scaled prediction.

    Parameters:
    - df: the main dataframe containing all data.
    Optional:
    - reference_key: the index for the denominator (default: "pred_scaled_for_unc")
    - uncertainty_keys: the indices for the numerators (default: "total_unc, "ale_unc", "epi_unc")
    """
    # Define helper functions
    to_percentage = lambda data, key: np.abs(df.loc[key] / df.loc[reference_key]) * 100
    update_key = lambda key: key + "_pct"

    # Perform the operation on the specified keys
    result = {update_key(key): to_percentage(df, key) for key in uncertainty_keys}
    result = pd.concat(result)

    return result


def read_data(filename: Path | str) -> pd.DataFrame:
    """
    Read data from a dataframe and process it.
    """
    df = pd.read_csv(filename)
    df = reorganise_df(df)

    df_percent = calculate_percentage_uncertainty(df)
    df = pd.concat([df, df_percent])

    return df


### RUN SCRIPT
### LOAD DATA
network_types = ["mdn", "bnn_dropconnect", "bnn_mcd", "ensemble", "rnn"]
split_types = ["wd", "ood", "random_split"]
results = {f"{network}_{split}": read_data(pred_path/f"{network}_{split}_preds.csv") for network, split in itertools.product(network_types, split_types)}
print("Read results into `results` dictionary")
print(results.keys())

### LINE PLOT
# Calculate log-binned statistics
def log_binned_statistics(x: pd.Series, y: pd.Series, *,
                          vmin: float=1e-4, vmax: float=40, binwidth: float=0.2, n: int=100) -> pd.DataFrame:
    """
    Calculate statistics (mean, std) for y as a function of x, in log-space bins.
    binwidth and n can be chosen separately to allow for overlapping bins.
    Example:
        x = in situ data
        y = total uncertainty in prediction
        vmin = 0.1 ; vmax = 10 ; binwidth = 1 ; n = 2
        Calculates mean and std uncertainty in prediction at (0.1 < x < 1), (1 < x < 10)
    """
    # Setup
    x_log = np.log10(x)
    bins_log = np.linspace(np.log10(vmin), np.log10(vmax), n)
    bins = np.power(10, bins_log)

    # Find the indices per bin
    slices = [(x_log >= centre - binwidth) & (x_log <= centre + binwidth) for centre in bins_log]
    binned = [y.loc[s] for s in slices]

    # Calculate statistics
    binned_means = [b.mean() for b in binned]
    binned_stds = [b.std() for b in binned]

    # Wrap into dataframe
    binned = pd.DataFrame(index=bins, data={"mean": binned_means, "std": binned_stds})
    return binned


def log_binned_statistics_all_variables(reference: pd.DataFrame, data: pd.DataFrame, *,
                                        columns: Iterable[str]=variables) -> pd.DataFrame:
    """
    Calculate log binned statistics for each of the variables in one dataframe.
    """
    # Perform calculations
    binned = {key: log_binned_statistics(reference.loc[:,key], data.loc[:,key]) for key in columns}

    # Add suffices to columns and merge
    binned = [df.add_prefix(f"{key}_") for key, df in binned.items()]
    binned = binned[0].join(binned[1:])

    return binned


def log_binned_statistics_combined(df: pd.DataFrame, *,
                                   reference_key: str="y_true", uncertainty_keys: Iterable[str]=("total_unc_pct", "ale_unc_pct", "epi_unc_pct")) -> pd.DataFrame:
    """
    Calculate log binned statistics for each of the uncertainty dataframes relative to x, and combine them into a single dataframe.
    """
    # Bin individual uncertainty keys
    binned = {key: log_binned_statistics_all_variables(df.loc[reference_key], df.loc[key]) for key in uncertainty_keys}

    # Combine into one multi-index DataFrame
    binned = pd.concat(binned)

    return binned


def plot_log_binned_statistics(binned: pd.DataFrame, variable: str, *,
                               ax: Optional[plt.Axes]=None,
                               uncertainty_keys: Iterable[str]=("total_unc_pct", "ale_unc_pct", "epi_unc_pct")) -> None:
    """
    Given a DataFrame containing log-binned statistics, plot the total/aleatoric/epistemic uncertainties for one variable.
    Plots a line for the mean uncertainty and a shaded area for the standard deviation.
    If no ax is provided, a new figure is created.
    """
    # Set up keys
    mean, std = f"{variable}_mean", f"{variable}_std"

    # Set up a new figure if desired
    NEW_FIGURE = (ax is None)
    if NEW_FIGURE:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure

    # Loop over uncertainty types and plot each
    for unc, label, color in zip(uncertainty_keys, uncertainty_labels, colors):
        df = binned.loc[unc]
        df.plot.line(ax=ax, y=mean, color=color, label=label)
        ax.fill_between(df.index, df[mean] - df[std], df[mean] + df[std], color=color, alpha=0.1)

    # Labels
    ax.set_xlabel(variable)
    ax.grid(True, ls="--")


# mdn_wd, mdn_ood, dc_wd, dc_ood, mcd_wd, mcd_ood, ens_wd, ens_ood, rnn_wd, rnn_ood
binned = log_binned_statistics_combined(results["mdn_wd"])

# Plot
fig, axs = plt.subplots(nrows=1, ncols=len(variables), sharex=True, figsize=(15, 5), layout="constrained")

for ax, var in zip(axs, variables):
    plot_log_binned_statistics(binned, var, ax=ax)

# Settings
axs[0].set_xscale("log")
for ax in axs:
    ax.set_ylim(ymin=0)

fig.suptitle("")
fig.supxlabel("In situ value", fontweight="bold")
fig.supylabel("Mean uncertainty [%]", fontweight="bold")

plt.savefig(save_path/"uncertainty_line.png")
plt.show()
plt.close()


### LOLLIPOP PLOT
metrics_results = {key: metrics.calculate_metrics(df) for key, df in results.items()}
# The metrics_results dictionary now contains dfs with calculated metrics for each algorithm and variable

# Separating the results for the scenarios
random_metrics_results = {k: v for k, v in metrics_results.items() if '_random' in k}
wd_metrics_results = {k: v for k, v in metrics_results.items() if '_wd' in k}
ood_metrics_results = {k: v for k, v in metrics_results.items() if '_ood' in k}

# Custom y-axis labels and their limits
y_axis_labels = list(metrics.metrics_display.keys())
y_axis_limits = [(0, 140), (-140, 50), (0, 1)]

# Model and display titles for the variables
new_model_labels = ['MDN', 'BNN DC', 'BNN MCD', 'ENS NN', 'RNN']
display_titles = ['a$_{CDOM}$ 443', 'a$_{CDOM}$ 675', 'a$_{ph}$ 443', 'a$_{ph}$ 675']

def plot_vertical_lollipop_charts(metrics_results_list, titles):
    n_groups = len(display_titles)
    n_metrics = len(metrics.metrics_display)
    n_models = len(metrics_results_list[0])
    bar_width = 0.15
    opacity = 0.8

    fig, axs = plt.subplots(n_metrics, 3, figsize=(14, 8), sharex='col')
    #fig.suptitle("Performance Metrics", y=0.99, fontsize=16, fontweight='bold')

    for i, (metric_column, metric_display) in enumerate(metrics.metrics_display.items()):
        for j, metrics_results in enumerate(metrics_results_list):
            ax = axs[i, j]
            ax.set_ylim(y_axis_limits[i])

            for model_idx, (model_key, df) in enumerate(metrics_results.items()):
                model_short = model_key.split('_')[0]
                color = model_colors.get(model_short, 'gray')

                # Drop aNAP
                df = df.drop(columns=[key for key in df.columns if "NAP" in key])

                x = df.loc[metric_column]
                y = np.arange(n_groups) - (bar_width * (n_models - 1) / 2) + model_idx * bar_width
                ax.scatter(y, x, color=color, label=new_model_labels[model_idx], s=50, zorder=3)  # Draw points
                ax.vlines(y, 0, x, colors='grey', lw=1, alpha=0.7)

            if j == 0:
                ax.set_ylabel(metric_display,fontweight='bold',fontsize=12)
            if i == 0:
                ax.set_title(titles[j])

            ax.set_xticks(np.arange(n_groups))
            ax.set_xticklabels(display_titles)
            ax.grid(True, which='both', linestyle='--', linewidth='0.5', color='black', alpha=0.4, axis='y')

    # Plot legend outside the subplots
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(save_path/'performance_lolliplot_vertical.png', dpi=200, bbox_inches='tight')
    plt.show()

# Example usage with the revised function and hypothetical data structure
plot_vertical_lollipop_charts([random_metrics_results, wd_metrics_results, ood_metrics_results], ['Random split', 'Within-distribution split', 'Out-of-distribution split'])
