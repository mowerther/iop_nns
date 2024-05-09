"""
Various plotting functions
"""
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.style.use("default")

from .constants import iops, iops_main, network_types, split_types, uncertainty_colors, uncertainty_types, save_path
from .metrics import metrics_display


### CONSTANTS
model_colors = {
    "mdn": "#FF5733",
    "bnn_dropconnect": "#3357FF",
    "bnn_mcd": "#33FF57",
    "ensemble": "#F933FF",
    "rnn": "#FFC733",}


### FUNCTIONS
def plot_log_binned_statistics_line(binned: pd.DataFrame, variable: str, ax: plt.Axes, *,
                                    uncertainty_keys: Iterable[str]=uncertainty_types.keys(), **kwargs) -> None:
    """
    Given a DataFrame containing log-binned statistics, plot the total/aleatoric/epistemic uncertainties for one variable.
    Plots a line for the mean uncertainty and a shaded area for the standard deviation.
    If no ax is provided, a new figure is created.
    """
    # Set up keys
    mean, std = f"{variable}_mean", f"{variable}_std"

    # Loop over uncertainty types and plot each
    for unc, label in uncertainty_types.items():
        df = binned.loc[unc]
        color = uncertainty_colors[unc]

        df.plot.line(ax=ax, y=mean, color=color, label=label, **kwargs)
        ax.fill_between(df.index, df[mean] - df[std], df[mean] + df[std], color=color, alpha=0.1)

    # Labels
    ax.set_xlabel(variable)
    ax.grid(True, ls="--")


def plot_log_binned_statistics(binned: Iterable[pd.DataFrame], *,
                               saveto: Path | str=save_path/"uncertainty_line.png") -> None:
    """
    Plot some number of DataFrames containing log-binned statistics.
    """
    # If only one DataFrame is provided, wrap it in a list
    if isinstance(binned, pd.DataFrame):
        binned = [binned]

    # Generate figure
    fig, axs = plt.subplots(nrows=len(binned), ncols=len(iops), sharex=True, figsize=(15, 25), layout="constrained", squeeze=False)

    # Plot lines
    for ax_row, (label, df) in zip(axs, binned.items()):
        for ax, var in zip(ax_row, iops):
            plot_log_binned_statistics_line(df, var, ax=ax, legend=False)

        ax_row[0].set_ylabel(label)

    # Settings
    axs[0, 0].set_xscale("log")
    for ax in axs.ravel():
        ax.set_ylim(ymin=0)

    fig.suptitle("")
    fig.supxlabel("In situ value", fontweight="bold")
    fig.supylabel("Mean uncertainty [%]", fontweight="bold")
    fig.align_ylabels()

    plt.savefig(saveto)
    plt.close()

## Lollipop plot performance

def plot_performance_metrics_lollipop(metrics_results: dict[str, pd.DataFrame], *,
                                      groups: dict[str, str]=iops_main, metrics_to_plot: dict[str, str]=metrics_display, models_to_plot: dict[str, str]=network_types, splits: dict[str, str]=split_types,
                                      saveto: Path | str=save_path/"performance_lolliplot_vertical.png") -> None:
    """
    Plot some number of DataFrames containing performance metric statistics.
    """
    # Constants
    bar_width = 0.15

    # Separating the results for the scenarios
    metrics_results_split = {label: {key: val for key, val in metrics_results.items() if f"_{label}" in key} for label in splits}

    # Generate figure ; rows are metrics, columns are split types
    n_groups = len(groups)
    n_metrics = len(metrics_to_plot)
    n_models = len(models_to_plot)
    n_splits = len(splits)
    fig, axs = plt.subplots(nrows=n_metrics, ncols=n_splits, figsize=(14, 8), sharex=True, sharey="row", squeeze=False)

    # Plot results; must be done in a loop because there is no Pandas lollipop function
    for ax_row, metric_label in zip(axs, metrics_to_plot):
        for ax, (split_label, metrics_split) in zip(ax_row, metrics_results_split.items()):
            for model_idx, (network_type, df) in enumerate(zip(models_to_plot, metrics_split.values())):
                color = model_colors.get(network_type, "gray")
                label = models_to_plot.get(network_type, "model")

                values = df.loc[metric_label][groups.keys()]
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
    axs[2, 0].set_ylim(0, 1)

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

### 1. Scatterplot for a single Algo + scenario
# -> for appendix, just performance, not (fractional) uncertainties

def calculate_percentage_from_category(df, category1, category2, columns_of_interest):
    """
    Filters two categories from the main dataframe, resets their indexes,
    selects the specific columns, and calculates the percentage of category1 over category2 for those columns.

    Parameters:
    - df: the main dataframe containing all data.
    - category1: the category for the numerator.
    - category2: the category for the denominator.
    - columns_of_interest: the IOPs to perform the operation on

    Returns:
    - A dataframe with the calculated percentages for the specified columns.
    """
    # Filter dataframes by category and reset indexes
    df_cat_reset_1 = df[df['Category'] == category1].reset_index(drop=True)
    df_cat_reset_2 = df[df['Category'] == category2].reset_index(drop=True)

    # Perform the operation on the specified columns
    result = np.abs(df_cat_reset_1[columns_of_interest] / df_cat_reset_2[columns_of_interest]) * 100

    return result

# Define the columns of interest
columns_of_interest = ['aCDOM_443', 'aCDOM_675', 'aNAP_443', 'aNAP_675', 'aph_443', 'aph_675']

# Use the updated function to calculate percentages directly from the main dataframe
percent_total_uncertainty = calculate_percentage_from_category(mcd_ood, 'total_unc', 'pred_scaled_for_unc', columns_of_interest)
percent_aleatoric_uncertainty = calculate_percentage_from_category(mcd_ood, 'ale_unc', 'pred_scaled_for_unc', columns_of_interest)
percent_epistemic_uncertainty = calculate_percentage_from_category(mcd_ood, 'epi_unc', 'pred_scaled_for_unc', columns_of_interest)

# Other categories
pred_scaled = mcd_ood[mcd_ood['Category'] == 'pred_scaled_for_unc']
y_true = mcd_ood[mcd_ood['Category'] == 'y_true']
y_pred = mcd_ood[mcd_ood['Category'] == 'y_pred']
std_pred = mcd_ood[mcd_ood['Category'] == 'pred_std'].reset_index(drop=True)

fraction_aleatoric_unc = percent_aleatoric_uncertainty / percent_total_uncertainty

column_names = ['aCDOM_443', 'aCDOM_675', 'aNAP_443', 'aNAP_675', 'aph_443', 'aph_675']
display_titles = ['a$_{CDOM}$ 443', 'a$_{CDOM}$ 675', 'a$_{NAP}$ 443', 'a$_{NAP}$ 675', 'a$_{ph}$ 443', 'a$_{ph}$ 675']
uncertainty_labels = ['Total uncertainty', 'Fraction of Aleatoric']

norm_total = plt.Normalize(vmin=0, vmax=20)
norm_aleatoric = plt.Normalize(vmin=0, vmax=1)


fig, axs = plt.subplots(2, 6, figsize=(20, 10), constrained_layout=True)
axs = axs.ravel()


### scatterplot starts here
def plot_uncertainty(ax, y_true, y_pred, uncertainties, actual_title, display_title, show_title, row, norm, cmap='viridis'):
    y_true_reset = y_true.reset_index(drop=True)
    y_pred_reset = y_pred.reset_index(drop=True)

    # Apply the mask for values greater than 10^-4 for each variable
    # Not needed I think?
    mask = (y_true_reset[actual_title] > 1e-4) & (y_pred_reset[actual_title] > 1e-4)

    # Apply the mask
    x_values = y_true_reset.loc[mask, actual_title]
    y_values = y_pred_reset.loc[mask, actual_title]
    color_values = uncertainties.loc[mask]

    if show_title:
      ax.set_title(display_title, fontsize=16)

    # Scatter plot
    sc = ax.scatter(x_values, y_values, c=color_values, cmap=cmap, norm=norm, alpha=1)

    # Calculate regression
    slope, intercept, r_value, p_value, std_err = linregress(np.log(x_values), np.log(y_values))

    # Set x_reg to span the entire x-axis range of the plot
    # x_reg = np.linspace(*ax.get_xlim(), 500)
    # y_reg = np.exp(intercept + slope * np.log(x_reg))
    limits = [min(ax.get_xlim()[0], ax.get_ylim()[0]), 10, 10]
    ax.plot(limits, limits, ls='--', color='black')

    if row in [0,1,2,3,4,5]:
        x_reg = np.logspace(-3, 1, 500)  # Generates 500 points between 10^-3 and 10^1
        y_reg = np.exp(intercept + slope * np.log(x_reg))

        ax.plot(x_reg, y_reg, color='grey', label=f'R²={r_value**2:.2f}')
        ax.legend(loc='upper left')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-3, 10)
    ax.set_ylim(1e-3, 10)
    ax.grid(True, ls='--', alpha=0.5)

    if row in [0,1,2,3,4,5]:
      tr = x_values
      pred = y_values

      sspb_b = sspb(tr, pred)
      mdsa_b = mdsa(tr, pred)
      mape_b = mape(tr, pred)

      textstr_b = '\n'.join((
      r'$\mathrm{Bias = %.2f}$' % (sspb_b, ) + '%',
      r'$\mathrm{MdSA = %.2f}$' % (mdsa_b, )  + '%',
      r'$\mathrm{MdAPE = %.2f}$' % (mape_b, )  + '%',
          ))

      x_set_g = 0.18
      y_set_g= 0.0011

      ax.text(x_set_g,y_set_g, textstr_b, color='black', size=9,
          bbox=dict(facecolor='none', edgecolor="none"))

    return sc

scalars = []  # To keep track of scatter objects for colorbars
for i, uncertainty_label in enumerate(['Total uncertainty', 'Fraction of Aleatoric']):
    uncertainties = percent_total_uncertainty if i == 0 else fraction_aleatoric_unc
    norm = norm_total if i == 0 else norm_aleatoric
    # requires import cmcrameri
    cmap = cm.batlowK if i == 0 else cm.managua

    for j, (actual_title, display_title) in enumerate(zip(column_names, display_titles)):
        index = i * len(column_names) + j
        sc = plot_uncertainty(axs.ravel()[index], y_true, y_pred, uncertainties[actual_title], actual_title, display_title, i == 0, index, norm, cmap)
        if i == 1 or (i == 0 and j == 0):  # Keep first scalar of each row for colorbars
            scalars.append(sc)

fig.supxlabel('In-situ (actual)', fontsize='x-large', fontweight='bold')
fig.supylabel('Model estimate', x=-0.02, fontsize='x-large', fontweight='bold')

# Colorbar for Total Uncertainty (top row)
cbar_ax1 = fig.add_axes([1.01, 0.575, 0.02, 0.35])  # Adjust position for the top row
cbar1 = fig.colorbar(scalars[0], cax=cbar_ax1, norm=norm_total)
cbar1.set_label('Total uncertainty [%]', fontsize=12,fontweight='bold')
cbar_ticks = cbar1.get_ticks()
if cbar_ticks[-1] == 20:  # If the last tick is exactly at the max value (20)
    cbar_ticklabels = [f"{tick:.0f}" for tick in cbar_ticks[:-1]] + ["$\geq$ 20"]
    cbar1.set_ticks(cbar_ticks)  
    cbar1.set_ticklabels(cbar_ticklabels) 

# Colorbar for Fraction of Aleatoric Uncertainty (second row)
cbar_ax2 = fig.add_axes([1.01, 0.102, 0.02, 0.35])  # Adjust position for the bottom row
cbar2 = fig.colorbar(scalars[1], cax=cbar_ax2, norm=norm_aleatoric, cmap='coolwarm')
cbar2.set_label('Aleatoric fraction', fontsize=12, fontweight='bold')

#cbar.set_label('Uncertainty [%]')
#save_path = '/content/drive/My Drive/iop_ml/plots/'
#plt.savefig(save_path + 'mcd_wd_unc.png',dpi=200,bbox_inches='tight')
plt.show()


########
### heatmap of the mean uncertainty and the aleatoric fraction
########

import pandas as pd
import numpy as np

# the data is organised differently now?
dataframes = {
    'mdn_wd': mdn_wd,
    'mdn_ood': mdn_ood,
    'mdn_random':mdn_random,
    'dc_wd': dc_wd,
    'dc_ood': dc_ood,
    'dc_random':dc_random,
    'mcd_wd': mcd_wd,
    'mcd_ood': mcd_ood,
    'mcd_random': mcd_random,
    'ens_wd': ens_wd,
    'ens_ood': ens_ood,
    'ens_random':ens_random,
    'rnn_wd' : rnn_wd,
    'rnn_ood' : rnn_ood,
    'rnn_random':rnn_random,
}

columns_of_interest = ['aCDOM_443', 'aCDOM_675', 'aph_443', 'aph_675']

def calculate_uncertainties_and_categories(df, columns_of_interest):
    result_dict = {
        'percent_total_uncertainty': calculate_percentage_from_category(df, 'total_unc', 'pred_scaled_for_unc', columns_of_interest),
        'percent_aleatoric_uncertainty': calculate_percentage_from_category(df, 'ale_unc', 'pred_scaled_for_unc', columns_of_interest),
        'percent_epistemic_uncertainty': calculate_percentage_from_category(df, 'epi_unc', 'pred_scaled_for_unc', columns_of_interest),
        'pred_scaled': df[df['Category'] == 'pred_scaled_for_unc'],
        'y_true': df[df['Category'] == 'y_true'],
        'y_pred': df[df['Category'] == 'y_pred'],
        'std_pred': df[df['Category'] == 'pred_std'].reset_index(drop=True)
    }
    return result_dict

# Apply the function to each DataFrame and store the results in a new dictionary
results = {df_name: calculate_uncertainties_and_categories(df, columns_of_interest) for df_name, df in dataframes.items()}


def filter_uncertainties_with_count(df_total, df_aleatoric, df_epistemic):
    # Identify indices where any of the uncertainties are < 0 or > 200 in any of the three dataframes
    mask = (df_total < 0) | (df_total > 1000) | (df_aleatoric < 0) | (df_aleatoric > 1000) | (df_epistemic < 0) | (df_epistemic > 1000)
    filtered_indices = mask.any(axis=1)

    # Count the number of observations to remove
    removed_count = filtered_indices.sum()

    # Filter out the rows from all dataframes
    df_total_filtered = df_total[~filtered_indices]
    df_aleatoric_filtered = df_aleatoric[~filtered_indices]
    df_epistemic_filtered = df_epistemic[~filtered_indices]

    return df_total_filtered, df_aleatoric_filtered, df_epistemic_filtered, removed_count

# Initialize a counter
total_removed = 0

for key, result in results.items():
    result['percent_total_uncertainty'], result['percent_aleatoric_uncertainty'], result['percent_epistemic_uncertainty'], removed_count = filter_uncertainties_with_count(
        result['percent_total_uncertainty'],
        result['percent_aleatoric_uncertainty'],
        result['percent_epistemic_uncertainty']
    )
    total_removed += removed_count

print(f"Total observations removed: {total_removed}")

def calculate_average_uncertainties(results, columns_of_interest):
    averages = {}
    for model, data in results.items():
        model_averages = {}
        for uncertainty_type in ['percent_total_uncertainty', 'percent_aleatoric_uncertainty', 'percent_epistemic_uncertainty']:
            df = data[uncertainty_type]
            avg_uncertainties = df[columns_of_interest].mean().to_dict()
            model_averages[uncertainty_type] = avg_uncertainties
        averages[model] = model_averages
    return averages

columns_of_interest = ['aCDOM_443', 'aCDOM_675', 'aph_443', 'aph_675']  # subset
average_uncertainties = calculate_average_uncertainties(results, columns_of_interest)

from matplotlib.colors import Normalize

def prepare_heatmap_data(average_uncertainties, uncertainty_type):
    data_random = pd.DataFrame(columns=columns_of_interest)
    data_wd = pd.DataFrame(columns=columns_of_interest)
    data_ood = pd.DataFrame(columns=columns_of_interest)

    for model, data in average_uncertainties.items():
        if 'random' in model:
          data_random.loc[model.replace('_random', '')] = pd.Series(data[uncertainty_type])
        elif 'wd' in model:
            data_wd.loc[model.replace('_wd', '')] = pd.Series(data[uncertainty_type])
        elif 'ood' in model:
            data_ood.loc[model.replace('_ood', '')] = pd.Series(data[uncertainty_type])

    return data_random, data_wd, data_ood

data_total_random, data_total_wd, data_total_ood = prepare_heatmap_data(average_uncertainties, 'percent_total_uncertainty')
data_aleatoric_random, data_aleatoric_wd, data_aleatoric_ood = prepare_heatmap_data(average_uncertainties, 'percent_aleatoric_uncertainty')
data_epistemic_random, data_epistemic_wd, data_epistemic_ood = prepare_heatmap_data(average_uncertainties, 'percent_epistemic_uncertainty')

fraction_aleatoric_wd = data_aleatoric_wd / data_total_wd
fraction_aleatoric_ood = data_aleatoric_ood / data_total_ood
fraction_aleatoric_random = data_aleatoric_random / data_total_random

data_uncertainty = [data_total_random, data_total_wd, data_total_ood]
cmap_uncertainty = plt.cm.cividis.resampled(10)
norm_uncertainty = Normalize(vmin=0, vmax=30)

data_fraction = [fraction_aleatoric_random, fraction_aleatoric_wd, fraction_aleatoric_ood]
cmap_fraction = plt.cm.BrBG.resampled(10)
norm_fraction = Normalize(vmin=0, vmax=1)


from matplotlib.cm import ScalarMappable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
fig, axs = plt.subplots(2, 3, figsize=(11, 9), sharex=False, sharey=True, gridspec_kw={"wspace": 0.1, "hspace": -0.15})
plt.style.use('default')

new_model_labels = ['MDN', 'BNN DC', 'BNN MCD', 'ENS NN', 'RNN']
variables = ['aCDOM_443', 'aCDOM_675', 'aph_443', 'aph_675']
display_titles = ['a$_{CDOM}$ 443', 'a$_{CDOM}$ 675', 'a$_{ph}$ 443', 'a$_{ph}$ 675']

viridis = plt.cm.viridis

new_colors = viridis(np.linspace(0, 1, 15))
cmap_uncertainty = ListedColormap(new_colors)

cmap_fraction = cm.managua.resampled(5)

for ax, data in zip(axs[0], data_uncertainty):
    ax.imshow(data, cmap=cmap_uncertainty, norm=norm_uncertainty)

for ax, data in zip(axs[1], data_fraction):
    ax.imshow(data, cmap=cmap_fraction, norm=norm_fraction)

for ax, data in zip(axs.ravel(), [*data_uncertainty, *data_fraction]):
    for i, x in enumerate(new_model_labels):
        for j, y in enumerate(variables):
            ax.text(j, i, f"{data.iloc[i, j]:.2f}", ha="center", va="center", color="w")

wrap_labels = lambda labels: list(zip(*enumerate(labels)))
for ax in axs[1]:
    ax.set_xticks(*wrap_labels(display_titles), rotation=45, ha="right")

for ax in axs[:, 0]:
    ax.set_yticks(*wrap_labels(new_model_labels))

# Colour bars - shrink factor is manual, but easier than using make_axes_locatable with multiple axes
cbar = fig.colorbar(ScalarMappable(norm=norm_uncertainty, cmap=cmap_uncertainty), ax=axs[0].tolist(), fraction=0.05, pad=0.03, shrink=0.7)
cbar.set_label("Mean uncertainty [%]", size=12, weight="bold")
cbar_ticks = cbar.get_ticks()
if cbar_ticks[-1] == 30:
    cbar_ticklabels = [f"{tick:.0f}" for tick in cbar_ticks[:-1]] + ["$\geq$ 30"]
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_ticklabels)

cbar = fig.colorbar(ScalarMappable(norm=norm_fraction, cmap=cmap_fraction), ax=axs[1].tolist(), fraction=0.05, pad=0.03, shrink=0.7)
cbar.set_label("Aleatoric fraction", size=12, weight="bold")
cbar.set_ticks(np.arange(0, 1.01, 0.2))

axs[0, 0].set_title("Random split", fontsize=12, fontweight="bold")
axs[0, 1].set_title("Within-distribution", fontsize=12, fontweight="bold")
axs[0, 2].set_title("Out-of-distribution", fontsize=12, fontweight="bold")

# align the x-limits of the bottom row subplots to match the top row
for ax_row in axs:
    for ax in ax_row[1:]:
        ax.set_xlim(axs[0, 0].get_xlim())

# turn off the x-axis labels and ticks for the first row only
for ax in axs[0]:
    ax.set_xticklabels([])
    ax.set_xticks([])

# x-axis labels and ticks for the second row
for ax in axs[1]:
    ax.set_xticklabels(display_titles, rotation=45, ha="right")
    ax.set_xticks(np.arange(len(display_titles)))

fig.supxlabel('IOPs', y=0.03, fontsize=12, fontweight='bold')
fig.supylabel('Models',x=0.001, fontsize=12, fontweight='bold')

save_path = '/content/drive/My Drive/iop_ml/plots/'
plt.savefig(save_path + 'avg_uncertainty_w_random_2.png', dpi=200, bbox_inches='tight')

plt.show()


### sharpness and coverage factor plots
# -> turn into normal barplots?

import numpy as np

def assert_is_flat_same_shape(*arrays):
    """Ensure all input arrays are 1D and of the same shape."""
    shape = None
    for arr in arrays:
        assert arr.ndim == 1, "All input arrays must be 1D."
        if shape is None:
            shape = arr.shape
        assert arr.shape == shape, "All input arrays must have the same shape."

def assert_is_positive(array):
    """Ensure all elements in the array are positive."""
    assert np.all(array > 0), "All elements in the array must be positive."

def sharpness(y_std: np.ndarray) -> float:
    """Return sharpness (a single measure of the overall confidence).

    Args:
        y_std: 1D array of the predicted standard deviations for the held out dataset.

    Returns:
        A single scalar which quantifies the average of the standard deviations.
    """
    assert_is_flat_same_shape(y_std)
    assert_is_positive(y_std)

    # Compute sharpness
    sharp_metric = np.sqrt(np.mean(y_std**2))

    return sharp_metric

import pandas as pd
import numpy as np

sharpness_results = {}

for col in std_pred.columns[std_pred.columns.get_loc('Instance')+1:]:
    sharpness_results[col] = sharpness(std_pred[col].values)

sharpness_results

sharpness_results_dfs = {}

for key, result in results.items():
    std_pred = result['std_pred']
    sharpness_scores = {}

    # Ensure there's an 'Instance' column to offset from; adjust as necessary
    if 'Instance' in std_pred.columns:
        start_col = std_pred.columns.get_loc('Instance') + 1
    else:
        # If 'Instance' column is not present, assume calculation starts from the first column
        start_col = 0

    # Iterate over the relevant columns to calculate sharpness
    for col in std_pred.columns[start_col:]:
        sharpness_scores[col] = sharpness(std_pred[col].values)

    # Convert sharpness scores to a DataFrame and store in the dictionary with an updated key
    sharpness_df = pd.DataFrame(sharpness_scores, index=[0])
    sharpness_results_dfs[f'{key}_sharpness'] = sharpness_df

# This dictionary now contains a DataFrame of sharpness results for each dataset, identified by keys like 'mdn_wd_sharpness'
# sharpness_results_dfs.keys()
# dict_keys(['mdn_wd_sharpness', 'mdn_ood_sharpness', 'dc_wd_sharpness', 'dc_ood_sharpness', 'mcd_wd_sharpness', 'mcd_ood_sharpness', 'ens_wd_sharpness', 'ens_ood_sharpness', 'rnn_wd_sharpness', 'rnn_ood_sharpness', 'dc_random_sharpness', 'ens_random_sharpness', 'mdn_random_sharpness', 'rnn_random_sharpness', 'mcd_random_sharpness'])

import matplotlib.pyplot as plt
import numpy as np

# Variables and their display titles
#variables = ['aCDOM_443', 'aCDOM_675', 'aNAP_443', 'aNAP_675', 'aph_443', 'aph_675']
#display_titles = ['a$_{CDOM}$ 443', 'a$_{CDOM}$ 675', 'a$_{NAP}$ 443', 'a$_{NAP}$ 675', 'a$_{ph}$ 443', 'a$_{ph}$ 675']

variables = ['aCDOM_443', 'aCDOM_675', 'aph_443', 'aph_675']
display_titles = ['a$_{CDOM}$ 443', 'a$_{CDOM}$ 675', 'a$_{ph}$ 443', 'a$_{ph}$ 675']

models_and_colors = {
    'mdn': 'blue',
    'dc': 'green',
    'mcd': 'red',
    'ens': 'purple',
    'rnn': 'black'
}
new_model_labels = ['MDN', 'BNN DC', 'BNN MCD', 'ENS NN', 'RNN']

fig, axs = plt.subplots(1, 4, figsize=(15, 4))

bar_width = 0.35
index = np.arange(len(models_and_colors))

for i, var in enumerate(variables):
    ax = axs[i]
    ax.set_title(display_titles[i])

    wd_scores = []
    ood_scores = []
    for model_key in models_and_colors.keys():
        wd_label = f'{model_key}_wd_sharpness'
        ood_label = f'{model_key}_ood_sharpness'

        wd_score = sharpness_results_dfs[wd_label].loc[0, var] if wd_label in sharpness_results_dfs else 0
        ood_score = sharpness_results_dfs[ood_label].loc[0, var] if ood_label in sharpness_results_dfs else 0
        wd_scores.append(wd_score)
        ood_scores.append(ood_score)

    wd_bars = ax.bar(index, wd_scores, bar_width, label='Within-distribution', color='#469990')
    ood_bars = ax.bar(index + bar_width, ood_scores, bar_width, label='Out-of-distribution', color='#f4bf75')

    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(new_model_labels, rotation=45)  
    ax.set_ylim(0, 0.3)  
    ax.grid(True, axis='y', ls='--',c='black', alpha=0.5)

fig.supylabel('Sharpness',x=-0.005,y=0.55, fontweight='bold',fontsize=14)  
fig.supxlabel('Models', fontweight='bold',fontsize=14)  

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 0.6))

plt.tight_layout() 

save_path = '/content/drive/My Drive/iop_ml/plots/'
plt.savefig(save_path + 'sharpness_results_wd_ood.png',dpi=200,bbox_inches='tight')

plt.show()

### Coverage factor

def calculate_coverage_factor(y_true, y_pred_mean, y_pred_std, k=1):
    """
    Calculate the coverage factor (P_unc) which is the percentage of true values
    that fall within the predicted interval [y_pred_mean - k * y_pred_std, y_pred_mean + k * y_pred_std].

    Args:
    - y_true: Array of true values, shape [n_data_points, n_variables].
    - y_pred_mean: Array of predicted means, shape [n_data_points, n_variables].
    - y_pred_std: Array of predicted standard deviations, shape [n_data_points, n_variables].
    - k: The factor by which to multiply the standard deviation to create the interval.

    Returns:
    - coverage_factors: Array of coverage factor values for each output variable.
    """
    assert y_true.shape == y_pred_mean.shape == y_pred_std.shape, "Shapes of y_true, y_pred_mean, and y_pred_std must match."

    lower_bounds = y_pred_mean - k * y_pred_std
    upper_bounds = y_pred_mean + k * y_pred_std
    within_bounds = (y_true >= lower_bounds) & (y_true <= upper_bounds)
    coverage_factors = np.mean(within_bounds, axis=0) * 100  
    return coverage_factors

coverage_factors_results = {}

for key, result in results.items():
    
    y_true_df = result['y_true']
    y_pred_df = result['y_pred']
    std_pred_df = result['std_pred']

    start_col_index = y_true_df.columns.get_loc('Instance') + 1
    coverage_factors = {}
    for col in y_true_df.columns[start_col_index:]:
        coverage_factors[col] = calculate_coverage_factor(
            y_true_df[col].values,
            y_pred_df[col].values,
            std_pred_df[col].values,
            k=1
        )

    coverage_factors_df = pd.DataFrame(coverage_factors, index=[0])
    coverage_factors_results[f'{key}_coverage_factors'] = coverage_factors_df

### Coverage factor matrix

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

new_model_labels = ['MDN', 'BNN DC', 'BNN MCD', 'ENS NN', 'RNN']

# Separate the coverage factors results into 'wd' and 'ood' - but should also have random split for completeness?
wd_coverage_factors = {k: v for k, v in coverage_factors_results.items() if 'wd_coverage_factors' in k}
ood_coverage_factors = {k: v for k, v in coverage_factors_results.items() if 'ood_coverage_factors' in k}

def prepare_data_for_heatmap(coverage_dict):
    data = pd.DataFrame()
    for key, df in coverage_dict.items():
        model_name = key.split('_')[0]
        df_transposed = df.T
        df_transposed.columns = [model_name]
        data = pd.concat([data, df_transposed], axis=1)
    data = data.T
    return data

wd_data = prepare_data_for_heatmap(wd_coverage_factors)
ood_data = prepare_data_for_heatmap(ood_coverage_factors)

# Exclude 'aNAP_443' and 'aNAP_675' from both wd_data and ood_data
wd_data = wd_data[['aCDOM_443', 'aCDOM_675', 'aph_443', 'aph_675']]
ood_data = ood_data[['aCDOM_443', 'aCDOM_675', 'aph_443', 'aph_675']]

display_titles = ['a$_{CDOM}$ 443', 'a$_{CDOM}$ 675', 'a$_{ph}$ 443', 'a$_{ph}$ 675']
wd_data.columns = display_titles
ood_data.columns = display_titles

# Normalize the color range for both heatmaps - 0 - 100% coverage
norm = Normalize(vmin=0, vmax=100) 

fig, axs = plt.subplots(1, 2, figsize=(11, 5))
cmap = plt.get_cmap('viridis')

def plot_heatmap(data, ax, title, model_labels=None, norm=None):
    highest_in_columns = data.max()
    im = ax.imshow(data, cmap=cm.batlow, norm=norm)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(data.columns)
    ax.set_yticklabels(model_labels)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data.iloc[i, j] == highest_in_columns[j]:
                text_color = 'red'  # Red for the highest number in the column
            else:
                text_color = 'black'
            ax.text(j, i, format(data.iloc[i, j], '.1f'),
                    ha="center", va="center", color=text_color, fontsize=10)

    ax.set_title(title)

plot_heatmap(wd_data, axs[0], 'Within-distribution', model_labels=new_model_labels, norm=norm)
plot_heatmap(ood_data, axs[1], 'Out-of-distribution', model_labels=new_model_labels, norm=norm)

# Create an axis for the colorbar
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.8, 0.25, 0.02, 0.5])

# Create a colorbar in the created axis
sm = ScalarMappable(cmap=cm.batlow, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, ticks=np.arange(0, 101, 20))
cbar.set_label('Coverage factor [%]', fontsize=12)

fig.supxlabel('IOPs', x=0.45,y=-0.09, fontweight='bold',fontsize=14)
fig.supylabel('Models',x=0.04, fontweight='bold',fontsize=14)

save_path = '/content/drive/My Drive/iop_ml/plots/'
plt.savefig(save_path + 'coverage_factors_matrix_4_variables.png',dpi=200,bbox_inches='tight')

plt.show()

#####
## Calibration plots
#####

### Need to add the random split results as a calibration row

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
import pandas as pd

# Define the new model labels for the legend
new_model_labels = ['MDN', 'BNN DC', 'BNN MCD', 'ENS NN', 'RNN']

# Assuming you have all the necessary data loaded and the function from uct is imported
# pip install uncertainty-toolbox

# Define colors for each model for distinction
model_colors = {
    'mdn': 'red',
    'mcd': 'green',
    'dc': 'blue',
    'ens': 'orange',
    'rnn': 'cyan'
}

column_names = ['aCDOM_443', 'aCDOM_675', 'aph_443', 'aph_675']
display_titles = ['a$_{CDOM}$ 443', 'a$_{CDOM}$ 675', 'a$_{ph}$ 443', 'a$_{ph}$ 675']

# Assuming the original plot_calibration function is imported from uct
# Define a DataFrame to hold miscalibration areas for 'wd' and 'ood'
miscalibration_areas_wd = pd.DataFrame(columns=column_names, index=new_model_labels)
miscalibration_areas_ood = pd.DataFrame(columns=column_names, index=new_model_labels)

# Define a function to calculate miscalibration area
def calculate_miscalibration_area(exp_proportions, obs_proportions):
    # Ensure exp_proportions and obs_proportions are numpy arrays
    exp_proportions = np.array(exp_proportions)
    obs_proportions = np.array(obs_proportions)
    return np.abs(exp_proportions - obs_proportions).mean()

# Custom plot_calibration function with removed fill_between and updated labels
def custom_plot_calibration(y_true, y_std, y_pred, model_label, color, ax=None):
    # Call the original plot_calibration function with swapped y_pred and y_true
    ax = uct.viz.plot_calibration(y_pred, y_std, y_true, ax=ax)

    # Set the color and label of the last line (our calibration curve)
    ax.get_lines()[-1].set_color(color)
    ax.get_lines()[-1].set_label(model_label)

    # Remove the miscalibration area text box
    for text in ax.texts:
        if "Miscalibration area" in text.get_text():
            text.remove()

    for line in ax.lines:
        if line.get_label() == 'Ideal':
            line.remove()
    for coll in ax.collections:
        coll.remove()

    ax.plot([0, 1], [0, 1], 'k--', label='Ideal')

    ax.set_xlabel("Observed Proportion in Interval")
    ax.set_ylabel("Predicted Proportion in Interval")

    return ax

# Create the figure with the specified size
fig = plt.figure(figsize=(14, 5))

# Adjust your plotting function to the new layout and axis labels
def plot_calibration_curves(fig, results):
    # Creating 2 rows for 'WD' and 'OOD', and 6 columns for the variables
    for i, column in enumerate(column_names):
        for j, data_key in enumerate(['_wd', '_ood']):
            ax = fig.add_subplot(2, 6, i + j * 6 + 1)
            for model_idx, model_name in enumerate(['mdn', 'dc', 'mcd', 'ens', 'rnn']):
                model_key = f"{model_name}{data_key}"
                if model_key in results:
                    model_results = results[model_key]
                    y_pred = model_results['y_pred'][column].values
                    y_std = model_results['std_pred'][column].values
                    y_true = model_results['y_true'][column].values
                    custom_plot_calibration(y_pred, y_std, y_true, new_model_labels[model_idx], model_colors[model_name], ax=ax)

            ax.set_title(display_titles[i])
            ax.set_xlabel("")
            ax.set_ylabel("")

plot_calibration_curves(fig, results)

# Add a global legend outside of the subplots
handles = [mlines.Line2D([], [], color=color, label=label) for label, color in zip(new_model_labels, model_colors.values())]
fig.legend(handles=handles, loc='lower center', ncol=len(new_model_labels), bbox_to_anchor=(0.35, -0.06))

fig.supxlabel("Estimated proportion in interval", x=0.35,fontsize=12)
fig.supylabel("Actual proportion in interval",x=0.01, fontsize=12)

fig.text(0.35, 1.01, 'Within-distribution', ha='center', va='center', fontsize=14, fontweight='bold', transform=fig.transFigure)
fig.text(0.35, 0.52, 'Out-of-distribution', ha='center', va='center', fontsize=14, fontweight='bold', transform=fig.transFigure)

plt.tight_layout(h_pad=5)

save_path = '/content/drive/My Drive/iop_ml/plots/'
plt.savefig(save_path + 'calibration_curves_presentation.png',dpi=200,bbox_inches='tight')
# Show the plot
plt.show()

#####
# plot to exemplify sharpness, coverage factor, calibration confidence in the methods section - not a result plot

#### this could use actual data from a model! has to be the case for the paper if we use these plots

def make_plots(pred_mean, pred_std, y, scenario_title, plot_save_str="row"):
    savefig=False
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'sans-serif'

    """Make set of plots."""
    ylims = [-3, 3]
    n_subset = 50

    fig, axs = plt.subplots(1, 2, figsize=(14, 8))

    # Make ordered intervals plot
    uct.plot_intervals_ordered(
        pred_mean, pred_std, y, n_subset=n_subset, ylims=ylims, ax=axs[0]
    )

    # Make calibration plot
    uct.plot_calibration(pred_mean, pred_std, y, ax=axs[1])
    axs[1].plot([0, 1], [0, 1], color='black', linestyle='--')  # Calibration line

    axs[0].set_title('Ordered estimation intervals', fontsize=14)
    axs[0].set_xlabel('Index (ordered by observed value)', fontsize=12)
    axs[0].set_ylabel('Estimated values and intervals', fontsize=12)

    axs[1].set_title('Average calibration', fontsize=14)
    axs[1].set_xlabel('Estimated proportion in interval', fontsize=14)
    axs[1].set_ylabel('Observed proportion in interval', fontsize=14)

    # for text_obj in axs[1].texts:
    #     if "Miscalibration area" in text_obj.get_text():
    #         text_obj.set_fontsize('large')
    #         break

    fig.suptitle(scenario_title, y=0.85, fontsize=16, fontweight='bold')
    fig.subplots_adjust(wspace=0.25, top=0.85)

    if savefig:
        save_path = '/content/drive/My Drive/iop_ml/plots_scenarios/'
        os.makedirs(save_path, exist_ok=True)
        full_save_path = os.path.join(save_path, f"{plot_save_str}_save.png")
        plt.savefig(full_save_path, dpi=200, bbox_inches='tight')
        print(f"Plot saved to {full_save_path}")

    plt.show()


np.random.seed(11)

# Generate synthetic predictive uncertainty results
n_obs = 650
f, std, y, x = uct.synthetic_sine_heteroscedastic(n_obs)

pred_mean_list = [f]

pred_std_list = [
    std * 0.5,  # overconfident
    std * 2.0,  # underconfident
    std,  # correct
]

scenario_titles = ['Overconfident', 'Underconfident', 'Well-calibrated']

idx_counter = 0
for i, pred_mean in enumerate(pred_mean_list):
    for j, pred_std in enumerate(pred_std_list):
        scenario_title = scenario_titles[j]  # Get the scenario title
        mace = uct.mean_absolute_calibration_error(pred_mean, pred_std, y)
        rmsce = uct.root_mean_squared_calibration_error(pred_mean, pred_std, y)

        idx_counter += 1
        print(f"Scenario {idx_counter}: MACE: {mace:.4f}, RMSCE: {rmsce:.4f}, MA: {ma:.4f}")
        make_plots(pred_mean, pred_std, y, scenario_title, f"row_{idx_counter}")
        print('saved')

