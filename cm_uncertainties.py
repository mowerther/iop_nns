import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import pandas as pd
import numpy as np

pred_path = 'iop_model_predictions/'

mdn_wd = pd.read_csv(pred_path + 'mdn_wd_preds.csv')
mdn_ood = pd.read_csv(pred_path + 'mdn_ood_preds.csv')
dc_wd = pd.read_csv(pred_path + 'bnn_dropconnect_wd_preds.csv')
dc_ood = pd.read_csv(pred_path + 'bnn_dropconnect_ood_preds.csv')
mcd_wd = pd.read_csv(pred_path + 'bnn_mcd_wd_preds.csv')
mcd_ood = pd.read_csv(pred_path + 'bnn_mcd_ood_preds.csv')
ens_wd = pd.read_csv(pred_path + 'ensemble_wd_preds.csv')
ens_ood = pd.read_csv(pred_path + 'ensemble_ood_preds.csv')
rnn_wd = pd.read_csv(pred_path + 'rnn_wd_preds.csv')
rnn_ood = pd.read_csv(pred_path + 'rnn_ood_preds.csv')

# Ensure all dfs are loaded
dataframes = {
    'mdn_wd': mdn_wd,
    'mdn_ood': mdn_ood,
    'dc_wd': dc_wd,
    'dc_ood': dc_ood,
    'mcd_wd': mcd_wd,
    'mcd_ood': mcd_ood,
    'ens_wd': ens_wd,
    'ens_ood': ens_ood,
    'rnn_wd' : rnn_wd,
    'rnn_ood' : rnn_ood
}

iops_of_interest = ['aCDOM_443', 'aCDOM_675', 'aNAP_443', 'aNAP_675', 'aph_443', 'aph_675']

def calculate_percentage_from_category(df, category1, category2, iops_of_interest):
    """
    Filters two categories from the main dataframe, resets their indexes,
    selects the specific columns, and calculates the percentage of category1 over category2 for those columns.

    """
    df_cat_reset_1 = df[df['Category'] == category1].reset_index(drop=True)
    df_cat_reset_2 = df[df['Category'] == category2].reset_index(drop=True)

    result = (df_cat_reset_1[iops_of_interest] / df_cat_reset_2[iops_of_interest]) * 100

    return result

def calculate_uncertainties_and_categories(df, iops_of_interest):
    result_dict = {
        'percent_total_uncertainty': calculate_percentage_from_category(df, 'total_unc', 'pred_scaled_for_unc', iops_of_interest),
        'percent_aleatoric_uncertainty': calculate_percentage_from_category(df, 'ale_unc', 'pred_scaled_for_unc', iops_of_interest),
        'percent_epistemic_uncertainty': calculate_percentage_from_category(df, 'epi_unc', 'pred_scaled_for_unc', iops_of_interest),
        'pred_scaled': df[df['Category'] == 'pred_scaled_for_unc'],
        'y_true': df[df['Category'] == 'y_true'],
        'y_pred': df[df['Category'] == 'y_pred'],
        'std_pred': df[df['Category'] == 'pred_std'].reset_index(drop=True)
    }
    return result_dict

# Apply the function to each df and store the results in a new dict
results = {df_name: calculate_uncertainties_and_categories(df, iops_of_interest) for df_name, df in dataframes.items()}


# Formatting for CM

def filter_uncertainties(df_total, df_aleatoric, df_epistemic):
    # Identify indices where any of the uncertainties are < 0 or > 200 in any of the three dataframes
    mask = (df_total < 0) | (df_total > 1000) | (df_aleatoric < 0) | (df_aleatoric > 1000) | (df_epistemic < 0) | (df_epistemic > 1000)
    filtered_indices = mask.any(axis=1)

    # Filter out the rows from all dataframes
    df_total_filtered = df_total[~filtered_indices]
    df_aleatoric_filtered = df_aleatoric[~filtered_indices]
    df_epistemic_filtered = df_epistemic[~filtered_indices]

    return df_total_filtered, df_aleatoric_filtered, df_epistemic_filtered

# Iterate over each result and apply the filtering function
for key, result in results.items():
    result['percent_total_uncertainty'], result['percent_aleatoric_uncertainty'], result['percent_epistemic_uncertainty'], = filter_uncertainties(
        result['percent_total_uncertainty'],
        result['percent_aleatoric_uncertainty'],
        result['percent_epistemic_uncertainty']
    )

# example
results['mdn_wd']['percent_total_uncertainty']['aCDOM_675'].median()
# should be 10.73736

# Assuming the `results` dictionary is already populated as described,
# and contains the necessary data structures from earlier operations.

def calculate_average_uncertainties(results, iops_of_interest):
    averages = {}
    for model, data in results.items():
        model_averages = {}
        for uncertainty_type in ['percent_total_uncertainty', 'percent_aleatoric_uncertainty', 'percent_epistemic_uncertainty']:
            df = data[uncertainty_type]
            avg_uncertainties = df[iops_of_interest].mean().to_dict()
            model_averages[uncertainty_type] = avg_uncertainties
        averages[model] = model_averages
    return averages

# Calculate average uncertainties
average_uncertainties = calculate_average_uncertainties(results, iops_of_interest)

def prepare_cm_data(average_uncertainties, uncertainty_type):
    # Initialize empty dfs for WD and OOD data
    data_wd = pd.DataFrame(columns=iops_of_interest)
    data_ood = pd.DataFrame(columns=iops_of_interest)

    # Iterate through models and organize data into WD and OOD dfs
    for model, data in average_uncertainties.items():
        if 'wd' in model:
            data_wd.loc[model.replace('_wd', '')] = pd.Series(data[uncertainty_type])
        elif 'ood' in model:
            data_ood.loc[model.replace('_ood', '')] = pd.Series(data[uncertainty_type])

    return data_wd, data_ood

data_total_wd, data_total_ood = prepare_cm_data(average_uncertainties, 'percent_total_uncertainty')
data_aleatoric_wd, data_aleatoric_ood = prepare_cm_data(average_uncertainties, 'percent_aleatoric_uncertainty')
data_epistemic_wd, data_epistemic_ood = prepare_cm_data(average_uncertainties, 'percent_epistemic_uncertainty')

fraction_aleatoric_wd = data_aleatoric_wd / data_total_wd
fraction_aleatoric_ood = data_aleatoric_ood / data_total_ood

data_uncertainty = [data_total_wd, data_total_ood]
cmap_uncertainty = plt.cm.cividis.resampled(10)
norm_uncertainty = Normalize(vmin=0, vmax=20)

data_fraction = [fraction_aleatoric_wd, fraction_aleatoric_ood]
cmap_fraction = plt.cm.BrBG.resampled(10)
norm_fraction = Normalize(vmin=0, vmax=1)

# Setup figure and axes
# fig, axs = plt.subplots(2, 2, figsize=(8, 10), sharex=True, sharey=True, gridspec_kw={'width_ratios': [1, 1],'wspace': -0.55, 'hspace': 0.2})
fig, axs = plt.subplots(2, 2, figsize=(8, 7), sharex=True, sharey=True, gridspec_kw={"wspace": 0.1, "hspace": -0.15})
plt.style.use('default')

new_model_labels = ['MDN', 'BNN DC', 'BNN MCD', 'ENS NN', 'RNN']
variables = ['aCDOM_443', 'aCDOM_675', 'aNAP_443', 'aNAP_675', 'aph_443', 'aph_675']
display_titles = ['a$_{CDOM}$ 443', 'a$_{CDOM}$ 675', 'a$_{NAP}$ 443', 'a$_{NAP}$ 675', 'a$_{ph}$ 443', 'a$_{ph}$ 675']

# Plot data as matrices
for ax, data in zip(axs[0], data_uncertainty):
    ax.imshow(data, cmap=cmap_uncertainty, norm=norm_uncertainty)

for ax, data in zip(axs[1], data_fraction):
    ax.imshow(data, cmap=cmap_fraction, norm=norm_fraction)

# Plot individual values
for ax, data in zip(axs.ravel(), [*data_uncertainty, *data_fraction]):
    for i, x in enumerate(new_model_labels):
        for j, y in enumerate(variables):
            ax.text(j, i, f"{data.iloc[i, j]:.1f}", ha="center", va="center", color="w")

# Adjust x/y labels
wrap_labels = lambda labels: list(zip(*enumerate(labels)))  # (0, 1, ...) (label0, label1, ...)
for ax in axs[1]:
    ax.set_xticks(*wrap_labels(display_titles), rotation=45, ha="right")

for ax in axs[:, 0]:
    ax.set_yticks(*wrap_labels(new_model_labels))

# Colour bars - shrink factor is manual, but easier than using make_axes_locatable with multiple axes
cbar = fig.colorbar(ScalarMappable(norm=norm_uncertainty, cmap=cmap_uncertainty), ax=axs[0].tolist(), fraction=0.05, pad=0.03, shrink=0.7)
cbar.set_label("Mean uncertainty [%]", size=12, weight="bold")

cbar = fig.colorbar(ScalarMappable(norm=norm_fraction, cmap=cmap_fraction), ax=axs[1].tolist(), fraction=0.05, pad=0.03, shrink=0.7)
cbar.set_label("Aleatoric fraction", size=12, weight="bold")
cbar.set_ticks(np.arange(0, 1.01, 0.2))

# Main titles for each row
axs[0, 0].set_title("Within distribution (WD)", fontsize=12, fontweight="bold")
axs[0, 1].set_title("Out of distribution (OOD)", fontsize=12, fontweight="bold")

# Super labels
fig.supxlabel('IOPs', fontsize=12, fontweight='bold')
fig.supylabel('Models', fontsize=12, fontweight='bold')

# Save
save_path = './'
plt.savefig(save_path + 'avg_uncertainty_results.png', dpi=200, bbox_inches='tight')

plt.show()
