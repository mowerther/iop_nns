from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Define constants
pred_path = Path("iop_model_predictions/")
columns_of_interest = ['aCDOM_443', 'aCDOM_675', 'aNAP_443', 'aNAP_675', 'aph_443', 'aph_675']

# Load data
mdn_wd = pd.read_csv(pred_path/'mdn_wd_preds.csv')
mdn_ood = pd.read_csv(pred_path/'mdn_ood_preds.csv')

dc_wd = pd.read_csv(pred_path/'bnn_dropconnect_wd_preds.csv')
dc_ood = pd.read_csv(pred_path/'bnn_dropconnect_ood_preds.csv')

mcd_wd = pd.read_csv(pred_path/'bnn_mcd_wd_preds.csv')
mcd_ood = pd.read_csv(pred_path/'bnn_mcd_ood_preds.csv')

ens_wd = pd.read_csv(pred_path/'ensemble_wd_preds.csv')
ens_ood = pd.read_csv(pred_path/'ensemble_ood_preds.csv')

rnn_wd = pd.read_csv(pred_path/'rnn_wd_preds.csv')
rnn_ood = pd.read_csv(pred_path/'rnn_ood_preds.csv')


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
    result = (df_cat_reset_1[columns_of_interest] / df_cat_reset_2[columns_of_interest]) * 100

    # Use absolute value to account for negative values
    result = np.abs(result)

    return result

# Use the updated function to calculate percentages directly from the main dataframe
percent_total_uncertainty = calculate_percentage_from_category(rnn_ood, 'total_unc', 'pred_scaled_for_unc', columns_of_interest)

percent_aleatoric_uncertainty = calculate_percentage_from_category(rnn_ood, 'ale_unc', 'pred_scaled_for_unc', columns_of_interest)

percent_epistemic_uncertainty = calculate_percentage_from_category(rnn_ood, 'epi_unc', 'pred_scaled_for_unc', columns_of_interest)

# Other categories
pred_scaled = rnn_ood[rnn_ood['Category'] == 'pred_scaled_for_unc']

y_true = rnn_ood[rnn_ood['Category'] == 'y_true']
y_pred = rnn_ood[rnn_ood['Category'] == 'y_pred']

std_pred = rnn_ood[rnn_ood['Category'] == 'pred_std'].reset_index(drop=True)

# Calculate log-binned statistics
statistics = ["mean", "std"]
col = "aph_675"
# y_log = np.log10(y_true[col])
# vmin, vmax = y_log.min(), y_log.max()

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

    # Find the indices per bin
    slices = [(x_log >= centre - binwidth) & (x_log <= centre + binwidth) for centre in bins_log]
    binned = [y.loc[s] for s in slices]

    # Calculate statistics
    binned_means = [b.mean() for b in binned]
    binned_stds = [b.std() for b in binned]

    # Wrap into dataframe
    binned = pd.DataFrame(data={"centre": np.power(10, bins_log), "mean": binned_means, "std": binned_stds})
    return binned


def log_binned_statistics_combined(x: pd.DataFrame, total: pd.DataFrame, aleatoric: pd.DataFrame, epistemic: pd.DataFrame, column: str, **kwargs) -> pd.DataFrame:
    """
    Calculate log binned statistics for each of the uncertainty dataframes relative to x, and combine them into a single dataframe.
    """
    # Bin individually
    total_binned = log_binned_statistics(x[column], total[column])
    aleatoric_binned = log_binned_statistics(x[column], aleatoric[column])
    epistemic_binned = log_binned_statistics(x[column], epistemic[column])

    # Extract data columns
    aleatoric_binned = aleatoric_binned[statistics].add_suffix("_aleatoric")
    epistemic_binned = epistemic_binned[statistics].add_suffix("_epistemic")

    # Combine and return
    total_binned = total_binned.join(aleatoric_binned)
    total_binned = total_binned.join(epistemic_binned)

    return total_binned


def plot_log_binned_statistics(binned: pd.DataFrame, *, ax: Optional[plt.Axes]=None) -> None:
    """
    Given a DataFrame containing log-binned statistics, plot the total/aleatoric/epistemic uncertainties.
    Plots a line for the mean uncertainty and a shaded area for the standard deviation.
    If no ax is provided, a new figure is created.
    """
    # Set up a new figure if desired
    NEW_FIGURE = (ax is None)
    if NEW_FIGURE:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure


binned = log_binned_statistics_combined(y_true, percent_total_uncertainty, percent_aleatoric_uncertainty, percent_epistemic_uncertainty, col)

# Plot
fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)

# Plot means
binned.plot("centre", "mean", ax=ax, color="black", label="Total")
binned.plot("centre", "mean_aleatoric", ax=ax, color="blue", label="Aleatoric")
binned.plot("centre", "mean_epistemic", ax=ax, color="orange", label="Epistemic")

# Plot stds
ax.fill_between(binned["centre"], binned["mean"] - binned["std"], binned["mean"] + binned["std"], color="black", alpha=0.2)
ax.fill_between(binned["centre"], binned["mean_aleatoric"] - binned["std_aleatoric"], binned["mean_aleatoric"] + binned["std_aleatoric"], color="blue", alpha=0.2)
ax.fill_between(binned["centre"], binned["mean_epistemic"] - binned["std_epistemic"], binned["mean_epistemic"] + binned["std_epistemic"], color="orange", alpha=0.2)

ax.set_xscale("log")
ax.set_ylim(ymin=0)
ax.set_xlabel(f"in situ {col}")
ax.set_ylabel(f"average uncertainty {col}")
ax.grid(True)

plt.show()
plt.close()
