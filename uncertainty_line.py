from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

### Define constants
pred_path = Path("iop_model_predictions/")
save_path = Path("results/")
variables = ("aCDOM_443", "aCDOM_675", "aNAP_443", "aNAP_675", "aph_443", "aph_675")
uncertainty_labels = ("Total", "Aleatoric", "Epistemic")
colors = ("black", "blue", "orange")

### Functions for loading and processing dataframes
def filter_df(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """
    Reorganise a single dataframe to have Instance as its index, only contain data columns, and (optionally) have a suffix added to all columns.
    """
    df_filtered = df.loc[df["Category"] == category]
    return df_filtered.set_index("Instance") \
                      .drop(columns=["Category"])


def reorganise_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorganise an input dataframe so that "Category" becomes a multi-index.
    """
    # Filter dataframes by category into a dictionary
    dataframes = {key: filter_df(df, key) for key in df["Category"].unique()}

    # Join everything back into a single dataframe
    df_combined = pd.concat(dataframes)

    return df_combined


def calculate_percentage_uncertainty(df: pd.DataFrame, *,
                                     reference_key: str="pred_scaled_for_unc", uncertainty_keys: Iterable[str]=("total_unc", "ale_unc", "epi_unc")) -> pd.DataFrame:
    """
    Calculates the percentage uncertainty (total, aleatoric, epistemic) relative to the scaled prediction.

    Parameters:
    - df: the main dataframe containing all data.
    Optional:
    - reference_key: the index for the denominator (default: "pred_scaled_for_unc")
    - uncertainty_keys: the indices for the numerators (default: "total_unc, "ale_unc", "epi_unc")

    Returns:
    - A dataframe with the calculated percentages for the specified columns.
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


# Load data
mdn_wd = read_data(pred_path/"mdn_wd_preds.csv")
mdn_ood = read_data(pred_path/"mdn_ood_preds.csv")

dc_wd = read_data(pred_path/"bnn_dropconnect_wd_preds.csv")
dc_ood = read_data(pred_path/"bnn_dropconnect_ood_preds.csv")

mcd_wd = read_data(pred_path/"bnn_mcd_wd_preds.csv")
mcd_ood = read_data(pred_path/"bnn_mcd_ood_preds.csv")

ens_wd = read_data(pred_path/"ensemble_wd_preds.csv")
ens_ood = read_data(pred_path/"ensemble_ood_preds.csv")

rnn_wd = read_data(pred_path/"rnn_wd_preds.csv")
rnn_ood = read_data(pred_path/"rnn_ood_preds.csv")


### Calculate log-binned statistics
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
binned = log_binned_statistics_combined(rnn_wd)

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

plt.show()
plt.close()
