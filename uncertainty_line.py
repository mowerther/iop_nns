from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

### Define constants
pred_path = Path("iop_model_predictions/")
save_path = Path("results/")
variables = ["aCDOM_443", "aCDOM_675", "aNAP_443", "aNAP_675", "aph_443", "aph_675"]


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

raise Exception

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
    total_binned = total_binned.join(aleatoric_binned).join(epistemic_binned)

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
