"""
Script for splitting a dataset using random, within-distribution, and out-of-distribution splits.
Data are split on one system column, provided with the `-s` flag (default: "lake_name").
(Dis)similarity scores are evaluated on multiple `summary_cols`, specified at the start of the script.

Example:
    python dataset_split.py datasets_train_test/filtered_df_2319.csv
    python dataset_split.py datasets_train_test/filtered_df_2319.csv -s site_name
"""
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import dual_annealing
from sklearn.model_selection import train_test_split

# Set up constants
summary_cols = ["aph_443", "aNAP_443", "aCDOM_443"]  # Variables used in (dis)similarity scores


################################
# 1. Random split
################################
def random_split(data: pd.DataFrame, *, test_size: float=0.5, seed: int=1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into train and test sets, completely at random.

    Parameters:
    data (pd.DataFrame): Input dataset to be split.
    test_size (float): Fraction of data to be assigned to the test set.
    seed (int): Random seed for reproducibility.

    Returns:
    train_set (pd.DataFrame): Train set.
    test_set (pd.DataFrame): Test set.
    """
    np.random.seed(seed)
    randomized_df = data.sample(frac=1).reset_index(drop=True)
    train_set, test_set = train_test_split(randomized_df, test_size=test_size)

    return train_set, test_set


################################
# Dataset split algorithm - common functionality
################################
def progress_callback(xk, fk, *args):
    """
    Callback function to print progress during optimization.

    Parameters:
    xk: Current parameter vector
    fk: Current objective function value
    *args: Additional arguments (max_iterations)

    Returns:
    bool: True if maximum iterations reached, False otherwise
    """
    global iteration
    global best_obj_val
    max_iterations = args[0]

    iteration += 1

    if fk < best_obj_val:
        best_obj_val = fk
        print(f"Objective function value at iteration {iteration}: {fk}")

    if iteration >= max_iterations:
        return True

iteration = 0
best_obj_val = float("inf")


def objective(x: np.ndarray,
              system_column: str, unique_system_names: np.ndarray, train_size: int, data: pd.DataFrame,
              scoring_func: Callable) -> float:
    """
    Objective function for the optimization problem to maximize similarity.

    `x` is what the dual_annealing algorithm modifies.
    All arguments after `x` are fixed parameters needed to completely specify the objective function.
    For convenience, create partial functions for corresponding `scoring_func`s, e.g. similarity_score, dissimilarity_score.

    Parameters:
    x (np.array): Array of indices for system names

    system_column (str): Name of the column to split on (e.g. lake_name).
    unique_system_names (np.array): Array of unique system names
    train_size (int): Size of the training set
    data (pd.DataFrame): Input dataset
    scoring_func (Callable): Function that determines the score

    Returns:
    float: Objective function value (similarity score + balance penalty)
    """
    # Convert system names into data indices
    x_unique = np.unique(x.astype(int))
    train_systems = unique_system_names[x_unique]
    test_systems = np.setdiff1d(unique_system_names, train_systems)

    D_train = data[data[system_column].isin(train_systems)]
    D_test = data[data[system_column].isin(test_systems)]

    # Calculate and return score
    balance_penalty = np.abs(len(D_train) - len(D_test))
    return scoring_func(D_train, D_test) + balance_penalty


################################
# 2. Dataset split algorithm - within-distribution
################################
def similarity_score(D1, D2):
    """
    Calculate the similarity score between two datasets.

    Parameters:
    D1 (pd.DataFrame): First dataset
    D2 (pd.DataFrame): Second dataset

    Returns:
    float: Similarity score based on the mean difference of summary columns
    """
    return np.abs(D1[summary_cols].mean() - D2[summary_cols].mean()).sum()

similarity_objective = partial(objective, scoring_func=similarity_score)

def system_data_split(data: pd.DataFrame, system_column: str, *,
                      train_ratio: float=0.5, seed: int=11, max_iterations: int=10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into train and test sets, ensuring that each set has unique system names.

    The function uses dual_annealing optimization to find the best split of system names between
    train and test sets. The objective function measures the similarity between train and test
    sets based on the splitting (summary) columns and adds a penalty term for imbalance in the number of
    observations between the sets. The dual_annealing algorithm is run for a specified number
    of iterations to find the split that minimizes the objective function value.

    Parameters:
    data (pd.DataFrame): Input dataset to be split.
    system_column (str): Name of the column to split on (e.g. lake_name).
    train_ratio (float): Ratio of unique system names to be assigned to the train set.
    seed (int): Random seed for reproducibility.
    max_iterations (int): Maximum number of iterations for the optimization algorithm.

    Returns:
    train_set (pd.DataFrame): Train set with unique system names.
    test_set (pd.DataFrame): Test set with unique system names.
    """
    np.random.seed(seed)

    # Find unique systems and determine train/test set size
    unique_system_names = data[system_column].unique()
    n_systems = len(unique_system_names)
    train_size = int(train_ratio * n_systems)

    # Set up variables for dual_annealing function
    x0 = np.random.permutation(n_systems)[:train_size]
    bounds = [(0, n_systems - 1)] * train_size

    # Apply dual_annealing `max_iterations`, using the previous best estimate as the new starting condition
    best_res = None
    best_obj_val = float("inf")

    for i in range(max_iterations):
        res = dual_annealing(similarity_objective, bounds, x0=x0, args=(system_column, unique_system_names, train_size, data), seed=seed, callback=progress_callback)
        # Note that the `seed` kwarg is being deprecated and should be replaced with `rng`
        if res.fun < best_obj_val:
            best_res = res
            best_obj_val = res.fun
        x0 = best_res.x.astype(int)

    # Apply final result
    x_unique = np.unique(x0)
    optimal_train_systems = unique_system_names[x_unique]
    optimal_test_systems = np.setdiff1d(unique_system_names, optimal_train_systems)

    train_set = data[data[system_column].isin(optimal_train_systems)]
    test_set = data[data[system_column].isin(optimal_test_systems)]

    return train_set, test_set


##############################
# 3. Out-of-distribution split
##############################
def dissimilarity_score(D1, D2):
    """
    Calculate the dissimilarity score between two datasets.

    Parameters:
    D1 (pd.DataFrame): First dataset
    D2 (pd.DataFrame): Second dataset

    Returns:
    float: Dissimilarity score based on percentile differences of summary columns
    """
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    score = 0
    for col in summary_cols:
        for percentile in percentiles:
            d1_percentile = np.percentile(D1[col], percentile)
            d2_percentile = np.percentile(D2[col], percentile)
            score += np.abs(d1_percentile - d2_percentile)
    return -score

dissimilarity_objective = partial(objective, scoring_func=dissimilarity_score)

def system_data_split_ood(data: pd.DataFrame, system_column: str, *,
                          train_ratio: float=0.5, seed: int=12, max_iterations: int=15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into train and test sets for out-of-distribution (OOD) scenario.

    The function uses dual_annealing optimization to find the best split of system names between
    train and test sets. The objective function measures the dissimilarity between train and test
    sets based on the splitting (summary) columns and adds a penalty term for imbalance in the number of
    observations between the sets. The dual_annealing algorithm is run for a specified number
    of iterations to find the split that maximizes the dissimilarity.

    Parameters:
    data (pd.DataFrame): Input dataset to be split.
    train_ratio (float): Ratio of unique system names to be assigned to the train set.
    seed (int): Random seed for reproducibility.
    max_iterations (int): Maximum number of iterations for the optimization algorithm.

    Returns:
    train_set (pd.DataFrame): Train set with unique system names.
    test_set (pd.DataFrame): Test set with unique system names.
    """
    np.random.seed(seed)

    # Find unique systems and determine train/test set size
    unique_system_names = data[system_column].unique()
    n_systems = len(unique_system_names)
    train_size = int(train_ratio * n_systems)

    # Set up variables for dual_annealing function
    x0 = np.random.permutation(n_systems)[:train_size]
    bounds = [(0, n_systems - 1)] * train_size

    # Apply dual_annealing `max_iterations`, using the previous best estimate as the new starting condition
    best_res = None
    best_obj_val = float("inf")

    for i in range(max_iterations):
        res = dual_annealing(dissimilarity_objective, bounds, x0=x0, args=(system_column, unique_system_names, train_size, data), seed=seed, callback=progress_callback)
        if res.fun < best_obj_val:
            best_res = res
            best_obj_val = res.fun
        x0 = best_res.x.astype(int)

    # Apply final result
    x_unique = np.unique(x0)
    optimal_train_systems = unique_system_names[x_unique]
    optimal_test_systems = np.setdiff1d(unique_system_names, optimal_train_systems)

    train_set = data[data[system_column].isin(optimal_train_systems)]
    test_set = data[data[system_column].isin(optimal_test_systems)]

    return train_set, test_set


################################
# 4. Inspect datasets, check uniqueness of system names
################################
def print_set_length(name: str, train_set: pd.DataFrame, test_set: pd.DataFrame) -> None:
    """
    Print the lengths of the train and test sets with formatting.

    Parameters:
    name (str): Name of the split type
    train_set (pd.DataFrame): Training dataset
    test_set (pd.DataFrame): Test dataset
    """
    print(f"{name.capitalize()} split: {len(train_set)} in train set; {len(test_set)} in test set.")


def check_system_name_uniqueness(train_set: pd.DataFrame, test_set: pd.DataFrame, system_column: str="system_name") -> bool:
    """
    Check if system names are unique between train and test sets.

    Parameters:
    train_set (pd.DataFrame): Training dataset
    test_set (pd.DataFrame): Test dataset
    system_column (str): Name of the column containing system names

    Returns:
    bool: True if system names are unique, False otherwise
    """
    # Find common elements
    train_system_names = train_set[system_column].unique()
    test_system_names = test_set[system_column].unique()

    common_system_names = np.intersect1d(train_system_names, test_system_names)
    any_in_common = (len(common_system_names) > 0)

    # User feedback
    print(f"Number of unique system names in train set: {len(train_system_names)}")
    print(f"Number of unique system names in test set: {len(test_system_names)}")
    print(f"Number of common system names: {len(common_system_names)}")

    if any_in_common:
        print(f"Common system names in train and test sets: {common_system_names}")

    return any_in_common


################################
# Run script - this code is not executed if the script is imported
################################
if __name__ == "__main__":
    # Parse command-line args
    import argparse
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("filename", help="File with data to split", type=Path)
    parser.add_argument("-s", "--system_column", help="Column with system names, on which to split the data", default="lake_name")
    args = parser.parse_args()

    # Load file
    my_data = pd.read_csv(args.filename)

    # Random split
    print("Now applying random split:")
    train_set_random, test_set_random = random_split(my_data, seed=41)
    print_set_length("random", train_set_random, test_set_random)
    check_system_name_uniqueness(train_set_random, test_set_random, args.system_column)

    # Within-distribution split
    print("Now applying within-distribution split:")
    train_set_wd, test_set_wd = system_data_split(my_data, args.system_column, seed=43)
    print_set_length("within-distribution", train_set_wd, test_set_wd)
    check_system_name_uniqueness(train_set_wd, test_set_wd, args.system_column)

    # Out-of-distribution split
    print("Now applying out-of-distribution split:")
    train_set_ood, test_set_ood = system_data_split_ood(my_data, args.system_column, seed=42)
    print_set_length("out-of-distribution", train_set_ood, test_set_ood)
    check_system_name_uniqueness(train_set_ood, test_set_ood, args.system_column)
