"""
Script for splitting a dataset using random, within-distribution, and out-of-distribution splits.
Data are split on one system column, provided with the `-s` flag (default: "lake_name").
(Dis)similarity scores are evaluated on multiple `summary_cols`, specified at the start of the script.
Please note that the script can slightly run past its `timeout`; this is not a bug.

Example:
    python dataset_split.py datasets_train_test/filtered_df_2319.csv
    python dataset_split.py datasets_train_test/filtered_df_2319.csv -o path/to/outputs/ -s site_name -t 10 -r 42
"""
from functools import partial
from pathlib import Path
from time import time
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
def random_split(data: pd.DataFrame, *args,
                 test_size: float=0.5, seed: int=1, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into train and test sets, completely at random.

    Parameters:
    data (pd.DataFrame): Input dataset to be split.
    test_size (float): Fraction of data to be assigned to the test set.
    seed (int): Random seed for reproducibility.
    Additional *args/**kwargs are allowed for consistency with the other splits; they are not used.

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
class CallbackProgressor:
    """
    Callback function to print progress during optimization.
    Implemented as a class so it can keep track of the number of minimums, timeout, etc.
    """
    def __init__(self, timeout: int):
        self.current_minimum_number = 0
        self.timeout = timeout  # Minutes
        self.starting_time = time()

    def elapsed_time(self) -> tuple[int, int]:
        """
        Determine elapsed time in minutes and seconds.
        """
        elapsed_seconds = time() - self.starting_time
        elapsed_minutes, spare_seconds = elapsed_seconds // 60, elapsed_seconds % 60
        elapsed_minutes, spare_seconds = int(elapsed_minutes), int(spare_seconds)
        return elapsed_minutes, spare_seconds

    def __call__(self, xk: np.ndarray, fk: float, context: int) -> bool | None:
        """
        Parameters:
        xk (np.ndarray): Current parameter vector.
        fk (float): Current objective function value.
        context (int): Context provided by dual_annealing; 0, 1, or 2.

        Returns:
        bool: True if maximum iterations reached, False otherwise
        """
        # Count up
        self.current_minimum_number += 1
        current_minutes, current_seconds = self.elapsed_time()

        # User feedback
        print(f"{current_minutes:02d}:{current_seconds:02d} - Minimum #{self.current_minimum_number:>3}. Objective function value: {fk: 8.4f}")

        if current_minutes >= self.timeout:
            print(f"Timed out after {current_minutes:d} minute(s).")
            return True


def objective(x: np.ndarray,
              system_column: str, unique_system_names: np.ndarray, train_size: int, data: pd.DataFrame,
              scoring_func: Callable) -> float:
    """
    Objective function for the optimization problem to maximize similarity.
    Measures the similarity between train and test sets based on the splitting
    (summary) columns and adds a penalty term for imbalance in the number of
    observations between the sets.

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


def system_data_split(data: pd.DataFrame, system_column: str, objective_func: Callable, *,
                      train_ratio: float=0.5, seed: int=11, timeout: float=10.) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into train and test sets, ensuring that each set has unique system names.

    The function uses dual_annealing optimization to find the best split of system names between
    train and test sets. An objective function (e.g. similarity, dissimilarity) must be specified.
    The dual_annealing algorithm is run for a specified number of iterations to find the split that minimizes the objective function value.

    Parameters:
    data (pd.DataFrame): Input dataset to be split.
    system_column (str): Name of the column to split on (e.g. lake_name).
    objective_func (Callable): Objective function.

    train_ratio (float): Ratio of unique system names to be assigned to the train set.
    seed (int): Random seed for reproducibility.
    timeout (float): Maximum time [min] to spend optimizing.

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

    # Apply dual_annealing for up to `timeout` minutes
    progress_callback = CallbackProgressor(timeout=timeout)
    res = dual_annealing(objective_func, bounds, x0=x0, args=(system_column, unique_system_names, train_size, data), seed=seed, callback=progress_callback)
        # Note that the `seed` kwarg is being deprecated and should be replaced with `rng`
    x0 = res.x.astype(int)

    # Apply final result
    x_unique = np.unique(x0)
    optimal_train_systems = unique_system_names[x_unique]
    optimal_test_systems = np.setdiff1d(unique_system_names, optimal_train_systems)

    train_set = data[data[system_column].isin(optimal_train_systems)]
    test_set = data[data[system_column].isin(optimal_test_systems)]

    return train_set, test_set


################################
# 2. Dataset split algorithm - Within-distribution
################################
def similarity_score(D1: pd.DataFrame, D2: pd.DataFrame) -> float:
    """
    Calculate the similarity score between two datasets.

    Parameters:
    D1 (pd.DataFrame): First dataset
    D2 (pd.DataFrame): Second dataset

    Returns:
    float: Similarity score based on the mean difference of summary columns
    """
    return (D1[summary_cols].mean() - D2[summary_cols].mean()).abs().sum()

similarity_objective = partial(objective, scoring_func=similarity_score)
system_wd_split = partial(system_data_split, objective_func=similarity_objective)

##############################
# 3. Dataset split algorithm - Out-of-distribution
##############################
def dissimilarity_score(D1: pd.DataFrame, D2: pd.DataFrame) -> float:
    """
    Calculate the dissimilarity score between two datasets.

    Parameters:
    D1 (pd.DataFrame): First dataset
    D2 (pd.DataFrame): Second dataset

    Returns:
    float: Dissimilarity score based on percentile differences of summary columns
    """
    # Find percentile values in either dataset
    percentiles = np.arange(10, 91, 10)  # [10, 20, ..., 90]
    quantiles = percentiles / 100
    D1_quantiles, D2_quantiles = [df[summary_cols].quantile(quantiles) for df in (D1, D2)]

    # Calculate total difference between percentiles
    quantile_diff = (D1_quantiles - D2_quantiles).abs()
    score = quantile_diff.sum().sum()

    return -score

dissimilarity_objective = partial(objective, scoring_func=dissimilarity_score)
system_ood_split = partial(system_data_split, objective_func=dissimilarity_objective)

################################
# Dataset split algorithm - combined
################################
splitters = {"random": {"full_name": "random", "func": random_split},
             "wd": {"full_name": "within-distribution", "func": system_wd_split},
             "ood": {"full_name": "out-of-distribution", "func": system_ood_split},
            }

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename", help="File with data to split.", type=Path)
    parser.add_argument("-o", "--output_folder", help="Folder to save files with splits to.", type=Path, default=".")
    parser.add_argument("-s", "--system_column", help="Column with system names, on which to split the data.", default="lake_name")
    parser.add_argument("-t", "--timeout", help="Maximum time [min] to spend on each split.", type=float, default=10.)
    parser.add_argument("-r", "--rng", help="Seed for random number generator (RNG).", type=int, default=42)
    args = parser.parse_args()

    # Load file
    my_data = pd.read_csv(args.filename)

    # Apply splits
    for label, splitter in splitters.items():
        # Setup
        name, func = splitter["full_name"], splitter["func"]
        print("\n\n################################")
        print(f"Now applying {name} split:")

        # Application
        train_set, test_set = func(my_data, args.system_column, timeout=args.timeout, seed=args.rng)

        # Feedback
        print_set_length(name, train_set, test_set)
        check_system_name_uniqueness(train_set, test_set, args.system_column)

        # Save to file
        train_set.to_csv(args.output_folder/f"{label}_train_set.csv")
        test_set.to_csv(args.output_folder/f"{label}_test_set.csv")
        print(f"Saved {name} split train and test sets to {args.output_folder.absolute()}")
        print("################################")
