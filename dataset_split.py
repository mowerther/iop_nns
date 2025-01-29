"""
Script for splitting a dataset using random, within-distribution, and out-of-distribution splits.

Example:
    python dataset_split.py datasets_train_test/filtered_df_2319.csv
"""
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.optimize import dual_annealing
from sklearn.model_selection import train_test_split

# Set up constants
summary_cols = ["aph_443", "aNAP_443", "aCDOM_443"]

# 0. Load your dataset with splitting (herein referred to as summary variables/columns)
# Parse command-line args
import argparse
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("filename", help="File with data to split", type=Path)
args = parser.parse_args()

# Load file
my_data = pd.read_csv(args.filename)

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

def objective(x, unique_system_names, train_size, data):
    """
    Objective function for the optimization problem.

    Parameters:
    x (np.array): Array of indices for system names
    unique_system_names (np.array): Array of unique system names
    train_size (int): Size of the training set
    data (pd.DataFrame): Input dataset

    Returns:
    float: Objective function value (similarity score + balance penalty)
    """
    x_unique = np.unique(x.astype(int))
    train_systems = unique_system_names[x_unique]
    test_systems = np.setdiff1d(unique_system_names, train_systems)

    D_train = data[data['system_name'].isin(train_systems)]
    D_test = data[data['system_name'].isin(test_systems)]

    balance_penalty = np.abs(len(D_train) - len(D_test))
    return similarity_score(D_train, D_test) + balance_penalty

def system_data_split(data, train_ratio=0.5, seed=11, max_iterations=10):
    """
    Splits the dataset into train and test sets, ensuring that each set has unique system names.

    The function uses dual_annealing optimization to find the best split of system names between
    train and test sets. The objective function measures the similarity between train and test
    sets based on the splitting (summary) columns and adds a penalty term for imbalance in the number of
    observations between the sets. The dual_annealing algorithm is run for a specified number
    of iterations to find the split that minimizes the objective function value.

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
    unique_system_names = data['system_name'].unique()
    n_systems = len(unique_system_names)

    train_size = int(train_ratio * n_systems)

    x0 = np.random.permutation(n_systems)[:train_size]
    bounds = [(0, n_systems - 1)] * train_size

    best_res = None
    best_obj_val = float("inf")

    for i in range(max_iterations):
        res = dual_annealing(objective, bounds, x0=x0, args=(unique_system_names, train_size, data), seed=seed, callback=progress_callback)
        if res.fun < best_obj_val:
            best_res = res
            best_obj_val = res.fun
        x0 = best_res.x.astype(int)

    x_unique = np.unique(x0)
    optimal_train_systems = unique_system_names[x_unique]
    optimal_test_systems = np.setdiff1d(unique_system_names, optimal_train_systems)

    train_set = data[data['system_name'].isin(optimal_train_systems)]
    test_set = data[data['system_name'].isin(optimal_test_systems)]

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

def dissimilarity_objective(x, unique_system_names, train_size, data):
    """
    Objective function for the optimization problem to maximize dissimilarity.

    Parameters:
    x (np.array): Array of indices for system names
    unique_system_names (np.array): Array of unique system names
    train_size (int): Size of the training set
    data (pd.DataFrame): Input dataset

    Returns:
    float: Objective function value (dissimilarity score + balance penalty)
    """
    x_unique = np.unique(x.astype(int))
    train_systems = unique_system_names[x_unique]
    test_systems = np.setdiff1d(unique_system_names, train_systems)

    D_train = data[data['system_name'].isin(train_systems)]
    D_test = data[data['system_name'].isin(test_systems)]

    balance_penalty = np.abs(len(D_train) - len(D_test))
    return dissimilarity_score(D_train, D_test) + balance_penalty

def system_data_split_oos(data, train_ratio=0.5, seed=12, max_iterations=15):
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
    unique_system_names = data['system_name'].unique()
    n_systems = len(unique_system_names)

    train_size = int(train_ratio * n_systems)

    x0 = np.random.permutation(n_systems)[:train_size]
    bounds = [(0, n_systems - 1)] * train_size

    best_res = None
    best_obj_val = float("inf")

    for i in range(max_iterations):
        res = dual_annealing(dissimilarity_objective, bounds, x0=x0, args=(unique_system_names, train_size, data), seed=seed, callback=progress_callback)
        if res.fun < best_obj_val:
            best_res = res
            best_obj_val = res.fun
        x0 = best_res.x.astype(int)

    x_unique = np.unique(x0)
    optimal_train_systems = unique_system_names[x_unique]
    optimal_test_systems = np.setdiff1d(unique_system_names, optimal_train_systems)

    train_set = data[data['system_name'].isin(optimal_train_systems)]
    test_set = data[data['system_name'].isin(optimal_test_systems)]

    return train_set, test_set

################################
# 4. Inspect datasets, check uniqueness of system names
################################

def check_system_name_uniqueness(train_set, test_set, system_name_col='system_name'):
    """
    Check if system names are unique between train and test sets.

    Parameters:
    train_set (pd.DataFrame): Training dataset
    test_set (pd.DataFrame): Test dataset
    system_name_col (str): Name of the column containing system names

    Returns:
    bool: True if system names are unique, False otherwise
    """
    train_system_names = set(train_set[system_name_col])
    test_system_names = set(test_set[system_name_col])

    train_test_intersection = train_system_names.intersection(test_system_names)

    if not train_test_intersection:
        print("System names are unique in each dataset.")
        return True
    else:
        print("System names are not unique in each dataset.")
        if train_test_intersection:
            print(f"Common system names in train and test sets: {train_test_intersection}")
        return False

################################
# Run script - this code is not executed if the script is imported
################################
if __name__ == "__main__":
    # Random split
    train_set_random, test_set_random = random_split(my_data)
    print(len(train_set_random))
    print(len(test_set_random))

    # Within-distribution split
    train_set_wd, test_set_wd = system_data_split(my_data, seed=43)

    print(len(train_set_wd))
    print(len(test_set_wd))

    # Out-of-distribution split
    train_set_oos, test_set_oos = system_data_split_oos(my_data, seed=42)

    print(len(train_set_oos))
    print(len(test_set_oos))

    # Inspection
    train_system_names = train_set_wd["system_name"].unique()
    test_system_names = test_set_wd["system_name"].unique()

    common_system_names = np.intersect1d(train_system_names, test_system_names)

    print(f"Number of unique system names in train set: {len(train_system_names)}")
    print(f"Number of unique system names in test set: {len(test_system_names)}")
    print(f"Number of common system names: {len(common_system_names)}")

    unique_system_names = check_system_name_uniqueness(train_set_wd,test_set_wd)
