"""
Functions for randomly sampling data.
"""
import itertools as it

import numpy as np
import pandas as pd


### HELPER FUNCTIONS
def roundrobin(*iterables):
    """
    Visit input iterables in a cycle until each is exhausted.
    From the itertools recipes.
    """
    iterators = map(iter, iterables)
    for num_active in range(len(iterables), 0, -1):
        iterators = it.cycle(it.islice(iterators, num_active))
        yield from map(next, iterators)


### SAMPLING FUNCTIONS
def mcar(X: pd.DataFrame, *args: tuple[pd.DataFrame],
         frac_test: float=0.5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Missing Completely At Random.
    """
    # Sample X (main dataframe)
    X_train = X.sample(frac=1-frac_test)
    X_test = X.drop(index=X_train.index)

    # Sample additional dataframes, if provided
    args_train = [y.loc[X_train.index] for y in args]
    args_test = [y.loc[X_test.index] for y in args]
    args_mixed = list(roundrobin(args_train, args_test))

    return X_train, X_test, *args_mixed


### DICTIONARY FOR EASY ACCESS
splitters = {"mcar": mcar,
             }
