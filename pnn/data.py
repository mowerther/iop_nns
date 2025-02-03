"""
Functions for reading (split) input data.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from . import constants as c


### HELPER FUNCTIONS
capitalise_iops = {"acdom_443": "aCDOM_443", "acdom_675": "aCDOM_675", "anap_443": "aNAP_443", "anap_675": "aNAP_675",}

def _find_rrs_columns(data: pd.DataFrame) -> list[str]:
    """
    Find all columns that contain the string "Rrs_" and return these in order.
    """
    return sorted([col for col in data.columns if "Rrs_" in col])


### PRE-PROCESSING
def select_scenarios(prisma: bool) -> tuple[c.Parameter, list[c.Parameter], list[c.Parameter], Callable]:
    """
    Select the desired scenarios - in situ (`prisma=False`) or PRISMA (`prisma=True`).
    Returns:
        - Label (in situ or PRISMA)
        - Scenarios for plotting: flat list
        - IOPs for plotting: flat list
        - Function to load data
    """
    if prisma:
        return c.prisma, c.scenarios_prisma, c.iops_aph, read_prisma_matchups
    else:
        return c.gloria, c.scenarios_123, c.iops, read_insitu_data


@dataclass
class DataScenario:
    """
    Simple class to combine training/testing data, ensuring they are always in the correct order.
    Optionally includes a re-scaler for X, so that it may be used again later.
    """
    train_scenario: c.Parameter
    train_data: pd.DataFrame
    test_scenarios_and_data: dict[c.Parameter, pd.DataFrame]
    X_scaler: Optional[RobustScaler] = None

    def __iter__(self):
        """
        Enables unpacking, e.g.
            for train_scenario, train_data, test_scenarios in datascenarios:
        """
        return iter([self.train_scenario, self.train_data, self.test_scenarios_and_data])


### DATA RESCALING
def generate_rescaler_rrs(data: np.ndarray) -> RobustScaler:
    """
    Train a rescaler on R_rs data, but do not apply it.
    Just a thin wrapper around RobustScaler for interface consistency.
    """
    scaler = RobustScaler().fit(data)
    return scaler


def generate_rescaler_iops(data: np.ndarray) -> MinMaxScaler:
    """
    Train a rescaler on R_rs data, but do not apply it.
    Just a thin wrapper around RobustScaler for interface consistency.
    """
    scaler = MinMaxScaler().fit(data)
    return scaler


### INPUT / OUTPUT
def read_insitu_full(folder: Path | str=c.insitu_data_path) -> pd.DataFrame:
    """
    Read the original in situ dataset from a given folder into a DataFrame.
    """
    folder = Path(folder)
    data = pd.read_csv(folder/"filtered_df_2319.csv")
    return data


def _read_and_preprocess_insitu_data(filename: Path | str) -> pd.DataFrame:
    """
    Read and pre-process a single in situ dataset.
    """
    data = pd.read_csv(filename)

    # Filter wavelengths
    data = data.drop(columns=[col for col in _find_rrs_columns(data) if int(col.split("_")[1]) not in c.wavelengths_123])

    return data


_filenames_insitu = ["random_train_set.csv", "random_test_set.csv", "wd_train_set.csv", "wd_test_set.csv", "ood_train_set.csv", "ood_test_set.csv"]
def read_insitu_data(folder: Path | str=c.insitu_data_path) -> tuple[DataScenario]:
    """
    Read the random/wd/ood-split in situ data from a given folder into a number of DataFrames.
    The output consists of DataScenario objects which can be iterated over.
        the scalers are included in the DataScenarios for re-use if desired.
    Filenames are hardcoded.
    """
    folder = Path(folder)

    ### LOAD DATA IN ORDER
    train_set_random, test_set_random, train_set_wd, test_set_wd, train_set_ood, test_set_ood = [_read_and_preprocess_insitu_data(folder/filename) for filename in _filenames_insitu]

    ### ORGANISE TRAIN/TEST SETS
    random = DataScenario(c.random_split, train_set_random, {c.random_split: test_set_random})
    wd = DataScenario(c.wd, train_set_wd, {c.wd: test_set_wd})
    ood = DataScenario(c.ood, train_set_ood, {c.ood: test_set_ood})

    return random, wd, ood


### READ PRISMA MATCH-UP DATA (INCLUDING GLORIA+)
def _convert_excel_date(dates: pd.Series) -> pd.Series:
    """
    Convert Excel-format dates (e.g. 44351) to strings in PRISMA/CNR format (e.g. "04.06.2021").
    """
    datetimes = pd.to_datetime(dates.astype(float), unit="D", origin="1899-12-30")
    strings = datetimes.dt.strftime("%d.%m.%Y")
    return strings


def read_prisma_insitu(filename: Path | str=c.prisma_matchup_path/"case_1_insitu_vs_insitu"/"prisma_insitu.csv", *,
                       filter_invalid_dates=False) -> pd.DataFrame:
    """
    Read the PRISMA in situ match-up data.
    If `filter_invalid_dates`, filter out invalid time stamps.
    """
    data = pd.read_csv(filename)
    data = data.rename(columns={f"{nm}_prisma_insitu": f"Rrs_{nm}" for nm in c.wavelengths_prisma})
    data = data.rename(columns=capitalise_iops)
    if filter_invalid_dates:
        to_filter = ~data["date"].str.contains(".", regex=False)
        data.loc[to_filter, "date"] = _convert_excel_date(data.loc[to_filter, "date"])

    return data


def read_prisma_matchups(folder: Path | str=c.prisma_matchup_path) -> tuple[DataScenario]:
    """
    Read the PRISMA match-up data from a given folder into a number of DataFrames.
    The output consists of DataScenario objects which can be iterated over.
    Note that the data in the CSV files have NOT been rescaled, so this must be done here;
        the scalers are included in the DataScenarios for re-use if desired.
    Filenames are hardcoded.
    """
    ### LOAD DATA AND RENAME COLUMNS TO CONSISTENT FORMAT
    # train_X, train_y for GLORIA-based scenarios (general case)
    gloria_resampled = pd.read_csv(folder/"case_1_insitu_vs_insitu"/"gloria_res_prisma_s.csv").rename(columns={f"{nm}_prisma_res": f"Rrs_{nm}" for nm in c.wavelengths_prisma})

    # train_X, train_y for GLORIA+PRISMA-based scenarios (local knowledge)
    combined_insitu = pd.read_csv(folder/"case_3_local_insitu_vs_aco"/"combined_local_train_df.csv").rename(columns={f"{nm}_prisma_local": f"Rrs_{nm}" for nm in c.wavelengths_prisma}).rename(columns={"cdom_443": "aCDOM_443", "cdom_675": "aCDOM_675", "nap_443": "aNAP_443", "nap_675": "aNAP_675", "ph_443": "aph_443", "ph_675": "aph_675",})

    # test_X, test_y for in situ vs in situ
    prisma_insitu = read_prisma_insitu(folder/"case_1_insitu_vs_insitu"/"prisma_insitu.csv")

    # test_X, test_y for ACOLITE scenarios
    prisma_acolite = pd.read_csv(folder/"case_2_insitu_vs_aco"/"prisma_aco.csv").rename(columns={f"aco_{nm}": f"Rrs_{nm}" for nm in c.wavelengths_prisma}).rename(columns=capitalise_iops)
    prisma_acolite_gen = prisma_acolite
    prisma_acolite_lk = prisma_acolite.copy()

    # test_X, test_y for L2 scenarios
    prisma_l2 = pd.read_csv(folder/"case_2_insitu_vs_l2"/"prisma_l2.csv").rename(columns={f"L2C_{nm}": f"Rrs_{nm}" for nm in c.wavelengths_prisma}).rename(columns=capitalise_iops)
    prisma_l2_gen = prisma_l2
    prisma_l2_lk = prisma_l2.copy()

    ### APPLY ROBUST SCALERS
    # Setup
    rrs_columns = _find_rrs_columns(gloria_resampled)
    gloria_scaler, combined_scaler = RobustScaler(), RobustScaler()

    # Train on training data
    gloria_resampled[rrs_columns] = gloria_scaler.fit_transform(gloria_resampled[rrs_columns])
    combined_insitu[rrs_columns] = combined_scaler.fit_transform(combined_insitu[rrs_columns])

    # Apply to test data
    prisma_insitu[rrs_columns] = gloria_scaler.transform(prisma_insitu[rrs_columns])  # case 1

    prisma_l2_gen[rrs_columns] = gloria_scaler.transform(prisma_l2_gen[rrs_columns])  # general case
    prisma_acolite_gen[rrs_columns] = gloria_scaler.transform(prisma_acolite_gen[rrs_columns])

    prisma_l2_lk[rrs_columns] = combined_scaler.transform(prisma_l2_lk[rrs_columns])  # local knowledge
    prisma_acolite_lk[rrs_columns] = combined_scaler.transform(prisma_acolite_lk[rrs_columns])

    ### ORGANISE TRAIN/TEST SETS
    general = DataScenario(c.prisma_gen, gloria_resampled, {c.prisma_insitu: prisma_insitu,
                                                            c.prisma_gen_L2: prisma_l2_gen,
                                                            c.prisma_gen_ACOLITE: prisma_acolite_gen}, X_scaler=gloria_scaler)

    local_knowledge = DataScenario(c.prisma_lk, combined_insitu, {c.prisma_lk_L2: prisma_l2_lk,
                                                                  c.prisma_lk_ACOLITE: prisma_acolite_lk}, X_scaler=combined_scaler)

    return general, local_knowledge


def extract_inputs_outputs(data: pd.DataFrame, *,
                           y_columns: Iterable[str]=c.iops) -> tuple[np.ndarray, np.ndarray]:
    """
    For a given DataFrame, extract the Rrs columns (X) and IOP columns (y).
    """
    rrs_columns = _find_rrs_columns(data)
    X = data[rrs_columns].values
    y = data[y_columns].values
    return X, y


### SCALING
def scale_y(y_train: np.ndarray, *y_other: Optional[Iterable[np.ndarray]],
            scaled_range: tuple[int]=(-1, 1)) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply log and minmax scaling to the input arrays, training the scaler on y_train only (per IOP).
    y_other can be any number (including 0) of other arrays.
    """
    # Apply log transformation to the target variables
    y_train_log = np.log(y_train)
    y_other_log = [np.log(y) for y in y_other]

    # Apply Min-Max scaling to log-transformed target variables
    scaler_y = MinMaxScaler(feature_range=scaled_range)
    y_train_scaled = scaler_y.fit_transform(y_train_log)
    y_other_scaled = [scaler_y.transform(y_log) for y_log in y_other_log]

    return scaler_y, y_train_scaled, *y_other_scaled


def inverse_scale_y(mean_scaled: np.ndarray, variance_scaled: np.ndarray, scaler_y: MinMaxScaler) -> tuple[np.ndarray, np.ndarray]:
    """
    Go back from log-minmax-scaled space to real units.
    """
    # Setup
    N = scaler_y.n_features_in_  # Number of predicted values
    original_shape = mean_scaled.shape

    # Convert from scaled space to log space: Means
    mean_scaled = mean_scaled.reshape(-1, N)
    mean_log = scaler_y.inverse_transform(mean_scaled)
    mean_log = mean_log.reshape(original_shape)

    # Convert from scaled space to log space: Variance
    scaling_factor = (scaler_y.data_max_ - scaler_y.data_min_) / 2
    variance_log = variance_scaled * (scaling_factor**2)  # Uncertainty propagation for linear equations

    # Convert from log space to the original space, i.e. actual IOPs in [m^-1]
    mean = np.exp(mean_log)  # Geometric mean / median
    variance = np.exp(2*mean_log + variance_log) * (np.exp(variance_log) - 1)  # Arithmetic variance

    return mean, variance
