# On the generalization of probabilistic neural networks for hyperspectral remote sensing of absorption properties in optically complex waters
This repository contains the Python code used in our paper in *Remote Sensing of Environment*, [DOI 10.1016/j.rse.2025.114820](https://doi.org/10.1016/j.rse.2025.114820).
The following sections describe the repository and codebase in detail.
Text *written in italics* is work-in-progress.
The [last section](#reproducing-the-paper) provides a summary containing just the information necessary to reproduce the paper.

### Abstract
<p align="justify">
Machine learning models have steadily improved in estimating inherent optical properties (IOPs) from remote sensing observations.
Yet, their generalization ability when applied to new water bodies, beyond those they were trained on, is not well understood.
We present a novel approach for assessing model generalization across various scenarios, including interpolation within <i>in situ</i> observation datasets, extrapolation beyond the training scope, and application to hyperspectral observations from the PRecursore IperSpettrale della Missione Applicativa (PRISMA) satellite involving atmospheric correction.
We evaluate five probabilistic neural networks (PNNs), including novel architectures like recurrent neural networks, for their ability to estimate absorption at 443 and 675 nm from hyperspectral reflectance.
The median symmetric accuracy (MdSA) worsens from ≥25% in interpolation scenarios to ≥50% in extrapolation scenarios, and reaches ≥80% when applied to PRISMA satellite imagery.
Across all scenarios, models produce uncertainty estimates exceeding 40%, often reflecting systematic underconfidence.
PNNs show better calibration during extrapolation, suggesting an intrinsic awareness of retrieval constraints.
To address this miscalibration, we introduce an uncertainty recalibration method that only withholds 10% of the training dataset, but improves model calibration in 86% of PRISMA evaluations with minimal accuracy trade-offs.
Resulting well-calibrated uncertainty estimates enable reliable uncertainty propagation for downstream applications.
IOP retrieval uncertainty is predominantly aleatoric (inherent to the observations).
Therefore, increasing the number of measurements from the same distribution or selecting a different neural network architecture trained on the same dataset does not enhance model accuracy.
Our findings indicate that we have reached a predictability limit in retrieving IOPs using purely data-driven approaches.
We therefore advocate embedding physical principles of IOPs into model architectures, creating physics-informed neural networks capable of surpassing current limitations.
</p>

## Overview

### `pnn` module
Most of the functionalities related to data handling, model training, and analysis have been refactored into the [`pnn`](pnn) module.
This module handles
constructing, training, testing, recalibrating, and applying the neural network models; 
analysing and visualising the model estimates; 
generating outputs for the paper.
A more detailed overview is provided in [its documentation](pnn/README.md).

### Scripts
The code used to (re-)produce the results in the paper is organised into multiple scripts within the top-level folder.
Detailed descriptions of the various scripts are provided in the following sections.
All scripts use `argparse`, meaning the help flag `-h` is available, e.g.:
```console
python train_nn.py -h
```
will return:
```console
usage: train_nn.py [-h] [-o OUTPUT_FOLDER] [-e ESTIMATES_FOLDER] [-p] [-c]
                   [-n N_MODELS]
                   pnn_type

Script for loading data and training a probabilistic neural network.
Trains N networks, evaluates them, and saves their outputs.
Selects the type of network from the first argument: [bnn_dc, bnn_mcd, ens_nn, mdn, rnn].
Note that model files and estimates are saved to default folders and will override existing files; custom locations can be specified (-o, -e).

Example:
    python train_nn.py bnn_mcd
    python train_nn.py bnn_mcd -p
    python train_nn.py bnn_mcd -c
    python train_nn.py bnn_mcd -pc -o path/to/models/ -e path/to/estimates/ -n 10

positional arguments:
  pnn_type              PNN architecture to use

options:
  -h, --help            show this help message and exit
  -d DATA_FOLDER, --data_folder DATA_FOLDER
                        Folder to load data from.
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        Folder to save models to.
  -e ESTIMATES_FOLDER, --estimates_folder ESTIMATES_FOLDER
                        Folder to save model estimates to.
  -p, --prisma          Use PRISMA data.
  -c, --recalibrate     Apply recalibration.
  -n N_MODELS, --n_models N_MODELS
                        Number of models to train per scenario (default: 25).
```


## *In situ* data
The core *in situ* datasets used in our study originate from [GLORIA](https://doi.org/10.1038/s41597-023-01973-y) and [SeaBASS](https://seabass.gsfc.nasa.gov/).
The combined *in situ* dataset is available from
[Zenodo: 10.5281/zenodo.14893798](https://doi.org/10.5281/zenodo.14893798).

The *in situ* dataset is located at [datasets_train_test/insitu_data.csv](datasets_train_test) by default.
Also available is the same dataset resampled to the PRISMA spectral bands, at [datasets_train_test/insitu_data_resampled.csv](datasets_train_test)


## Dataset splitting
A data file in CSV format with column headers can be split using the [dataset_split.py](dataset_split.py) script.
This script does not require installation of the wider `pnn` module, but can be used by itself.
[dataset_split.py](dataset_split.py) can also be safely imported if you want to re-use its functionality elsewhere.
Its dependencies are Numpy, Pandas, Scipy, and Scikit-learn.
The within-distribution and out-of-distribution splits are implemented through general functions called `system_data_split` and `objective`,
which can easily be built upon for new data split types,
by introducing new scoring functions.

The script is called as follows:
```console
python dataset_split.py path/to/data.csv
```

Several options are available, as shown below:
```console
python dataset_split.py -h

usage: dataset_split.py [-h] [-o OUTPUT_FOLDER] [-s SYSTEM_COLUMN]
                        [-t TIMEOUT] [-r RNG]
                        filename

Script for splitting a dataset using random, within-distribution, and out-of-distribution splits.
Data are split on one system column, provided with the `-s` flag (default: "lake_name").
(Dis)similarity scores are evaluated on multiple `summary_cols`, specified at the start of the script.
Please note that the script can slightly run past its `timeout`; this is not a bug.

Example:
    python dataset_split.py datasets_train_test/insitu_data.csv
    python dataset_split.py datasets_train_test/insitu_data.csv -o path/to/outputs/ -s site_name -t 10 -r 42

positional arguments:
  filename              File with data to split.

options:
  -h, --help            show this help message and exit
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        Folder to save files with splits to.
  -s SYSTEM_COLUMN, --system_column SYSTEM_COLUMN
                        Column with system names, on which to split the data.
  -t TIMEOUT, --timeout TIMEOUT
                        Maximum time [min] to spend on each split.
  -r RNG, --rng RNG     Seed for random number generator (RNG).
```

The dataset_split.py script will save the resulting dataframes to your chosen location (by default, the working directory) in 6 new CSV files:
(`"random_train_set.csv"`, `"random_test_set.csv"`, `"wd_train_set.csv"`, `"wd_test_set.csv"`, `"ood_train_set.csv"`, `"ood_test_set.csv"`)


### Loading split datasets
All other steps in the model training, estimation, and analysis for the *in situ* scenarios use the resulting split data files (random, within-distribution, out-of-distribution).
The split data files used in the paper are available from [Zenodo](https://doi.org/10.5281/zenodo.14893798).

These files are read using the [`pnn.data.read_insitu_data`](pnn/data.py#L82) function.
This function can load files from any folder on your computer; it will try to load the aforementioned 6 CSV files from the given folder.
By default, it uses the [datasets_train_test](datasets_train_test) folder within the repository folder; this setting can be changed at [`pnn.constants.insitu_data_path`](pnn/constants.py#L14).

Example:
```python
random_split, wd_split, ood_split = pnn.read_insitu_data()
```

The data are loaded into [`DataScenario`](pnn/data.py#L43) objects, which are dataclasses containing:
* the training scenario label – e.g. `random_split.train_scenario` is `pnn.constants.random_split`.
* the training data – e.g. `random_split.train_data` is a Pandas DataFrame with 1159 rows and 132 columns including Rrs and IOPs.
* a `dict` of testing scenarios and testing data – e.g. `random_split.test_scenarios_and_data` is a dict of one item, with key `pnn.constants.random_split` and value a Pandas DataFrame with 1160 rows and 132 columns including Rrs and IOPs.

The `DataScenario` object can be unpacked if wanted, e.g. `scenario_train, data_train, scenarios_and_data_test = random_split`.
This makes it easy to iterate over them, as in the [train_nn.py](train_nn.py) script:
```python
datascenarios = pnn.read_insitu_data()
for scenario_train, data_train, scenarios_and_data_test in datascenarios:
    ...
```


### Plotting data
The input data can be plotted using [plot_data.py](plot_data.py),
which generates figures showing the IOP distributions in the input data and train/test sets in each split scenario.
The figures will be saved to the [manuscript_figures](manuscript_figures) folder.

The script can be run without any arguments to use the default file locations:
```console
python plot_data.py
```

Alternative file locations can be provided with keyword arguments, as shown below:
```console
python plot_data.py -h

usage: plot_data.py [-h] [-i INSITU_FOLDER] [-p PRISMA_FOLDER]
                    [--system_column SYSTEM_COLUMN]

Script for loading and plotting input data.
Data are assumed to be in default locations, but different folders can be specified:
    in situ: use the -i flag
    PRISMA match-ups: use the -p flag

Example:
    python plot_data.py
    python plot_data.py -i path/to/insitudata/ -p path/to/prismadata/

options:
  -h, --help            show this help message and exit
  -i INSITU_FOLDER, --insitu_folder INSITU_FOLDER
                        Folder containing in situ data.
  -p PRISMA_FOLDER, --prisma_folder PRISMA_FOLDER
                        Folder containing PRISMA match-up data.
  --system_column SYSTEM_COLUMN
                        Column containing system names.
```


## PRISMA match-up data
*In situ* match-up data from PRISMA validation campaigns are used to evaluate the performance of PNN models in realistic scenarios.
The *in situ* match-up data are read using the [`pnn.data.read_prisma_insitu`](pnn/data.py#L111) function;
by default, this reads [datasets_train_test/prisma_insitu_data.csv](datasets_train_test).

Matching measurements from the PRISMA satellite,
atmospherically corrected using ACOLITE or the PRISMA L2C processor,
are also used.
By default, these files are located at [datasets_train_test/prisma_acolite.csv](datasets_train_test) and [datasets_train_test/prisma_l2c.csv](datasets_train_test), respectively.

The [`pnn.data.read_prisma_matchups`](pnn/data.py#L126) function is used to read PRISMA match-up data.
This function will return two [`DataScenario`](pnn/data.py#L43) objects,
the first representing the general case (train on the resampled *in situ* dataset; test on the PRISMA *in situ* data or the ACOLITE/L2C-processed satellite data),
the second representing the local knowledge case (train on the resampled *in situ* and PRISMA *in situ* match-up data together; test on the ACOLITE/L2C-processed satellite data).


## PNN training & application
Neural network models are trained using the [train_nn.py](train_nn.py) script.
This script uses the architectures [defined in the `pnn.nn`](pnn/README.md) submodule, currently:
* `bnn_dc` (Bayesian Neural Network with DropConnect)
* `bnn_mcd` (Bayesian Neural Network with Monte Carlo Dropout)
* `ens_nn` (Ensemble Neural Network)
* `mdn` (Mixture Density Network)
* `rnn` (Recurrent Neural Network)

These architectures use the same backbone, namely the [`BasePNN` class](pnn/nn/pnn_base.py), ensuring consistency and making it very simple to add new architectures in the future.
The base class and its descendants are all implemented using TensorFlow/Keras.

To train one of these architectures, call the `train_nn.py` script with the architecture abbreviation as a command-line argument, e.g. to train a BNN-DC:
```console
python train_nn.py bnn_dc
```

By default, the script will [load the *in situ* data](#loading-split-datasets) and train/evaluate on its corresponding scenarios in sequence.
To use the PRISMA data scenarios (general or local knowledge), use the `-p` flag, e.g.:
```console
python train_nn.py bnn_dc -p
```

To train models with recalibration, simply add the `-c` flag, e.g.:
```console
python train_nn.py bnn_dc -c
```
or
```console
python train_nn.py bnn_dc -pc
```

Lastly, you can specify the number of models to train/evaluate for each scenario using the `-n` argument;
the default value is 25.
Say you want to train only 5 models, use:
```console
python train_nn.py bnn_dc -n 5
```


### Output format
Trained models are saved to the [`pnn.model_path`](pnn/constants.py#L18) constant;
by default, this is the [pnn_tf_models folder](pnn_tf_models).
Models are saved in ZIP files containing TensorFlow/Keras (`.keras`) files, re-scaling functions for X and y, and recalibration functions (where appropriate).
A saved model can be loaded using the [`.load`](pnn/nn/pnn_base.py#L203) function of the relevant class, e.g. `BNN_DC.load` or `RNN.load`.
The ENS-NN (ensemble neural network) is saved as a ZIP file containing its constituent networks within the general ZIP file, but otherwise works the same;
the [`Ensemble.save`](pnn/nn/ens.py#L98) and [`Ensemble.load`](pnn/nn/ens.py#L121) functions will automatically take care of filename extensions.
Recalibrated models are saved and loaded with their corresponding recalibration functions.

In addition to the models, [train_nn.py](train_nn.py) also saves the IOP estimates from each model in each scenario to file.
These are saved into the [`model_estimates_path`](pnn/constants.py#L18) variable,
which defaults to the [pnn_model_estimates](pnn_model_estimates/) folder.
This folder will contain N subfolders corresponding to the N trained model instances (25 by default, see above),
each containing a CSV file for every combination of model architecture and scenario,
tabulating the reference and estimated values and uncertainties (total / aleatoric / epistemic) for every point in the testing dataset.
The main folder also contains a CSV file for each combination of model architecture and scenario tabulating the performance metrics
(MdSA, SSPB, MAD, MAPE, R², coverage, miscalibration area)
for each of the N model instances.


### Application to PRISMA scenes
Trained models can be applied to PRISMA scenes using the [`apply_to_prisma.py`](apply_to_prisma.py) script.
This script takes all atmospherically corrected (L2C or ACOLITE; specified with the `-a` flag) NetCDF4 (`.nc`) files in a given folder ([`prisma_map_data`](prisma_map_data) by default) and applies a given model type (`bnn_mcd`, `bnn_dc`, etc.) to each.
**Note that there is currently a bug in the implementation causing L2C and ACOLITE to be switched around.**
By default, the average-performing instance of that architecture is found in [`pnn_tf_models`](pnn_tf_models);
a custom model file can be used through the `-m` flag.
The script takes the following arguments:
```console
python apply_to_prisma.py -h

usage: apply_to_prisma.py [-h] [-f FOLDER] [-a] [-c] [-m MODEL_FILE] pnn_type

Script for loading PRISMA scenes and applying PNN estimation pixel-wise.
- Loads (atmospherically corrected) reflectance data.
- Generates a water mask and RGB/greyscale background image.
- Finds and applies the average-performing model (by median MdSA; custom model can be used instead).

Data are loaded from pnn.c.map_data_path by default, but a custom folder can be supplied using the -f flag (e.g. `python apply_to_prisma.py -f /path/to/my/folder`).
Please note that figures will still be saved to the same location.
The average-performing model is used by default, but a specific model file can be used with the -m flag.

Recalibrated models are not currently supported.

Example:
    python apply_to_prisma.py bnn_mcd
    python apply_to_prisma.py bnn_mcd -a -m pnn_tf_models/0/ens_nn_ood_split.zip

positional arguments:
  pnn_type              PNN architecture to use

options:
  -h, --help            show this help message and exit
  -f FOLDER, --folder FOLDER
                        folder to load data from
  -a, --acolite         use acolite data (if False: use L2C)
  -c, --recalibrate     use recalibrated model
  -m MODEL_FILE, --model_file MODEL_FILE
                        specific file to load model from
```

A water mask will be generated using a normalised differential water index (NDWI).
If available, level-1 HDF5 (`.h5`) data for the same scene will be used to generate a greyscale background, using an approximated luminance function.
The script will save the results (IOP estimates, water mask, greyscale background) to a new NetCDF4 file so they can be re-used without having to re-run the analysis.
Please note that due to the number of pixels (up to a million), PNN application can be very slow, especially for RNNs.


## Analysis

### Match-up analysis
IOP estimates from trained models are analysed using the [analyze_estimates.py](analyze_estimates.py) script.
This script reads the IOP reference values and model estimates [generated by train_nn.py](#pnn-training--application)
and produces figures and statistics describing the performance of the models,
as in the paper.

```console
python analyze_estimates -h

usage: analyze_estimates.py [-h] [-f FOLDER] [-p] [-c]

Script for loading PNN outputs and generating plots.
First loads and analyzes the model metrics from all runs combined.
Next finds and analyzes the average-performing model (by median MdSA) for each network-scenario combination.

Data are loaded from pnn.c.model_estimates_path by default, but a custom folder can be supplied using the -f flag (e.g. `python analyze_estimates.py -f /path/to/my/folder`).
Please note that figures will still be saved to the same location.

To plot the PRISMA data, use the -p flag.
To include recalibrated data, use the -c flag.

options:
  -h, --help            show this help message and exit
  -f FOLDER, --folder FOLDER
                        folder to load data from
  -p, --prisma          use PRISMA data
  -c, --recal           use recalibrated data
```

The data are read into a Pandas DataFrame with a column for each IOP and a four-level index:
- `category`: reference value, estimated value, estimated variance, etc.
- `scenario`: random split, within-distribution, out-of-distribution, etc.
- `network`: BNN-DC, BNN-MCD, MDN, etc.
- `instance`: Match-ups in sequential order (not to be confused with model instance number!).

```python
                                       aph_443    aph_675  ...   aNAP_443   aNAP_675
category scenario  network instance                        ...                      
ale_frac ood_split bnn_dc  0         96.442658  96.207698  ...  94.889745  97.520820
                           1         97.633417  97.693073  ...  96.486127  98.700787
                           2         96.781594  96.121351  ...  95.950880  97.570668
                           3         98.706212  98.817885  ...  97.680038  99.422378
                           4         96.385136  95.717353  ...  95.375517  97.243007
                                       ...        ...  ...        ...        ...
y_true   wd_split  rnn     1028       0.504809   0.307788  ...   0.420340   0.090161
                           1029       0.372461   0.231302  ...   0.414890   0.086045
                           1030       0.429146   0.261521  ...   0.422664   0.097788
                           1031       0.457969   0.284133  ...   0.512828   0.121846
                           1032       0.394617   0.253642  ...   0.488469   0.121405
```

### PRISMA scenes
[Processed PRISMA scenes](#application-to-prisma-scenes) can be plotted using the [plot_prisma_scenes.py](plot_prisma_scenes.py) script.
This script will load scenes from a specified folder ([`pnn.c.map_output_path`](pnn/constants.py#L19) by default),
as well as match-up pixels,
and plot them on a map with the appropriate water mask and land background (if available).

```console
python plot_prisma_scenes.py -h

usage: plot_prisma_scenes.py [-h] [-f FOLDER]

Script for loading processed PRISMA scenes and creating specific figures.
- Loads IOP estimate scenes.
- Creates specified figures.

Data are loaded from pnn.c.map_output_path by default, but a custom folder can be supplied using the -f flag (e.g. `python plot_prisma_scenes.py -f /path/to/my/folder`).
Please note that figures will still be saved to the same location.

Example:
    python plot_prisma_scenes.py bnn_mcd

options:
  -h, --help            show this help message and exit
  -f FOLDER, --folder FOLDER
                        folder to load processed scenes from
```

Note that this script is much more hard-coded than the others, having been tailored for the figures in our paper.
It can be easily adapted for a different dataset.
Generic functions for plotting PRISMA scene outputs are available in the `pnn` module:
* [`pnn.maps.plot_Rrs`](pnn/maps.py#L322)
* [`pnn.maps.plot_IOP_single`](pnn/maps.py#L363)
* [`pnn.maps.plot_IOP_all`](pnn/maps.py#L414)
* [`pnn.maps.plot_Rrs_and_IOP`](pnn/maps.py#L440)
* [`pnn.maps.plot_Rrs_and_IOP_overview`](pnn/maps.py#L467)

## Reproducing the paper

### Setup
1. [Download the input data](#in-situ-data) from [Zenodo](https://doi.org/10.5281/zenodo.14893798).
2. Clone or download this repository, navigate into the folder on your computer, and install the `pnn` module using `pip install .` on the command line.

### Dataset splitting
3. [Split the dataset](#dataset-splitting) using [dataset_split.py](dataset_split.py) as follows: `python dataset_split.py datasets_train_test/insitu_data.csv`.
4. *(If desired)* [Plot the input data and split datasets](#plotting-data) using [plot_data.py](plot_data.py) as follows: `python plot_data.py`.

### Model training and application
5. [Train your model instances](#pnn-training--application) using [train_nn.py](train_nn.py) as follows (all separate commands, in any order):  
    a. *In situ*:
   * `python train_nn.py bnn_mcd`
   * `python train_nn.py bnn_dc`
   * `python train_nn.py ens_nn`
   * `python train_nn.py mdn`
   * `python train_nn.py rnn`

    b. *In situ* with recalibration:
   * `python train_nn.py bnn_mcd -c`
   * `python train_nn.py bnn_dc -c`
   * `python train_nn.py ens_nn -c`
   * `python train_nn.py mdn -c`
   * `python train_nn.py rnn -c`

    c. PRISMA:
   * `python train_nn.py bnn_mcd -p`
   * `python train_nn.py bnn_dc -p`
   * `python train_nn.py ens_nn -p`
   * `python train_nn.py mdn -p`
   * `python train_nn.py rnn -p`

    d. PRISMA with recalibration:
   * `python train_nn.py bnn_mcd -pc`
   * `python train_nn.py bnn_dc -pc`
   * `python train_nn.py ens_nn -pc`
   * `python train_nn.py mdn -pc`
   * `python train_nn.py rnn -pc`

6. *(If desired)*: [Apply the trained models to PRISMA data](#application-to-prisma-scenes) using the [apply_to_prisma.py](apply_to_prisma.py) script as follows:  
    a. L2C:
   * `python apply_to_prisma.py bnn_mcd`
   * `python apply_to_prisma.py bnn_dc`
   * `python apply_to_prisma.py ens_nn`
   * `python apply_to_prisma.py mdn`
   * `python apply_to_prisma.py rnn`

    b. L2C with recalibration:
   * `python apply_to_prisma.py bnn_mcd -c`
   * `python apply_to_prisma.py bnn_dc -c`
   * `python apply_to_prisma.py ens_nn -c`
   * `python apply_to_prisma.py mdn -c`
   * `python apply_to_prisma.py rnn -c`

    c. ACOLITE:
   * `python apply_to_prisma.py bnn_mcd -a`
   * `python apply_to_prisma.py bnn_dc -a`
   * `python apply_to_prisma.py ens_nn -a`
   * `python apply_to_prisma.py mdn -a`
   * `python apply_to_prisma.py rnn -a`

    d. ACOLITE with recalibration:
   * `python apply_to_prisma.py bnn_mcd -ac`
   * `python apply_to_prisma.py bnn_dc -ac`
   * `python apply_to_prisma.py ens_nn -ac`
   * `python apply_to_prisma.py mdn -ac`
   * `python apply_to_prisma.py rnn -ac`

### Analysis
7. *(If desired)*: Generate Figure 4 (recalibration example) using the [plot_calibration_example.py](plot_calibration_example.py) script as follows: `python plot_calibration_example.py`.
8. [Perform the match-up analysis](#match-up-analysis) using the [analyze_estimates.py](analyze_estimates.py) script as follows:  
    a. *In situ*: `python analyze_estimates.py`  
    b. *In situ* with recalibration: `python analyze_estimates.py -c`  
    c. PRISMA match-ups: `python analyze_estimates.py -p`  
    d. PRISMA match-ups with recalibration: `python analyze_estimates.py -pc`  
9. *(If desired)*: [Generate the PRISMA scene figures](#prisma-scenes) using the [plot_prisma_scenes.py](plot_prisma_scenes.py) script as follows: `python plot_prisma_scenes.py`.

Note that some additional manual analysis of the DataFrames loaded / generated in step 8 (analyze_estimates.py) is necessary to reproduce all of the statistics used in the paper.
