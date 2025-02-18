# On the generalization of probabilistic neural networks for hyperspectral remote sensing of absorption properties in optically complex waters

This repository contains the Python code used in our paper, submitted to *Remote Sensing of Environment*.
The following sections describe the repository and codebase in detail.
Text *written in italics* is work-in-progress.
The [last section](#reproducing-the-paper) provides a summary containing just the information necessary to reproduce the paper.

**Abstract**    
Machine learning models have steadily improved in estimating inherent optical properties (IOPs) from remote sensing observations. Yet, their generalization ability when applied to new water bodies, beyond those they were trained on, is not well understood. We present a novel approach for assessing model generalization across various scenarios, including interpolation within *in situ* observation datasets, extrapolation beyond the training scope, and application to hyperspectral observations from the PRecursore IperSpettrale della Missione Applicativa (PRISMA) satellite involving atmospheric correction. We evaluate five probabilistic neural networks (PNNs), including novel architectures like recurrent neural networks, for their ability to estimate absorption at 443 and 675 nm.
The median symmetric accuracy (MdSA) declines from ≥20% in interpolation scenarios to ≥50% in extrapolation scenarios, and ≥80% when applied to PRISMA satellite imagery. Including representative *in situ* observations in PRISMA applications improves accuracy by 10-15 percent points, highlighting the importance of regional knowledge. Uncertainty estimates exceed 40% across all scenarios, with models generally underconfident in their estimations. However, we observe better-calibrated uncertainties during extrapolation, indicating an inherent recognition of retrieval limitations. We introduce an uncertainty recalibration method using 10% of the dataset, which improves model reliability in 70% of PRISMA evaluations with minimal accuracy trade-offs. 
The uncertainty is predominantly aleatoric (inherent to the observations). Therefore, increasing the number of measurements from the same distribution does not enhance model accuracy. Similarly, selecting a different neural network architecture, trained on the same data, is unlikely to significantly improve retrieval accuracy. Instead, we propose that advancement in IOP estimation through neural networks lies in integrating the physical principles of IOPs into model architectures, thereby creating physics-informed neural networks.

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
These datasets are not currently hosted within this repository for licensing reasons; we aim to make them available so the study can be reproduced.

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
These files are read using the [`pnn.data.read_insitu_data`](pnn/data.py#L82) function.
This function can load files from any folder on your computer; it will try to load the aforementioned 6 CSV files from the given folder.
By default, it uses the [datasets_train_test](datasets_train_test) folder within the repository folder; this setting can be changed at [`pnn.constants.insitu_data_path`](pnn/constants.py#L14).

Example:
```python
train_data, test_data = pnn.read_insitu_data()
```

*Explain data format in Python.*

### Plotting data
[plot_data.py](plot_data.py) - Generates figures showing the IOP distributions in the input data and train/test sets in each split scenario.
as follows,
which will save the resulting figures to
[manuscript_figures/full_dataset.pdf](manuscript_figures)
and
[manuscript_figures/scenarios.pdf](manuscript_figures).
```console
python plot_data.py
```


## PRISMA data

### Match-up data


### Scenes


## Model training
Neural network models are trained using the [train_nn.py](train_nn.py) script.
This script uses the architectures [defined in the `pnn.nn`](pnn/README.md) submodule, currently
`bnn_dc` (Bayesian Neural Network with DropConnect),
`bnn_mcd` (Bayesian Neural Network with Monte Carlo Dropout),
`ens_nn` (Ensemble Neural Network),
`mdn` (Mixture Density Network),
and
`rnn` (Recurrent Neural Network).
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


## Analysis
*[analyze_estimates.py](analyze_estimates.py) - Analyze PNN model outputs to generate figures and statistics.*

*[apply_to_prisma.py](apply_to_prisma.py) - Apply PNNs to PRISMA scenes, plot the results.*


## Reproducing the paper

### Setup
*Run [dataset_split.py](dataset_split.py) as follows:*

*Run [train_nn.py](train_nn.py) as follows (with/without flags):*


### Figures
Figures 1 and 2 were generated using [plot_data.py](plot_data.py), with no command-line arguments.

Figure 3 was made by hand.

Figure 4 was generated using [plot_calibration_example.py](plot_calibration_example.py), with no command-line arguments.

*Run [analyze_estimates.py](analyze_estimates.py) as follows (with/without flags):*

*Run [apply_to_prisma.py](apply_to_prisma.py) as follows (with/without flags):*
