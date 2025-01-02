# On the generalization of probabilistic neural networks for hyperspectral remote sensing of absorption properties in optically complex waters

This repository contains the Python code that was used in our paper, submitted to *Remote Sensing of Environment*.
The following sections describe the repository and codebase in detail. The [last section](#reproducing-the-paper) provides a summary containing just the information necessary to reproduce the paper.

**Abstract**    
Machine learning models have steadily improved in estimating inherent optical properties (IOPs) from remote sensing observations. Yet, their generalization ability when applied to new water bodies, beyond those they were trained on, is not well understood. We present a novel approach for assessing model generalization across various scenarios, including interpolation within *in situ* observation datasets, extrapolation beyond the training scope, and application to hyperspectral observations from the PRecursore IperSpettrale della Missione Applicativa (PRISMA) satellite involving atmospheric correction. We evaluate five probabilistic neural networks (PNNs), including novel architectures like recurrent neural networks, for their ability to estimate absorption at 443 and 675 nm.
The median symmetric accuracy (MdSA) declines from ≥20% in interpolation scenarios to ≥50% in extrapolation scenarios, and ≥80% when applied to PRISMA satellite imagery. Including representative *in situ* observations in PRISMA applications improves accuracy by 10-15 percent points, highlighting the importance of regional knowledge. Uncertainty estimates exceed 40% across all scenarios, with models generally underconfident in their estimations. However, we observe better-calibrated uncertainties during extrapolation, indicating an inherent recognition of retrieval limitations. We introduce an uncertainty recalibration method using 10% of the dataset, which improves model reliability in 70% of PRISMA evaluations with minimal accuracy trade-offs. 
The uncertainty is predominantly aleatoric (inherent to the observations). Therefore, increasing the number of measurements from the same distribution does not enhance model accuracy. Similarly, selecting a different neural network architecture, trained on the same data, is unlikely to significantly improve retrieval accuracy. Instead, we propose that advancement in IOP estimation through neural networks lies in integrating the physical principles of IOPs into model architectures, thereby creating physics-informed neural networks.


## _In situ_ data
The core _in situ_ datasets used in our study originate from [GLORIA](https://doi.org/10.1038/s41597-023-01973-y) and [SeaBASS](https://seabass.gsfc.nasa.gov/).
These datasets are not currently hosted within this repository for licensing reasons; we aim to make them available so the study can be reproduced.

_Splitting: data format, data location, data handling._

The resulting split data files (random, within-distribution, out-of-distribution) are read using the [`pnn.data.read_scenario123_data`](pnn/data.py#L59) function.
This function can load files from any folder on your computer; it will try to load 6 CSV files (`"random_df_train_org.csv"`, `"random_df_test_org.csv"`, `"wd_train_set_org.csv"`, `"wd_test_set_org.csv"`, `"ood_train_set_2.csv"`, `"ood_test_set_2.csv"`) from the given folder.
By default, it uses the [datasets_train_test](datasets_train_test) folder within the repository folder; this setting can be changed at [`pnn.constants.data_path`](pnn/constants.py#L14).

## PRISMA data


## Outputs
_Folders, file types, structure._

## Dataset splitting
[dataset_split.py](dataset_split.py) - Split an *in situ* dataset into training and test set (random split, within-distribution, and out-of-distribution).

[plot_data.py](plot_data.py) - Generate figures showing the training and test set IOP distributions.

## Model training
[train_nn.py](train_nn.py) - Train a PNN of choice (out of `bnn_dc`, `bnn_mcd`, `ens_nn`, `mdn`, `rnn`).

## Model recalibration
[plot_calibration_example.py](plot_calibration_example.py) - Generate a figure explaining uncertainty calibration.

[train_nn.py](train_nn.py) - Train a PNN of choice (out of `bnn_dc`, `bnn_mcd`, `ens_nn`, `mdn`, `rnn`).

## Analysis
[analyze_estimates.py](analyze_estimates.py) - Analyze PNN model outputs to generate figures and statistics.

## Reproducing the paper
This section will explain step-by-step how the paper can be reproduced.

Run [dataset_split.py](dataset_split.py) as follows:
To do.

Run [train_nn.py](train_nn.py) as follows (with/without flags):
To do.
