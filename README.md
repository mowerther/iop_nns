# On the generalization of neural networks for hyperspectral remote sensing of absorption properties in optically complex waters (submitted to *Remote Sensing of Environment*)

**Abstract**

Machine learning models have steadily improved in estimating inherent optical properties (IOPs) from remote sensing observations. Yet, their generalization ability when applied to new water bodies, beyond those they were trained on, is not well understood. We present a novel approach for assessing model generalization across various scenarios, including interpolation within *in situ* observation datasets, extrapolation beyond the training scope, and application to hyperspectral observations from the PRecursore IperSpettrale della Missione Applicativa (PRISMA) satellite involving atmospheric correction. We evaluate five probabilistic neural networks (PNNs), including novel architectures like recurrent neural networks, for their ability to estimate absorption at 443 and 675 nm.
The median symmetric accuracy (MdSA) declines from ≥20% in interpolation scenarios to ≥50% in extrapolation scenarios, and ≥80% when applied to PRISMA satellite imagery. Including representative *in situ* observations in PRISMA applications improves accuracy by 10-15 percent points, highlighting the importance of regional knowledge. Uncertainty estimates exceed 40% across all scenarios, with models generally underconfident in their estimations. However, we observe better-calibrated uncertainties during extrapolation, indicating an inherent recognition of retrieval limitations. We introduce an uncertainty recalibration method using 10% of the dataset, which improves model reliability in 70% of PRISMA evaluations with minimal accuracy trade-offs. 
The uncertainty is predominantly aleatoric (inherent to the observations). Therefore, increasing the number of measurements from the same distribution does not enhance model accuracy. Similarly, selecting a different neural network architecture, trained on the same data, is unlikely to significantly improve retrieval accuracy. Instead, we propose that advancement in IOP estimation through neural networks lies in integrating the physical principles of IOPs into model architectures, thereby creating physics-informed neural networks.

## Scripts
[dataset_split.py](dataset_split.py) - Split the *in situ* dataset into training and test set (random split, within and out-of-distribution).

[train_nn.py](train_nn.py) - Train a PNN of choice (out of `bnn_dc`, `bnn_mcd`, `ens_nn`, `mdn`, `rnn`).

[plot_data.py](plot_data.py) - Generate the figures showing the training and test set IOP distributions.

[plot_calibration_example.py](plot_calibration_example.py) - Generate the figure explaining uncertainty calibration.

[analyze_estimates.py](analyze_estimates.py) - Generate figures relating to the PNN model outputs.
