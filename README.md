# On the generalization of neural networks for hyperspectral remote sensing of inherent optical properties in complex waters

This study examines five state-of-the-art PNNs for estimating absorption IOPs using semi-global _in situ_ datasets and match-ups with PRISMA.
Each observation included remote-sensing reflectance $R_\text{rs}$ and triplet measurements of phytoplankton absorption ($a_\text{ph}$), colored dissolved organic matter ($a_\text{CDOM}$), and non-algal particles ($a_\text{NAP}$) at 443 and 675 nm.
We evaluated the generalization ability of PNNs across optimal conditions with shared knowledge to increasingly challenging situations, culminating in the application to unknown waters using PRISMA data.

## Scripts
[train_nn.py](train_nn.py) - Train a PNN of choice (out of `bnn_dc`, `bnn_mcd`, `ens_nn`, `mdn`, `rnn`).

[plot_data.py](plot_data.py) - Generate the figures showing the input data distribution.

[plot_calibration_example.py](plot_calibration_example.py) - Generate the figure explaining uncertainty calibration.

[analyze_predictions.py](analyze_predictions.py) - Generate figures relating to the PNN model outputs.
