"""
Some constants to be used everywhere.
"""
from pathlib import Path


# Filenames
pred_path = Path("pnn_model_estimates/")
save_path = Path("manuscript_figures/")


# Variables
iops = ("aCDOM_443", "aCDOM_675", "aNAP_443", "aNAP_675", "aph_443", "aph_675")
network_types = ("mdn", "bnn_dropconnect", "bnn_mcd", "ensemble", "rnn")
split_types = ("wd", "ood", "random_split")
