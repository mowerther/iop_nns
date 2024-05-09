"""
Some constants to be used everywhere.
"""
from pathlib import Path


# Filenames
pred_path = Path("pnn_model_estimates/")
save_path = Path("manuscript_figures/")


# Variables
iops = {"aCDOM_443": r"a$_\text{CDOM}$(443)",
        "aCDOM_675": r"a$_\text{CDOM}$(675)",
        "aNAP_443": r"a$_\text{NAP}$(443)",
        "aNAP_675": r"a$_\text{NAP}$(675)",
        "aph_443": r"a$_\text{ph}$(443)",
        "aph_675": r"a$_\text{ph}$(675)",}

iops_main = {key: value for key, value in iops.items() if "NAP" not in key}

network_types = {"mdn": "MDN",
                 "bnn_dropconnect": "BNN DC",
                 "bnn_mcd": "BNN MCD",
                 "ensemble": "ENS NN",
                 "rnn": "RNN",}

split_types = {"random_split": "Random split",
               "wd": "Within-distribution split",
               "ood": "Out-of-distribution split",}

uncertainty_types = {"ale_unc_pct": "Aleatoric",
                     "epi_unc_pct": "Epistemic",
                     "total_unc_pct": "Total",}

uncertainty_colors = {"ale_unc_pct": "blue",
                      "epi_unc_pct": "orange",
                      "total_unc_pct": "black",}
