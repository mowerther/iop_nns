"""
Some constants to be used elsewhere.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
from cmcrameri.cm import managua


### FILENAMES
pred_path = Path("pnn_model_estimates/")
save_path = Path("manuscript_figures/")
supplementary_path = save_path/"supplementary/"


### UNITS
m1 = r"m$^{-1}$"


### PLOTTING
cmap_uniform = plt.cm.cividis.resampled(10)
cmap_aleatoric_fraction = managua.resampled(10)


### PARAMETERS
@dataclass
class Parameter:
    name: str
    label: str
    color: Optional[str] = "black"
    cmap: Optional[Colormap] = field(default_factory=lambda: cmap_uniform)
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    symmetric: bool = False


### NETWORKS
mdn = Parameter("mdn", "MDN", "#994455")
bnn_dropconnect = Parameter("bnn_dropconnect", "BNN DC", "#997700")
bnn_mcd = Parameter("bnn_mcd", "BNN MCD", "#6699CC")
ensemble = Parameter("ensemble", "ENN", "#EE99AA")
rnn = Parameter("rnn", "RNN", "#EECC66")

networks = [mdn, bnn_dropconnect, bnn_mcd, ensemble, rnn]
networks_fromkey = {p.name: p for p in networks}


### SPLIT TYPES
random_split = Parameter("random_split", "Random split")
wd = Parameter("wd", "Within-distribution split")
ood = Parameter("ood", "Out-of-distribution split")

splits = [random_split, wd, ood]
splits_fromkey = {p.name: p for p in splits}


### UNCERTAINTY TYPES
ale_unc = Parameter("ale_unc", f"Aleatoric uncertainty [{m1}]", cmap_aleatoric_fraction.colors[-3], vmin=0)
ale_unc_pct = Parameter("ale_unc_pct", "Aleatoric uncertainty [%]", cmap_aleatoric_fraction.colors[-3], vmin=0, vmax=20)
epi_unc = Parameter("epi_unc", f"Epistemic uncertainty [{m1}]", cmap_aleatoric_fraction.colors[2], vmin=0)
epi_unc_pct = Parameter("epi_unc_pct", "Epistemic uncertainty [%]", cmap_aleatoric_fraction.colors[2], vmin=0, vmax=20)
pred_std = Parameter("pred_std", f"Total uncertainty [{m1}]", "black", vmin=0)
pred_std_pct = Parameter("total_unc_pct", "Total uncertainty [%]", "black", vmin=0, vmax=20)
ale_frac = Parameter("ale_frac", "Aleatoric fraction", cmap=cmap_aleatoric_fraction, vmin=0, vmax=1)

uncertainties = [pred_std, ale_unc, epi_unc]
relative_uncertainties = [pred_std_pct, ale_unc_pct, epi_unc_pct]


### IOPs
_CDOM = r"\text{CDOM}"
_NAP = r"\text{NAP}"
_ph = r"\text{ph}"
aCDOM_443 = Parameter("aCDOM_443", f"$a_{_CDOM}$(443)")
aCDOM_675 = Parameter("aCDOM_675", f"$a_{_CDOM}$(675)")
aNAP_443 = Parameter("aNAP_443", f"$a_{_NAP}$(443)")
aNAP_675 = Parameter("aNAP_675", f"$a_{_NAP}$(675)")
aph_443 = Parameter("aph_443", f"$a_{_ph}$(443)")
aph_675 = Parameter("aph_675", f"$a_{_ph}$(675)")

iops = [aCDOM_443, aCDOM_675, aNAP_443, aNAP_675, aph_443, aph_675]
iops_main = [aCDOM_443, aCDOM_675, aph_443, aph_675]


### METRICS
mdsa = Parameter("mdsa", "MDSA [%]", vmin=0)
sspb = Parameter("sspb", "SSPB [%]", symmetric=True)
r_squared = Parameter("r_squared", r"$R^2$", vmax=1)
sharpness = Parameter("sharpness", f"Sharpness [{m1}]", vmin=0)
coverage = Parameter("coverage", "Coverage [%]", vmin=0, vmax=100)
