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
data_path = Path("datasets_train_test/")
pred_path = Path("pnn_model_estimates/")
save_path = Path("manuscript_figures/")
supplementary_path = save_path/"supplementary/"


### UNITS
m1 = r"m$^{-1}$"
m2 = r"m$^{-2}$"


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
    label_2lines: Optional[str] = None

    def __post_init__(self):
        if self.label_2lines is None:
            self.label_2lines = self.label

    # Makes it possible to use the Parameter object as an index.
    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        return (self.name == other)

    def __repr__(self) -> str:
        return self.name


### NETWORKS
mdn = Parameter("mdn", "MDN", "#994455")
bnn_dropconnect = Parameter("bnn_dropcon", "BNN DC", "#997700")
bnn_mcd = Parameter("bnn_mcd", "BNN MCD", "#6699CC")
ensemble = Parameter("ens_nn", "ENN", "#EE99AA")
rnn = Parameter("rnn", "RNN", "#EECC66")

networks = [mdn, bnn_dropconnect, bnn_mcd, ensemble, rnn]
networks_fromkey = {p.name: p for p in networks}


### SPLIT TYPES
random_split = Parameter("random_split", "Random split", label_2lines="Random\nsplit")
wd = Parameter("wd_split", "Within-distribution split", label_2lines="Within-distribution\nsplit")
ood = Parameter("ood_split", "Out-of-distribution split", label_2lines="Out-of-distribution\nsplit")

splits = [random_split, wd, ood]
splits_fromkey = {p.name: p for p in splits}


### UNCERTAINTY TYPES
ale_var = Parameter("ale_var", f"Aleatoric variance [{m2}]", cmap_aleatoric_fraction.colors[-3], vmin=0)
ale_unc = Parameter("ale_unc", f"Aleatoric uncertainty [{m1}]", cmap_aleatoric_fraction.colors[-3], vmin=0)
ale_unc_pct = Parameter("ale_unc_pct", "Aleatoric uncertainty [%]", cmap_aleatoric_fraction.colors[-3], vmin=0, vmax=20)

epi_var = Parameter("epi_var", f"Epistemic variance [{m2}]", cmap_aleatoric_fraction.colors[2], vmin=0)
epi_unc = Parameter("epi_unc", f"Epistemic uncertainty [{m1}]", cmap_aleatoric_fraction.colors[2], vmin=0)
epi_unc_pct = Parameter("epi_unc_pct", "Epistemic uncertainty [%]", cmap_aleatoric_fraction.colors[2], vmin=0, vmax=20)

total_var = Parameter("total_var", f"Total variance [{m2}]", "black", vmin=0)
total_unc = Parameter("std_dev", f"Total uncertainty [{m1}]", "black", vmin=0)
total_unc_pct = Parameter("std_dev_pct", "Total uncertainty [%]", "black", vmin=0, vmax=100)

ale_frac = Parameter("ale_frac", "Aleatoric fraction", cmap=cmap_aleatoric_fraction, vmin=0, vmax=1)

uncertainties = [total_unc, ale_unc, epi_unc]
relative_uncertainties = [total_unc_pct, ale_unc_pct, epi_unc_pct]


### OTHER KEYS
y_true = "y_true"
y_pred = "y_pred"


### IOPs
_CDOM = r"\text{CDOM}"
_NAP = r"\text{NAP}"
_ph = r"\text{ph}"
aCDOM_443 = Parameter("aCDOM_443", f"$a_{_CDOM}$(443)", label_2lines=f"$a_{_CDOM}$\n(443)", color="darkgoldenrod")
aCDOM_675 = Parameter("aCDOM_675", f"$a_{_CDOM}$(675)", label_2lines=f"$a_{_CDOM}$\n(675)", color="darkgoldenrod")
aNAP_443 = Parameter("aNAP_443", f"$a_{_NAP}$(443)", label_2lines=f"$a_{_NAP}$\n(443)", color="chocolate")
aNAP_675 = Parameter("aNAP_675", f"$a_{_NAP}$(675)", label_2lines=f"$a_{_NAP}$\n(675)", color="chocolate")
aph_443 = Parameter("aph_443", f"$a_{_ph}$(443)", label_2lines=f"$a_{_ph}$\n(443)", color="darkgreen")
aph_675 = Parameter("aph_675", f"$a_{_ph}$(675)", label_2lines=f"$a_{_ph}$\n(675)", color="darkgreen")

iops = [aCDOM_443, aCDOM_675, aNAP_443, aNAP_675, aph_443, aph_675]


### METRICS
mdsa = Parameter("mdsa", "MDSA [%]", vmin=0)
sspb = Parameter("sspb", "SSPB [%]", symmetric=True)
r_squared = Parameter("r_squared", r"$R^2$", vmax=1)
sharpness = Parameter("sharpness", f"Sharpness [{m1}]", vmin=0)
coverage = Parameter("coverage", "Coverage [%]", vmin=0, vmax=100)
