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
prisma_path = Path("prisma_subscenarios/")
model_path = Path("pnn_tf_models/")
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
bnn_dc = Parameter("bnn_dc", "BNN-DC", "#997700")
bnn_mcd = Parameter("bnn_mcd", "BNN-MCD", "#6699CC")
ensemble = Parameter("ens_nn", "ENS-NN", "#EE99AA")
rnn = Parameter("rnn", "RNN", "#EECC66")
mdn = Parameter("mdn", "MDN", "#994455")

networks = [bnn_mcd, bnn_dc, mdn, ensemble, rnn]
networks_fromkey = {p.name: p for p in networks}


### SCENARIOS
# Scenarios 1, 2, 3
random_split = Parameter("random_split", "Random split", label_2lines="Random\nsplit")
wd = Parameter("wd_split", "Within-distribution split", label_2lines="Within-distribution\nsplit")
ood = Parameter("ood_split", "Out-of-distribution split", label_2lines="Out-of-distribution\nsplit")

scenarios_123 = [random_split, wd, ood]
scenarios_123_fromkey = {p.name: p for p in scenarios_123}

wavelengths_123 = list(range(400, 701, 5))

# PRISMA scenarios 1, 2a, 2b, 3a, 3b
_insitu = r"$\it{in situ}$"
prisma_insitu = Parameter("prisma1", f"Case 1: {_insitu} vs. {_insitu}", label_2lines="Case 1\n{_insitu} vs. {_insitu}")
prisma_insitu_ACOLITE = Parameter("prisma2a", f"Case 2a: {_insitu} vs. ACOLITE", label_2lines="Case 2a\n{_insitu} vs. ACOLITE")
prisma_insitu_L2 = Parameter("prisma2b", f"Case 2b: combined {_insitu} vs. L2", label_2lines="Case 2b\n{_insitu} vs. L2")
prisma_gloria_ACOLITE = Parameter("prisma3a", f"Case 3a: {_insitu} vs. ACOLITE", label_2lines="Case 3a\ncombined {_insitu} vs. ACOLITE")
prisma_gloria_L2 = Parameter("prisma3b", f"Case 3b: combined {_insitu} vs. L2", label_2lines="Case 3b\ncombined {_insitu} vs. L2")

scenarios_prisma = [prisma_insitu, prisma_insitu_ACOLITE, prisma_insitu_L2, prisma_gloria_ACOLITE, prisma_gloria_L2]
scenarios_prisma_fromkey = {p.name: p for p in scenarios_prisma}

wavelengths_prisma = [406, 415, 423, 431, 438, 446, 453, 460, 468, 475, 482, 489, 497, 504, 512, 519, 527, 535, 542, 550, 559, 567, 575, 583, 592, 601, 609, 618, 627, 636, 645, 655, 664, 674, 684, 694]


### UNCERTAINTY TYPES
ale_var = Parameter("ale_var", f"Aleatoric variance [{m2}]", cmap_aleatoric_fraction.colors[-3], vmin=0)
ale_unc = Parameter("ale_unc", f"Aleatoric uncertainty [{m1}]", cmap_aleatoric_fraction.colors[-3], vmin=0)
ale_unc_pct = Parameter("ale_unc_pct", "Aleatoric uncertainty [%]", cmap_aleatoric_fraction.colors[-3], vmin=0, vmax=100)

epi_var = Parameter("epi_var", f"Epistemic variance [{m2}]", cmap_aleatoric_fraction.colors[2], vmin=0)
epi_unc = Parameter("epi_unc", f"Epistemic uncertainty [{m1}]", cmap_aleatoric_fraction.colors[2], vmin=0)
epi_unc_pct = Parameter("epi_unc_pct", "Epistemic uncertainty [%]", cmap_aleatoric_fraction.colors[2], vmin=0, vmax=100)

total_var = Parameter("total_var", f"Total variance [{m2}]", "black", vmin=0)
total_unc = Parameter("total_unc", f"Total uncertainty [{m1}]", "black", vmin=0)
total_unc_pct = Parameter("total_unc_pct", "Total uncertainty [%]", "black", vmin=0, vmax=200)

ale_frac = Parameter("ale_frac", "Aleatoric fraction [%]", cmap=cmap_aleatoric_fraction, vmin=0, vmax=100)

variances = [total_var, ale_var, epi_var]
uncertainties = [total_unc, ale_unc, epi_unc]
relative_uncertainties = [total_unc_pct, ale_unc_pct, epi_unc_pct]


### OTHER KEYS
y_true = "y_true"
y_pred = "y_pred"


### IOPs
_ph = r"\text{ph}"
_CDOM = r"\text{CDOM}"
_NAP = r"\text{NAP}"
aph_443 = Parameter("aph_443", f"$a_{_ph}$(443)", label_2lines=f"$a_{_ph}$\n(443)", color="darkgreen")
aph_675 = Parameter("aph_675", f"$a_{_ph}$(675)", label_2lines=f"$a_{_ph}$\n(675)", color="darkgreen")
aCDOM_443 = Parameter("aCDOM_443", f"$a_{_CDOM}$(443)", label_2lines=f"$a_{_CDOM}$\n(443)", color="darkgoldenrod")
aCDOM_675 = Parameter("aCDOM_675", f"$a_{_CDOM}$(675)", label_2lines=f"$a_{_CDOM}$\n(675)", color="darkgoldenrod")
aNAP_443 = Parameter("aNAP_443", f"$a_{_NAP}$(443)", label_2lines=f"$a_{_NAP}$\n(443)", color="saddlebrown")
aNAP_675 = Parameter("aNAP_675", f"$a_{_NAP}$(675)", label_2lines=f"$a_{_NAP}$\n(675)", color="saddlebrown")

iops = [aph_443, aph_675, aCDOM_443, aCDOM_675, aNAP_443, aNAP_675]
iops_names = [iop.name for iop in iops]
iops_443 = [iop for iop in iops if "443" in iop.name]
iops_675 = [iop for iop in iops if "675" in iop.name]


### METRICS
mdsa = Parameter("mdsa", "MdSA [%]", vmin=0)
sspb = Parameter("sspb", "SSPB [%]", symmetric=True)
r_squared = Parameter("r_squared", r"$R^2$", vmax=1)
interval_sharpness = Parameter("sharpness", f"Sharpness", vmin=0, vmax=1)
coverage = Parameter("coverage", "Coverage [%]", vmin=0, vmax=100)
