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
model_estimates_path = Path("pnn_model_estimates/")

output_path = Path("manuscript_figures/")
supplementary_path = output_path/"supplementary/"


### UNITS
m1 = r"m$^{-1}$"
m2 = r"m$^{-2}$"


### PLOTTING
_CMAP_N = 10
cmap_uniform = plt.cm.cividis.resampled(_CMAP_N)
cmap_aleatoric_fraction = managua.resampled(_CMAP_N)
cmap_difference = plt.cm.BrBG_r.resampled(_CMAP_N)


### PARAMETERS
@dataclass
class Parameter:
    name: str
    label: str
    color: Optional[str] = "black"
    cmap: Optional[Colormap] = field(default_factory=lambda: cmap_uniform)
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    extend_cbar: Optional[str] = "neither"
    symmetric: bool = False
    label_2lines: Optional[str] = None
    unit: Optional[str] = ""

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


### SCENARIOS
# Overview
gloria = Parameter("gloria", "GLORIA+")
prisma = Parameter("prisma", "PRISMA")

# Scenarios 1, 2, 3 (GLORIA+)
random_split = Parameter("random_split", "Random split", label_2lines="Random\nsplit")
wd = Parameter("wd_split", "Within-distribution", label_2lines="Within-\ndistribution")
ood = Parameter("ood_split", "Out-of-distribution", label_2lines="Out-of-\ndistribution")

training_123 = [random_split, wd, ood]
testing_123 = {s: [s] for s in training_123}  # 1:1 match between training and testing scenarios
scenarios_123 = training_123  # Scenarios for assessment

wavelengths_123 = list(range(400, 701, 5))

# PRISMA scenarios
_insitu = r"$\it{in}$ $\it{situ}$"
prisma_insitu = Parameter("prisma_insitu", f"{_insitu} vs. {_insitu}", label_2lines=f"{_insitu} vs.\n{_insitu}")

prisma_gen = Parameter("prisma_gen", "PRISMA: General")
prisma_gen_L2 = Parameter("prisma_gen_l2", "General: L2", label_2lines="General\nL2")
prisma_gen_ACOLITE = Parameter("prisma_gen_aco", "General: ACOLITE", label_2lines="General\nACOLITE")

prisma_lk = Parameter("prisma_lk", "PRISMA: Local knowledge")
prisma_lk_L2 = Parameter("prisma_lk_l2", "Local knowledge: L2", label_2lines="Local knowledge\nL2")
prisma_lk_ACOLITE = Parameter("prisma_lk_aco", "Local knowledge: ACOLITE", label_2lines="Local knowledge\nACOLITE")

_prisma_L2 = Parameter("prisma_l2", "L2")
_prisma_ACOLITE = Parameter("prisma_aco", "ACOLITE")
_scenarios_prisma_sub = [_prisma_L2, _prisma_ACOLITE]

testing_prisma = {prisma_gen: [prisma_insitu, prisma_gen_L2, prisma_gen_ACOLITE],
                  prisma_lk: [prisma_lk_L2, prisma_lk_ACOLITE],}
training_prisma = list(testing_prisma.keys())
scenarios_prisma = [s for sub in testing_prisma.values() for s in sub]  # Flattened version of testing scenarios

scenarios_prisma_overview = [prisma_insitu] + training_prisma  # General scenario descriptions
scenarios_prisma_scatter = [[prisma_insitu]] + [sub[-2:] for sub in testing_prisma.values()]  # Specifically for scatter plots

wavelengths_prisma = [406, 415, 423, 431, 438, 446, 453, 460, 468, 475, 482, 489, 497, 504, 512, 519, 527, 535, 542, 550, 559, 567, 575, 583, 592, 601, 609, 618, 627, 636, 645, 655, 664, 674, 684, 694]


### UNCERTAINTY TYPES
ale_var = Parameter("ale_var", f"Aleatoric variance [{m2}]", cmap_aleatoric_fraction.colors[-3], vmin=0, unit=m1)
ale_unc = Parameter("ale_unc", f"Aleatoric uncertainty [{m1}]", cmap_aleatoric_fraction.colors[-3], vmin=0, unit=m1)
ale_unc_pct = Parameter("ale_unc_pct", "Aleatoric uncertainty [%]", cmap_aleatoric_fraction.colors[-3], vmin=0, vmax=100, extend_cbar="max", unit="%")

epi_var = Parameter("epi_var", f"Epistemic variance [{m2}]", cmap_aleatoric_fraction.colors[2], vmin=0, unit=m2)
epi_unc = Parameter("epi_unc", f"Epistemic uncertainty [{m1}]", cmap_aleatoric_fraction.colors[2], vmin=0, unit=m1)
epi_unc_pct = Parameter("epi_unc_pct", "Epistemic uncertainty [%]", cmap_aleatoric_fraction.colors[2], vmin=0, vmax=100, extend_cbar="max", unit="%")

total_var = Parameter("total_var", f"Total variance [{m2}]", "black", vmin=0, unit=m2)
total_unc = Parameter("total_unc", f"Total uncertainty [{m1}]", "black", vmin=0, unit=m1)
total_unc_pct = Parameter("total_unc_pct", "Total uncertainty [%]", "black", vmin=0, vmax=200, extend_cbar="max", unit="%")

ale_frac = Parameter("ale_frac", "Aleatoric fraction [%]", cmap=cmap_aleatoric_fraction, vmin=0, vmax=100, unit="%")

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
aph_443 = Parameter("aph_443", f"$a_{_ph}$(443)", label_2lines=f"$a_{_ph}$\n(443)", color="darkgreen", unit=m1)
aph_675 = Parameter("aph_675", f"$a_{_ph}$(675)", label_2lines=f"$a_{_ph}$\n(675)", color="darkgreen", unit=m1)
aCDOM_443 = Parameter("aCDOM_443", f"$a_{_CDOM}$(443)", label_2lines=f"$a_{_CDOM}$\n(443)", color="darkgoldenrod", unit=m1)
aCDOM_675 = Parameter("aCDOM_675", f"$a_{_CDOM}$(675)", label_2lines=f"$a_{_CDOM}$\n(675)", color="darkgoldenrod", unit=m1)
aNAP_443 = Parameter("aNAP_443", f"$a_{_NAP}$(443)", label_2lines=f"$a_{_NAP}$\n(443)", color="saddlebrown", unit=m1)
aNAP_675 = Parameter("aNAP_675", f"$a_{_NAP}$(675)", label_2lines=f"$a_{_NAP}$\n(675)", color="saddlebrown", unit=m1)

iops = [aph_443, aph_675, aCDOM_443, aCDOM_675, aNAP_443, aNAP_675]
iops_names = [iop.name for iop in iops]

_iop_subset = lambda substring: [iop for iop in iops if substring in iop.name]
iops_aph = _iop_subset("aph")
iops_aCDOM = _iop_subset("aCDOM")
iops_aNAP = _iop_subset("aNAP")
iops_443 = _iop_subset("443")
iops_675 = _iop_subset("675")


### METRICS
# Accuracy
mdsa = Parameter("MdSA", "MdSA [%]", vmin=0, unit="%")
sspb = Parameter("SSPB", "SSPB [%]", symmetric=True, unit="%")
r_squared = Parameter("r_squared", r"$R^2$", vmax=1)
log_r_squared = Parameter("log_r_squared", r"$R^2$", vmax=1)  # R² of log, not log of R²

# Uncertainty
coverage = Parameter("coverage", "Coverage [%]", vmin=100, vmax=0, unit="%")
interval_sharpness = Parameter("sharpness", f"Sharpness", vmin=0, vmax=1)
miscalibration_area = Parameter("MA", "Miscalibration area", vmin=0, vmax=0.4, extend_cbar="max")  # Real max value is 1, but 0.4 is better for plots


### COMBINED KEYS
scenario_network = ["scenario", "network"]
