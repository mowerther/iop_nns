import numpy as np
import matplotlib.pyplot as plt

water_types = [
    "Open\nocean",
    "Polar\nwaters",
    "Coastal\nsea",
    "Estuary\nbrackish",
    "Lake\nfreshwater",
    "River\nfreshwater",
    "Hypersaline\nlagoon"
]

x = np.arange(len(water_types))
train_err = np.array([0.07, 0.07] + [0.07]*(len(water_types)-2))
test_err  = np.array([0.15, 0.15, 0.20, 0.25, 0.35, 0.40, 0.55])

plt.figure(figsize=(8, 4))
plt.axvspan(-0.5, 1.5, alpha=0.15, label="Seen water types\n(training domain)")
plt.plot(x, train_err, marker="o", label="Error on training set")
plt.plot(x, test_err,  marker="o", label="Error on test set")
plt.fill_between(x, train_err, test_err, where=test_err>train_err, alpha=0.25,
                 label="Generalisation gap")
plt.xticks(x, water_types)
plt.ylabel("Prediction error")
plt.title("Generalisation: from known (ocean) to new water types")
plt.ylim(0, 0.6)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(r'C:\SwitchDrive\Presentationen_not_L3P\figs_surf_day_2025\generalisation_gap.png', dpi=300)
plt.show()