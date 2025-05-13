import numpy as np, matplotlib.pyplot as plt
np.random.seed(1)

n_in, n_trans, n_out = 120, 80, 120
pred_in = np.linspace(0.5, 4.0, n_in) + np.random.normal(0, .05, n_in)
pred_trans = np.linspace(4.0, 6.0, n_trans) + np.random.normal(0, .08, n_trans)
pred_out = np.linspace(6.0, 10.0, n_out) + np.random.normal(0, .4, n_out)
pred = np.concatenate([pred_in, pred_trans, pred_out])

perf_in = 0.89 + np.random.normal(0, .02, n_in)

# --- ambiguous region --------------------------------------------------------
high_mode = 0.87 + np.random.normal(0, 0.03, n_trans)   # swim
low_mode  = 0.55 + np.random.normal(0, 0.08, n_trans)   # sink
p_low = (pred_trans - 4.0)/2.0                          # 0 at 4 → 1 at 6
rand  = np.random.rand(n_trans)
perf_trans = np.where(rand < p_low, low_mode, high_mode)


perf_at_ambiguous_end = 0.89 - 0.2 * np.maximum(0, 6.0 - 4.0) # Approx 0.49
perf_out = perf_at_ambiguous_end * np.exp(-1.5*(pred_out-6.0)) + np.random.normal(0, .05, n_out)
perf = np.concatenate([perf_in, perf_trans, perf_out])

# soft ceiling: any value above 0.95 is pulled back by a random offset in [0, 0.02)
upper = 0.94
mask = perf > upper
perf[mask] = upper - np.random.rand(mask.sum())*0.03   # 0.93-0.95 range
perf = np.clip(perf, 0, upper)


fig, ax = plt.subplots()
ax.scatter(pred, perf, s=10)
ax.axvspan(4.0, 6.0, alpha=.1)
ax.axvspan(6.0, 10, color='grey', alpha=.1)
ax.axvline(6.0, ls='--')
ax.set_xlabel('Model prediction value'); ax.set_ylabel('Performance')

ax.set_ylim(0, 1)

y_text_location = 1.05

ax.text(2.25, y_text_location, 'Interpolation\n(swim)',
        transform=ax.get_xaxis_transform(), ha='center', va='bottom', clip_on=False)
ax.text(5.0, y_text_location, 'Ambiguous\nregion',
        transform=ax.get_xaxis_transform(), ha='center', va='bottom', clip_on=False)
ax.text(8.0, y_text_location, 'Extrapolation\n(sink)',
        transform=ax.get_xaxis_transform(), ha='center', va='bottom', clip_on=False)

plt.tight_layout()
plt.savefig(r'C:\SwitchDrive\Presentationen_not_L3P\figs_surf_day_2025\sink_swim.png', dpi=300)
plt.show()
