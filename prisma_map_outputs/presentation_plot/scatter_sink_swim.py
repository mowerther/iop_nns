import numpy as np, matplotlib.pyplot as plt
np.random.seed(11)
n = 120
x = np.linspace(0, 10, n)

# ---------- “Paper results” ----------
y_paper = x + np.random.normal(0, 0.6, n)          # tight fit
r2_paper = np.corrcoef(x, y_paper)[0, 1]**2

fig, ax = plt.subplots()
ax.scatter(x, y_paper, s=18)
ax.plot(x, x, color='black', ls='-')
ax.set_xlabel('Variable X (reference measurement)')
ax.set_ylabel('Variable Y (model estimate)')
ax.set_title(f'Results reported in a paper', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(r'C:\SwitchDrive\Presentationen_not_L3P\figs_surf_day_2025\paper_scatter.png', dpi=300)
plt.show()

# ---------- “My application” ----------
# variant y_app: heteroskedastic + bias away from 1:1
sigma_base=1.0
sigma_ramp=np.clip((x-4)/2,0,1)*8+1  # 1 up to 9
sigma=np.where(x<4, sigma_base, sigma_ramp)

# two modes
high = x + np.random.normal(0, sigma/2)  # still close
low_mean = x - 0.6*(x-4)                 # pulls down below line starting at 4
low = low_mean + np.random.normal(0, sigma)

# probability of choosing low increases with x
p_low = np.clip((x-4)/6, 0, 1)           # 0 at 4, 1 at 10
rand=np.random.rand(n)
y_app = np.where(rand < p_low, low, high)


r2_app = np.corrcoef(x, y_app)[0, 1]**2

fig, ax = plt.subplots()
ax.scatter(x, y_app, s=18)
ax.plot(x, x, ls='--')
ax.axvspan(4, 6, alpha=.1)   # highlight breakdown zone
ax.plot(x, x, color='black', ls='-')
ax.axvspan(6, 10, color='grey', alpha=.1)   # highlight breakdown zone
ax.axvline(6.0, ls='--')
ax.set_xlabel('Variable X (reference measurement)')
ax.set_ylabel('Variable Y (model estimate)')
ax.set_title(f'When I apply the model to my dataset', fontsize=14, fontweight='bold')

# Add text annotations below the plot
# Use figure coordinates (0-1) for x and y, relative to the figure,
# to place text outside the axes area. Adjust y slightly to prevent overlap.
# We might need to adjust figure spacing later if text is cut off.
fig.text(0.28, 0.01, 'Interpolation\n(swim)', ha='center', va='bottom', fontsize=14)
fig.text(0.51, 0.01, 'Ambiguous\nregion', ha='center', va='bottom', fontsize=14)
fig.text(0.75, 0.01, 'Extrapolation\n(sink)', ha='center', va='bottom', fontsize=14)

# Increase bottom margin to make space for the text
plt.subplots_adjust(bottom=0.2) 

plt.savefig(r'C:\SwitchDrive\Presentationen_not_L3P\figs_surf_day_2025\my_appl_scatter.png', dpi=300)
plt.show()