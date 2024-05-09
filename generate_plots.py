import itertools
from pathlib import Path
from typing import Iterable, Optional
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter

from pnn import io, logbins, metrics, plot
from pnn.constants import pred_path, save_path, iops, network_types, split_types

### LOAD DATA
results = {f"{network}_{split}": io.read_data(pred_path/f"{network}_{split}_preds.csv") for network, split in itertools.product(network_types, split_types)}
print("Read results into `results` dictionary")
print(results.keys())


### LOG-BINNED UNCERTAINTY (LINE) PLOT
binned = {key: logbins.log_binned_statistics_combined(df) for key, df in results.items()}
plot.plot_log_binned_statistics(binned)
print("Saved log-binned uncertainty (line) plot")


### LOLLIPOP PLOT
metrics_results = {key: metrics.calculate_metrics(df) for key, df in results.items()}
# The metrics_results dictionary now contains dfs with calculated metrics for each algorithm and variable

# Separating the results for the scenarios
random_metrics_results = {k: v for k, v in metrics_results.items() if '_random' in k}
wd_metrics_results = {k: v for k, v in metrics_results.items() if '_wd' in k}
ood_metrics_results = {k: v for k, v in metrics_results.items() if '_ood' in k}

# Custom y-axis labels and their limits
y_axis_labels = list(metrics.metrics_display.keys())
y_axis_limits = [(0, 140), (-140, 50), (0, 1)]

# Model and display titles for the variables
new_model_labels = ['MDN', 'BNN DC', 'BNN MCD', 'ENS NN', 'RNN']
display_titles = ['a$_{CDOM}$ 443', 'a$_{CDOM}$ 675', 'a$_{ph}$ 443', 'a$_{ph}$ 675']

def plot_vertical_lollipop_charts(metrics_results_list, titles):
    n_groups = len(display_titles)
    n_metrics = len(metrics.metrics_display)
    n_models = len(metrics_results_list[0])
    bar_width = 0.15
    opacity = 0.8

    fig, axs = plt.subplots(n_metrics, 3, figsize=(14, 8), sharex='col')
    #fig.suptitle("Performance Metrics", y=0.99, fontsize=16, fontweight='bold')

    for i, (metric_column, metric_display) in enumerate(metrics.metrics_display.items()):
        for j, metrics_results in enumerate(metrics_results_list):
            ax = axs[i, j]
            ax.set_ylim(y_axis_limits[i])

            for model_idx, (model_key, df) in enumerate(metrics_results.items()):
                model_short = model_key.split('_')[0]
                color = plot.model_colors.get(model_short, 'gray')

                # Drop aNAP
                df = df.drop(columns=[key for key in df.columns if "NAP" in key])

                x = df.loc[metric_column]
                y = np.arange(n_groups) - (bar_width * (n_models - 1) / 2) + model_idx * bar_width
                ax.scatter(y, x, color=color, label=new_model_labels[model_idx], s=50, zorder=3)  # Draw points
                ax.vlines(y, 0, x, colors='grey', lw=1, alpha=0.7)

            if j == 0:
                ax.set_ylabel(metric_display,fontweight='bold',fontsize=12)
            if i == 0:
                ax.set_title(titles[j])

            ax.set_xticks(np.arange(n_groups))
            ax.set_xticklabels(display_titles)
            ax.grid(True, which='both', linestyle='--', linewidth='0.5', color='black', alpha=0.4, axis='y')

    # Plot legend outside the subplots
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(save_path/'performance_lolliplot_vertical.png', dpi=200, bbox_inches='tight')
    plt.show()

# Example usage with the revised function and hypothetical data structure
plot_vertical_lollipop_charts([random_metrics_results, wd_metrics_results, ood_metrics_results], ['Random split', 'Within-distribution split', 'Out-of-distribution split'])
print("Saved performance metric (lollipop) plot")
