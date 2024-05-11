"""
Main script for loading data and generating plots.
"""
from pnn import aggregate, io, logbins, metrics, plot

### LOAD DATA
results = io.read_all_data()
print("Read results into `results` dictionary")
print(results.keys())


### MODEL PERFORMANCE
# y vs y_hat scatter plots
plot.plot_performance_scatter(results)
print("Saved match-up (scatter) plots")

# Performance metrics and lollipop plot
metrics_results = {key: metrics.calculate_metrics(df) for key, df in results.items()}
plot.plot_performance_metrics_lollipop(metrics_results)
print("Saved performance metric (lollipop) plot")


### MODEL UNCERTAINTY
# Log-binned uncertainty and line plot
binned = {key: logbins.log_binned_statistics_combined(df) for key, df in results.items()}
plot.plot_log_binned_statistics(binned)
print("Saved log-binned uncertainty (line) plot")

# Average uncertainty heatmap
uncertainty_averages = aggregate.average_uncertainty(results)
plot.uncertainty_heatmap(uncertainty_averages)
print("Saved uncertainty heatmap plot")

# Sharpness/Coverage heatmap
sharpness_coverage = aggregate.sharpness_coverage(results)
print("Saved sharpness/coverage plot")
