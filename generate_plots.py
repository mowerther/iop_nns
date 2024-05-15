"""
Main script for loading data and generating plots.
The individual results files are combined into a single DataFrame (`results`) which is then used for plotting and aggregation.
"""
import pnn

### LOAD DATA
results = pnn.io.read_all_model_outputs()
print("Read results into `results` DataFrame:")
print(results)


### MODEL PERFORMANCE
# y vs y_hat scatter plots
pnn.plot.plot_performance_scatter(results)
print("Saved match-up (scatter) plots")

# Performance metrics and lollipop plot
metrics = pnn.aggregate.calculate_metrics(results)
pnn.plot.plot_performance_metrics_lollipop(metrics)
print("Saved performance metric (lollipop) plot")


### MODEL UNCERTAINTY
# Log-binned uncertainty and line plot
binned = pnn.logbins.log_binned_statistics_combined(results)
pnn.plot.plot_log_binned_statistics(binned)
print("Saved log-binned uncertainty (line) plot")

# Average uncertainty heatmap
uncertainty_averages = pnn.aggregate.average_uncertainty(results)
pnn.plot.uncertainty_heatmap(uncertainty_averages)
print("Saved uncertainty heatmap plot")

# Sharpness/Coverage heatmap
pnn.plot.plot_uncertainty_metrics_bar(metrics)
print("Saved sharpness/coverage plot")
