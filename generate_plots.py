"""
Script for loading data and PNN outputs and generating plots.
The individual results files are combined into a single DataFrame (`results`) which is then used for plotting and aggregation.
"""
import pnn

### LOAD DATA
results = pnn.modeloutput.read_all_model_outputs()
print("Read results into `results` DataFrame:")
print(results)


### GENERATE PLOTS
# Performance metrics and lollipop plot
metrics = pnn.aggregate.calculate_metrics(results)
pnn.plot.plot_performance_metrics_lollipop(metrics)
print("Saved performance metric (lollipop) plot")

# Average uncertainty heatmap
uncertainty_averages = pnn.aggregate.average_uncertainty(results)
pnn.plot.uncertainty_heatmap(uncertainty_averages)
print("Saved uncertainty heatmap plot")

# Sharpness/Coverage plot
pnn.plot.plot_uncertainty_metrics_bar(metrics)
print("Saved sharpness/coverage plot")

# Calibration curves
calibration_curves = pnn.aggregate.calibration_curve(results)
pnn.plot.plot_calibration_curves(calibration_curves)
print("Saved calibration curve plot")

# y vs y_hat scatter plots
pnn.plot.plot_performance_scatter(results)
print("Saved match-up (scatter) plots")

# Log-binned uncertainty and line plot
binned = pnn.logbins.log_binned_statistics_combined(results)
pnn.plot.plot_log_binned_statistics(binned)
print("Saved log-binned uncertainty (line) plot")
