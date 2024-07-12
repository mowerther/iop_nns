"""
Script for loading data and PNN outputs and generating plots.
The individual results files are combined into a single DataFrame (`results`) which is then used for plotting and aggregation.
"""
import pandas as pd
import pnn


### INPUT DATA
print("--- INPUT DATA ---")
# Load split data
train_set_random, test_set_random, train_set_wd, test_set_wd, train_set_ood, test_set_ood = pnn.data.read_all_data()
print("Read results into 6 DataFrames")

# Load full data
data_full = pd.concat([train_set_random, test_set_random])
print(data_full)

# Plot full data
pnn.plot.plot_full_dataset(data_full)
print("Saved full data plot")

# Plot split data
pnn.plot.plot_data_splits(train_set_random, test_set_random, train_set_wd, test_set_wd, train_set_ood, test_set_ood)
print("Saved data splits plot")


### MODEL OUTPUTS
print("\n\n\n--- MODEL OUTPUTS ---")

# Load data
results = pnn.modeloutput.read_all_model_outputs()
print("Read results into `results` DataFrame:")
print(results)

# Performance metrics and lollipop plot
metrics = pnn.aggregate.calculate_metrics(results)
pnn.plot.plot_performance_metrics_lollipop(metrics)
print("Saved performance metric (lollipop) plot")

# Average uncertainty heatmap
uncertainty_averages = pnn.aggregate.average_uncertainty(results)
pnn.plot.uncertainty_heatmap(uncertainty_averages)
print("Saved uncertainty heatmap plot")

# Coverage plot
pnn.plot.plot_coverage(metrics)
print("Saved coverage plot")

# Miscalibration area
miscalibration_areas = pnn.aggregate.miscalibration_area(results)

# Calibration curves
calibration_curves = pnn.aggregate.calibration_curve(results)
pnn.plot.plot_calibration_curves(calibration_curves)
print("Saved calibration curve plot")

raise Exception

# y vs y_hat scatter plots
pnn.plot.plot_performance_scatter(results)
print("Saved match-up (scatter) plots")

# Log-binned uncertainty and line plot
binned = pnn.logbins.log_binned_statistics_combined(results)
pnn.plot.plot_log_binned_statistics(binned)
print("Saved log-binned uncertainty (line) plot")
