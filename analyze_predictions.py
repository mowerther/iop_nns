"""
Script for loading PNN outputs and generating plots.
The individual results files are combined into a single DataFrame (`results`) which is then used for plotting and aggregation.

To plot the recalibration data, use the -c flag. Note: for now, this overwrites the existing plots, rather than using a separate filename.
"""
import pnn

### Parse command line arguments
import argparse
parser = argparse.ArgumentParser("Script for loading data and PNN outputs and generating plots.")
parser.add_argument("-c", "--recalibrate", help="apply recalibration", action="store_true")
args = parser.parse_args()


### MODEL METRICS
print("\n\n\n--- MODEL METRICS ---")

# Load data
metrics = pnn.modeloutput.read_all_model_metrics()
print("Read metrics into `metrics` DataFrame:")
print(metrics)

# Accuracy metrics and lollipop plot
metrics = metrics.groupby(level=["scenario", "network", "variable"]).first()  # Temporary
pnn.output.plot_accuracy_metrics(metrics)
print("Saved performance metric plot")

# Coverage plot
pnn.output.plot_coverage(metrics)
print("Saved coverage plot")

# Miscalibration area
pnn.output.table_miscalibration_area(metrics)
print("Saved miscalibration area table")


### TO DO: SELECT MEDIAN MODELS


### INDIVIDUAL MODEL OUTPUTS
print("\n\n\n--- INDIVIDUAL MODEL OUTPUTS ---")

# Load data
results = pnn.modeloutput.read_all_model_outputs(pnn.model_estimates_path/"0/", use_recalibration_data=args.recalibrate)
print("Read results into `results` DataFrame:")
print(results)

# Average uncertainty heatmap
uncertainty_averages = pnn.aggregate.average_uncertainty(results)
pnn.output.plot_uncertainty_heatmap(uncertainty_averages)
print("Saved uncertainty heatmap plot")

# Calibration curves
calibration_curves = pnn.aggregate.calibration_curve(results)
pnn.output.plot_calibration_curves(calibration_curves)
print("Saved calibration curve plot")

# y vs y_hat scatter plots
pnn.output.plot_performance_scatter_multi(results)
print("Saved match-up (scatter) plots")

# Log-binned uncertainty and line plot
binned = pnn.logbins.log_binned_statistics_combined(results)
pnn.output.plot_log_binned_statistics(binned)
print("Saved log-binned uncertainty (line) plot")
