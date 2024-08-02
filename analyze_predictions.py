"""
Script for loading PNN outputs and generating plots.
The individual results files are combined into a single DataFrame (`results`) which is then used for plotting and aggregation.

To plot the PRISMA data, use the -p flag.
If you don't want to plot recalibrated data, use the --no_recal flag.
"""
import pnn

### Parse command line arguments
import argparse
parser = argparse.ArgumentParser("Script for loading PNN outputs and generating plots.")
parser.add_argument("-p", "--prisma", help="use PRSIMA data", action="store_true")
parser.add_argument("-n", "--no_recal", help="do not use recalibrated data", action="store_true")
args = parser.parse_args()


### SELECT GLORIA OR PRISMA
tag, scenarios, _ = pnn.data.select_scenarios(prisma=args.prisma)


### MODEL METRICS
print("\n\n\n--- MODEL METRICS ---")

# Load data
metrics = pnn.modeloutput.read_all_model_metrics(scenarios=scenarios)
print("Read metrics into `metrics` DataFrame:")
print(metrics)

# Load recalibrated data
if not args.no_recal:
    metrics_recal = pnn.modeloutput.read_all_model_metrics(scenarios=scenarios, use_recalibration_data=True)
    print("\n\nRead recalibrationmetrics into `metrics_recal` DataFrame:")
    print(metrics_recal)

# Accuracy metric plot
pnn.output.plot_accuracy_metrics(metrics, scenarios=scenarios, tag=tag)
print("Saved accuracy metric plot")

# Coverage plot
if args.no_recal:
    pnn.output.plot_coverage(metrics, scenarios=scenarios, tag=tag)
else:
    pnn.output.plot_coverage_with_recal(metrics, metrics_recal, scenarios=scenarios, tag=tag)
print("Saved coverage plot")


### TO DO: SELECT MEDIAN MODELS
metrics_median = metrics.groupby(level=["scenario", "network", "variable"]).first()  # Temporary


# Miscalibration area
pnn.output.table_miscalibration_area(metrics_median, scenarios=scenarios, tag=tag)
print("Saved miscalibration area table")


### INDIVIDUAL MODEL OUTPUTS
print("\n\n\n--- INDIVIDUAL MODEL OUTPUTS ---")

# Load data
results = pnn.modeloutput.read_all_model_outputs(pnn.model_estimates_path/"0/", scenarios=scenarios)
print("Read results into `results` DataFrame:")
print(results)

# Average uncertainty heatmap
uncertainty_averages = pnn.aggregate.average_uncertainty(results)
pnn.output.plot_uncertainty_heatmap(uncertainty_averages, scenarios=scenarios, tag=tag)
print("Saved uncertainty heatmap plot")

# Calibration curves
calibration_curves = pnn.aggregate.calibration_curve(results)
pnn.output.plot_calibration_curves(calibration_curves, rows=scenarios, tag=tag)
print("Saved calibration curve plot")

# y vs y_hat scatter plots
pnn.output.plot_performance_scatter_multi(results, scenarios=scenarios, tag=tag)
print("Saved match-up (scatter) plots")

# Log-binned uncertainty and line plot
binned = pnn.logbins.log_binned_statistics_combined(results)
pnn.output.plot_log_binned_statistics(binned, scenarios=scenarios, tag=tag)
print("Saved log-binned uncertainty (line) plot")
