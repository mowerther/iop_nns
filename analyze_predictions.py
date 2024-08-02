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
    print("\n\nRead recalibration metrics into `metrics_recal` DataFrame:")
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


### SELECT MEDIAN MODELS
median_indices, median_metrics = pnn.modeloutput.select_median_metrics(metrics)
if not args.no_recal:
    median_indices_recal, median_metrics_recal = pnn.modeloutput.select_median_metrics(metrics_recal)

# Miscalibration area
pnn.output.table_miscalibration_area(median_metrics, scenarios=scenarios, tag=tag)
print("Saved miscalibration area table")


### INDIVIDUAL MODEL OUTPUTS
print("\n\n\n--- INDIVIDUAL MODEL OUTPUTS ---")

# Load data
results = pnn.modeloutput.read_all_model_outputs(pnn.model_estimates_path, scenarios=scenarios, subfolder_indices=median_indices)
print("Read results into `results` DataFrame:")
print(results)

# Load recalibrated data
if not args.no_recal:
    results_recal = pnn.modeloutput.read_all_model_outputs(pnn.model_estimates_path, scenarios=scenarios, subfolder_indices=median_indices_recal, use_recalibration_data=True)
    print("Read recalibration results into `results_recal` DataFrame:")
    print(results_recal)

# Average uncertainty heatmap
uncertainty_averages = pnn.aggregate.average_uncertainty(results)
pnn.output.plot_uncertainty_heatmap(uncertainty_averages, scenarios=scenarios, tag=tag)
print("Saved uncertainty heatmap plot")

# Calibration curves
calibration_curves = pnn.aggregate.calibration_curve(results)
if args.no_recal:
    pnn.output.plot_calibration_curves(calibration_curves, rows=scenarios, tag=tag)
else:
    calibration_curves_recal = pnn.aggregate.calibration_curve(results_recal)
    pnn.output.plot_calibration_curves_with_recal(calibration_curves, calibration_curves_recal, rows=scenarios, tag=tag)
print("Saved calibration curve plot")

# y vs y_hat scatter plots
pnn.output.plot_performance_scatter_multi(results, scenarios=scenarios, tag=tag)
print("Saved match-up (scatter) plots")

# Log-binned uncertainty and line plot
binned = pnn.logbins.log_binned_statistics_combined(results)
pnn.output.plot_log_binned_statistics(binned, scenarios=scenarios, tag=tag)
print("Saved log-binned uncertainty (line) plot")
