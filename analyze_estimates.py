"""
Script for loading PNN outputs and generating plots.
First loads and analyzes the model metrics from all runs combined.
Next finds and analyzes the average-performing model (by median MdSA) for each network-scenario combination.

Data are loaded from pnn.c.model_estimates_path by default, but a custom folder can be supplied using the -f flag (e.g. `python analyze_estimates.py -f /path/to/my/folder`).
Please note that figures will still be saved to the same location.

To plot the PRISMA data, use the -p flag.
To include recalibrated data, use the -c flag.
"""
import pnn

### Parse command line arguments
import argparse
parser = argparse.ArgumentParser("Script for loading PNN outputs and generating plots.")
parser.add_argument("-f", "--folder", help="folder to load data from", type=pnn.c.Path, default=pnn.c.model_estimates_path)
parser.add_argument("-p", "--prisma", help="use PRISMA data", action="store_true")
parser.add_argument("-c", "--recal", help="use recalibrated data", action="store_true")
args = parser.parse_args()


### SELECT GLORIA OR PRISMA
tag, scenarios, iops, *_ = pnn.data.select_scenarios(prisma=args.prisma)


### MODEL METRICS
print("\n\n\n--- MODEL METRICS ---")

# Load data
metrics = pnn.modeloutput.read_all_model_metrics(args.folder, scenarios=scenarios)
print("Read metrics into `metrics` DataFrame:")
print(metrics)

# Load recalibrated data
if args.recal:
    metrics_recal = pnn.modeloutput.read_all_model_metrics(args.folder, scenarios=scenarios, use_recalibration_data=True)
    print("\n\nRead recalibration metrics into `metrics_recal` DataFrame:")
    print(metrics_recal)

# Accuracy metric plot
pnn.output.plot_accuracy_metrics(metrics, scenarios=scenarios, tag=tag)
print("Saved accuracy metric plot")

# MdSA plot
pnn.output.print_mdsa_range(metrics, scenarios=scenarios, variables=iops)
if args.prisma:
    pnn.output.plot_mdsa(metrics, scenarios=scenarios, tag=tag)
    if args.recal:
        pnn.output.plot_mdsa(metrics_recal, scenarios=scenarios, tag=f"{tag}_recal")
    print("Saved MdSA plot")

if args.recal:
    pnn.output.recalibration_change_in_mdsa(metrics, metrics_recal)

# Coverage
pnn.output.print_coverage_range(metrics, variables=iops)
if args.recal:
    pnn.output.plot_coverage_with_recal(metrics, metrics_recal, scenarios=scenarios, groups=iops, tag=tag)
else:
    pnn.output.plot_coverage(metrics, scenarios=scenarios, tag=tag)
print("Saved coverage plot")

# Miscalibration area
if args.recal:
    pnn.output.recalibration_improvement(metrics, metrics_recal)
    pnn.output.recalibration_MA_threshold(metrics, metrics_recal)
    pnn.output.plot_recalibration_MA(metrics, metrics_recal, tag=tag)


### SELECT MEDIAN MODELS
print("\n\n\n--- AVERAGE-PERFORMING MODEL ---")
median_indices, median_metrics = pnn.modeloutput.select_median_metrics(metrics)
if args.recal:
    median_indices_recal, median_metrics_recal = pnn.modeloutput.select_median_metrics(metrics_recal)

# MdSA
pnn.output.print_mdsa(median_metrics)
if args.recal:
    print("Recalibration difference:")
    pnn.output.print_mdsa_difference(metrics, metrics_recal, median_metrics, median_metrics_recal)

# Miscalibration area
if args.recal:
    pnn.output.miscalibration_area_heatmap(median_metrics, median_metrics_recal, scenarios=scenarios, variables=iops, tag=tag)
print("Saved miscalibration area plot")


### INDIVIDUAL MODEL OUTPUTS
print("\n\n\n--- INDIVIDUAL (AVERAGE-PERFORMING) MODEL OUTPUTS ---")

# Load data
results = pnn.modeloutput.read_all_model_outputs(args.folder, pnn.model_estimates_path, scenarios=scenarios, subfolder_indices=median_indices)
print("Read results into `results` DataFrame:")
print(results)

# Load recalibrated data
if args.recal:
    results_recal = pnn.modeloutput.read_all_model_outputs(args.folder, pnn.model_estimates_path, scenarios=scenarios, subfolder_indices=median_indices_recal, use_recalibration_data=True)
    print("Read recalibration results into `results_recal` DataFrame:")
    print(results_recal)

# PRISMA only: scatter plot
if args.prisma:
    pnn.output.plot_prisma_scatter_multi(results, tag=tag)
    print("Saved PRISMA match-up (scatter) plots")

# Average uncertainty heatmap
uncertainty_averages = pnn.aggregate.average_uncertainty(results)
pnn.output.plot_uncertainty_heatmap(uncertainty_averages, scenarios=scenarios, tag=tag)
if args.recal:
    uncertainty_averages_recal = pnn.aggregate.average_uncertainty(results_recal)
    pnn.output.plot_uncertainty_heatmap_with_recal(uncertainty_averages, uncertainty_averages_recal, scenarios=scenarios, variables=iops, tag=tag)
print("Saved uncertainty heatmap plot")

# Average uncertainty: ratios between scenarios
if not args.prisma:
    pnn.output.compare_uncertainty_scenarios_123(uncertainty_averages)

# Calibration curves
calibration_curves = pnn.aggregate.calibration_curve(results)
if args.recal:
    calibration_curves_recal = pnn.aggregate.calibration_curve(results_recal)
    pnn.output.plot_calibration_curves_with_recal(calibration_curves, calibration_curves_recal, rows=scenarios, columns=iops, tag=tag)
else:
    pnn.output.plot_calibration_curves(calibration_curves, rows=scenarios, tag=tag)
print("Saved calibration curve plot")
