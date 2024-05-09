from pnn import io, logbins, metrics, plot

### LOAD DATA
results = io.read_all_data()
print("Read results into `results` dictionary")
print(results.keys())


### LOG-BINNED UNCERTAINTY (LINE) PLOT
binned = {key: logbins.log_binned_statistics_combined(df) for key, df in results.items()}
plot.plot_log_binned_statistics(binned)
print("Saved log-binned uncertainty (line) plot")


### LOLLIPOP PLOT
metrics_results = {key: metrics.calculate_metrics(df) for key, df in results.items()}
plot.plot_performance_metrics_lollipop(metrics_results)
print("Saved performance metric (lollipop) plot")
