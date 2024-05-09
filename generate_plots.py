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
from pnn.constants import pred_path, save_path, iops_main, network_types, split_types

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
plot.plot_performance_metrics_lollipop(metrics_results)
print("Saved performance metric (lollipop) plot")
