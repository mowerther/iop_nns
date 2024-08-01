"""
Script for loading data and PNN outputs and generating plots.
The individual results files are combined into a single DataFrame (`results`) which is then used for plotting and aggregation.

To plot the recalibration data, use the -c flag. Note: for now, this overwrites the existing plots, rather than using a separate filename.
"""
import pandas as pd
import pnn

### INPUT DATA
print("--- INPUT DATA ---")
# Load split data
train_data, test_data = pnn.read_scenario123_data()
print("Read results into 6 DataFrames")

# Load full data
data_full = pd.concat([train_data[0], test_data[0]])
print(data_full)

# Plot full data
pnn.output.plot_full_dataset(data_full)
print("Saved full data plot")

# Plot split data
pnn.output.plot_data_splits(train_data, test_data)
print("Saved data splits plot")
