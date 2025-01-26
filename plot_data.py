"""
Script for loading and plotting the input data.
Data are assumed to be in default locations.

Example:
    python plot_data.py
"""
import pandas as pd
import pnn

### GLORIA+ DATA
print("--- GLORIA+ DATA ---")
# Load split data
train_data, test_data = pnn.read_scenario123_data()
print(f"Read GLORIA+ data into {len(train_data)}+{len(test_data)} DataFrames")

# Load full data
data_full = pd.concat([train_data[0], test_data[0]])
print(data_full)

# Plot full data
pnn.output.plot_full_dataset(data_full)
print("Saved full data plot")

# Plot split data
pnn.output.plot_scenarios(train_data, test_data)
print("Saved data splits plot")



### PRISMA DATA
print("--- PRISMA DATA ---")
# Load split data
train_data_prisma, test_data_prisma = pnn.read_prisma_data()

# Since we only care about IOPs, use one dataset per subscenario
train_data_prisma, test_data_prisma = train_data_prisma[::2], test_data_prisma[::2]
print(f"Read PRISMA data into {len(train_data_prisma)}+{len(test_data_prisma)} DataFrames")

# Iterate over core scenarios
for scenario, train_data, test_data in zip(pnn.scenarios_prisma_overview, train_data_prisma, test_data_prisma):
    print(scenario)

    # Load full data
    data_full = pd.concat([train_data, test_data])
    print(data_full)

    # Plot full data
    pnn.output.plot_full_dataset(data_full, saveto=pnn.output_path/f"dataset_{scenario}.pdf")
    print("Saved full data plot")

# Plot split data
pnn.output.plot_prisma_scenarios(train_data_prisma, test_data_prisma)
print("Saved data splits plot")
