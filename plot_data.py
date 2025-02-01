"""
Script for loading and plotting input data.
Data are assumed to be in default locations, but different folders can be specified:
    in situ: use the -i flag
    PRISMA match-ups: use the -p flag

Example:
    python plot_data.py
    python plot_data.py -i path/to/insitudata/ -p path/to/prismadata/
"""
import pandas as pd
import pnn

# Parse command-line args
import argparse
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("-i", "--insitu_folder", help="Folder containing in situ data.", default=pnn.insitu_data_path)
parser.add_argument("-p", "--prisma_folder", help="Folder containing PRISMA match-up data.", default=pnn.prisma_matchup_path)
args = parser.parse_args()

### IN SITU DATA
print("--- IN SITU DATA ---")
# Full dataset
data_full = pnn.data.read_insitu_full(args.insitu_folder)
print(f"Loaded full in situ dataset from `{args.insitu_folder.absolute()}`.")
pnn.output.plot_full_dataset(data_full)
print("Saved full data plot")

# Split datasets
data_random, data_wd, data_ood = pnn.read_insitu_data(args.insitu_folder)
print(f"Read in situ data splits from `{args.insitu_folder.absolute()}`.")
pnn.output.plot_scenarios(data_random, data_wd, data_ood)
print("Saved data splits plot")


### PRISMA DATA
print("--- PRISMA MATCH-UP DATA ---")
raise NotImplementedError("PRISMA match-up plots currently not working.")
# Load split data
train_data_prisma, test_data_prisma = pnn.read_prisma_matchups(args.prisma_folder)

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
