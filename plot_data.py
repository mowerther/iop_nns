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
parser = pnn.ArgumentParser(description=__doc__)
parser.add_argument("-i", "--insitu_folder", help="Folder containing in situ data.", default=pnn.insitu_data_path, type=pnn.c.Path)
parser.add_argument("-p", "--prisma_folder", help="Folder containing PRISMA match-up data.", default=pnn.insitu_data_path, type=pnn.c.Path)
parser.add_argument("--system_column", help="Column containing system names.", default="Waterbody_name")
args = parser.parse_args()

### IN SITU DATA
print("--- IN SITU DATA ---")
# Full dataset
data_full = pnn.data.read_insitu_full(args.insitu_folder)
print(f"Loaded full in situ dataset from `{args.insitu_folder.absolute()}`.")

# Print statistics
print(f"Number of records in in situ dataset: {len(data_full)}")

unique_systems = data_full[args.system_column].unique()
print(f"Number of unique systems in in situ dataset: {len(unique_systems)}")
try:
    unique_countries = data_full["Country"].unique()
    print(f"Number of unique countries in in situ dataset: {len(unique_countries)}")
except:
    pass

# Plot full data
pnn.output.plot_full_dataset(data_full)
print("Saved full data plot")

# Split datasets
data_random, data_wd, data_ood = pnn.read_insitu_data(args.insitu_folder)
print(f"Read in situ data splits from `{args.insitu_folder.absolute()}`.")
pnn.output.plot_scenarios(data_random, data_wd, data_ood)
print("Saved data splits plot")


### PRISMA DATA
print("--- PRISMA MATCH-UP DATA ---")
# Load PRISMA in situ data
prisma_insitu = pnn.data.read_prisma_insitu()  # Default filename for now
print(f"Loaded PRISMA in situ data.")
print(f"Number of records in PRISMA in situ dataset: {len(prisma_insitu)}")

# Load data scenarios
prisma_gen, prisma_lk = pnn.read_prisma_matchups(args.prisma_folder)
print(f"Loaded PRISMA data scenarios from `{args.prisma_folder.absolute()}`.")

for testscenario, testdata in prisma_lk.test_scenarios_and_data.items():
    print(f"PRISMA test scenario {testscenario}: {len(testdata)} match-ups.")

# Plot split data
raise NotImplementedError("PRISMA match-up plots currently not working.")
pnn.output.plot_prisma_scenarios(train_data_prisma, test_data_prisma)
print("Saved data splits plot")
