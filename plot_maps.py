"""
Script for loading map data and applying PNN estimation pixel-wise.
First loads and plots (atmospherically corrected) reflectance data.
Next finds and applies the average-performing model (by median MdSA) for each network-scenario combination.

Data are loaded from pnn.c.map_data_path by default, but a custom folder can be supplied using the -f flag (e.g. `python plot_maps.py -f /path/to/my/folder`).
Please note that figures will still be saved to the same location.

Recalibrated models are not currently supported.
"""
import pnn

### Parse command line arguments
import argparse
parser = argparse.ArgumentParser(__doc__.splitlines()[1])
parser.add_argument("-f", "--folder", help="folder to load data from", type=pnn.c.Path, default=pnn.c.map_data_path)
args = parser.parse_args()


### Load data
filenames_l2c = args.folder.glob(pnn.maps.pattern_prisma_l2)


for filename in filenames_l2c:
    data = pnn.maps._load_general(filename)
    print(f"Read data from `{filename.absolute()}`")

    # Plot Rrs for reference
    pnn.maps.plot_Rrs(data)

    # Convert Rrs to list of spectra

    # Rescale Rrs

    # Load PNN

    # Apply PNN

    # Rescale IOPs

    # Convert IOPs to maps

    # Plot IOP maps - main output
