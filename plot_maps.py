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
