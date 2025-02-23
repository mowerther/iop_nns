"""
Script for loading processed PRISMA scenes and creating specific figures.
- Loads IOP estimate scenes.
- Creates specified figures.

Data are loaded from pnn.c.map_output_path by default, but a custom folder can be supplied using the -f flag (e.g. `python plot_prisma_scenes.py -f /path/to/my/folder`).
Please note that figures will still be saved to the same location.

Example:
    python plot_prisma_scenes.py bnn_mcd
"""
import pnn

### Parse command line arguments
parser = pnn.ArgumentParser(description=__doc__)
parser.add_argument("-f", "--folder", help="folder to load processed scenes from", type=pnn.c.Path, default=pnn.c.map_output_path)
args = parser.parse_args()


### Set up match-ups
matchups = pnn.data.read_prisma_insitu(filter_invalid_dates=True)


### Figure 1: Prisma_2023_05_24_10_17_20_converted L2C, 443 nm, ens-nn and mdn


### Figure 2: Prisma_2023_09_11_10_13_53_L2W, 675 nm, bnn-mcd and rnn


### Figure 3: Trasimeno
