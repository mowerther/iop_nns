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


### Setup
def load_data(template: str, pnn1, pnn2) -> tuple[pnn.maps.xr.Dataset, pnn.maps.xr.Dataset]:
    filename1, filename2 = [args.folder/template.format(pnn_type=pnn_type) for pnn_type in (pnn1, pnn2)]
    data1, data2 = [pnn.maps.load_map(filename) for filename in (filename1, filename2)]
    return data1, data2


### Figure 1: Prisma_2023_05_24_10_17_20_converted L2C, 443 nm, ens-nn and mdn
filename_template = "PRISMA_2023_05_24_10_17_20_converted_L2C-{pnn_type}-prisma_gen_aco_iops.nc"
pnn1, pnn2 = pnn.c.ensemble, pnn.c.mdn
data1, data2 = load_data(filename_template, pnn1, pnn2)

raise Exception

### Figure 2: Prisma_2023_09_11_10_13_53_L2W, 675 nm, bnn-mcd and rnn
filename_template = "PRISMA_2023_09_11_10_13_53_L2W-{pnn_type}-prisma_gen_l2_iops.nc"
pnn1, pnn2 = pnn.c.bnn_mcd, pnn.c.rnn
data1, data2 = load_data(filename_template, pnn1, pnn2)

### Figure 3: Trasimeno
filename_template = "PRISMA_2023_05_24_10_17_20_L2W-{pnn_type}-prisma_gen_l2_iops.nc"
pnn1, pnn2 = pnn.c.bnn_mcd, pnn.c.bnn_dc
data1, data2 = load_data(filename_template, pnn1, pnn2)
