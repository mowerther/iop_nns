"""
Script for loading processed PRISMA scenes and creating specific figures.
- Loads IOP estimate scenes.
- Creates specified figures.

Data are loaded from pnn.c.map_output_path by default, but a custom folder can be supplied using the -f flag (e.g. `python plot_prisma_scenes.py -f /path/to/my/folder`).
Please note that figures will still be saved to the same location.

Example:
    python plot_prisma_scenes.py bnn_mcd
"""
import numpy as np
from matplotlib import pyplot as plt
import pnn

### Parse command line arguments
parser = pnn.ArgumentParser(description=__doc__)
parser.add_argument("-f", "--folder", help="folder to load processed scenes from", type=pnn.c.Path, default=pnn.c.map_output_path)
args = parser.parse_args()


### Set up match-ups
matchups = pnn.data.read_prisma_insitu(filter_invalid_dates=True)


### Setup
def load_data(template: str, pnn1, pnn2) -> tuple[pnn.maps.xr.Dataset, pnn.maps.xr.Dataset]:
    # Load outputs
    filename1, filename2 = [args.folder/template.format(pnn_type=pnn_type) for pnn_type in (pnn1, pnn2)]
    iop1, iop2 = [pnn.maps.load_map(filename) for filename in (filename1, filename2)]

    # Load original scene for Rrs
    acolite = ("converted_L2C" in template)
    ac_label = "converted_L2C" if acolite else "L2W"
    scenename = pnn.c.map_data_path / (template[:26] + f"_{ac_label}" + ".nc")
    scene = pnn.maps.load_prisma_map(scenename, acolite=acolite)

    # Load RGB background image
    filename_h5 = pnn.maps.get_h5_filename(scenename)
    rgb_cube = pnn.maps.load_h5_as_rgb(filename_h5)
    background = pnn.maps.rgb_to_xarray(scene, rgb_cube)

    # Find match-ups
    _, date = pnn.maps.filename_to_date(scenename)
    matchups_here = pnn.maps.find_matchups_on_date(matchups, date)

    return scene, background, matchups_here, iop1, iop2


def create_figure() -> tuple[plt.Figure, np.ndarray]:
    fig, axs = pnn.maps._create_map_figure(nrows=5, ncols=3, projected=True, figsize=(9, 12))
    return fig, axs


def plot_Rrs(axs: np.ndarray, scene: pnn.maps.xr.Dataset, wavelength: int, **kwargs) -> None:
    ax_Rrs = axs[0, 1]
    pnn.maps.plot_Rrs(scene, wavelength, ax=ax_Rrs, **kwargs)
    pnn.output.label_topleft(ax_Rrs, r"$R_{rs}$" f"({wavelength})")
    axs[0, 0].axis("off")
    axs[0, 2].axis("off")

### Figure 1: Prisma_2023_05_24_10_17_20_converted L2C, 443 nm, ens-nn and mdn
filename_template = "PRISMA_2023_05_24_10_17_20_converted_L2C-{pnn_type}-prisma_gen_aco_iops.nc"
pnn1, pnn2 = pnn.c.ensemble, pnn.c.bnn_mcd
scene, background, matchups_here, iop1, iop2 = load_data(filename_template, pnn1, pnn2)

fig, axs = create_figure()

shared_kw = {"projected": True, "background": background, "matchups": matchups_here}
plot_Rrs(axs, scene, 446, **shared_kw)

axs1 = axs[1:3]
axs2 = axs[3:]

for data, pnn_type, axs_here in zip([iop1, iop2], [pnn1, pnn2], [axs1, axs2]):
    for iop, axs_iop in zip(pnn.c.iops_443, axs_here.T):
        pnn.maps.plot_IOP_single(data, iop, axs=axs_iop, uncmin=0, uncmax=300, **shared_kw)

        # Labels
        for ax in axs_iop:
            pnn.output.label_topleft(ax, f"{iop.label} ({pnn_type.label})")

plt.savefig("Map1.pdf", dpi=600)
plt.show()
plt.close()





### Figure 2: Prisma_2023_09_11_10_13_53_L2W, 675 nm, bnn-mcd and rnn
filename_template = "PRISMA_2023_09_11_10_13_53_L2W-{pnn_type}-prisma_gen_l2_iops.nc"
pnn1, pnn2 = pnn.c.mdn, pnn.c.rnn
scene, background, matchups_here, iop1, iop2 = load_data(filename_template, pnn1, pnn2)

fig, axs = create_figure()

shared_kw = {"projected": True, "background": background, "matchups": matchups_here}
plot_Rrs(axs, scene, 446, **shared_kw)

axs1 = axs[1:3]
axs2 = axs[3:]

for data, pnn_type, axs_here in zip([iop1, iop2], [pnn1, pnn2], [axs1, axs2]):
    for iop, axs_iop in zip(pnn.c.iops_675, axs_here.T):
        pnn.maps.plot_IOP_single(data, iop, axs=axs_iop, uncmin=0, uncmax=300, **shared_kw)

        # Labels
        for ax in axs_iop:
            pnn.output.label_topleft(ax, f"{iop.label} ({pnn_type.label})")

plt.savefig("Map2.pdf", dpi=600)
plt.show()
plt.close()



### Figure 3: Trasimeno
filename_template = "PRISMA_2022_07_20_10_08_04_L2W-{pnn_type}-prisma_gen_l2_iops.nc"
pnn1, pnn2 = pnn.c.rnn, pnn.c.bnn_dc
scene, background, matchups_here, iop1, iop2 = load_data(filename_template, pnn1, pnn2)

fig, axs = create_figure()

shared_kw = {"projected": True, "background": background, "matchups": matchups_here}
plot_Rrs(axs, scene, 446, **shared_kw)

axs1 = axs[1:3]
axs2 = axs[3:]

for data, pnn_type, axs_here in zip([iop1, iop2], [pnn1, pnn2], [axs1, axs2]):
    for iop, axs_iop in zip(pnn.c.iops_443, axs_here.T):
        pnn.maps.plot_IOP_single(data, iop, axs=axs_iop, uncmin=0, uncmax=300, **shared_kw)

        # Labels
        for ax in axs_iop:
            pnn.output.label_topleft(ax, f"{iop.label} ({pnn_type.label})")

plt.savefig("Map3.pdf", dpi=600)
plt.show()
plt.close()
