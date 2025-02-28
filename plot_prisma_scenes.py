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
def load_data(template: str, *pnn_types, use_recalibrated_data=False) -> tuple[pnn.maps.xr.Dataset, pnn.maps.pd.DataFrame]:
    # Load outputs
    filenames = [args.folder/template.format(pnn_type=pnn_type) for pnn_type in pnn_types]
    if use_recalibrated_data:
        filenames = [fn.with_stem(fn.stem.replace("_iops", "-recal_iops")) for fn in filenames]

    iop_maps = [pnn.maps.load_map(filename) for filename in filenames]

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

    return scene, background, matchups_here, *iop_maps


def create_figure() -> tuple[plt.Figure, np.ndarray]:
    fig, axs = pnn.maps._create_map_figure(nrows=5, ncols=3, projected=True, figsize=(6.6, 12), gridspec_kw={"hspace": 0.10, "wspace": 0.02})
    return fig, axs


def plot_Rrs(axs: np.ndarray, scene: pnn.maps.xr.Dataset, wavelength: int, **kwargs) -> None:
    ax_Rrs = axs[0, 1]
    pnn.maps.plot_Rrs(scene, wavelength, ax=ax_Rrs, **kwargs)
    axs[0, 0].axis("off")
    axs[0, 2].axis("off")




### Figure 1: Prisma_2023_05_24_10_17_20_converted L2C, 443 nm, ens-nn and mdn
# Map 1: Exchange MDN with BNN-MCD
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
            pnn.output.label_topleft(ax, pnn_type.label)

pnn.maps.o.label_axes_sequentially([ax for ax in axs.ravel() if ax.collections])

plt.savefig("Map1.pdf", dpi=600)
plt.show()
plt.close()




### Figure 1.5: uncertainties of map 1: without and with recalibration just from the two models, without the IOP maps
*_, iop1_recal, iop2_recal = load_data(filename_template, pnn1, pnn2, use_recalibrated_data=True)

fig, axs = pnn.maps._create_map_figure(nrows=4, ncols=3, projected=True, figsize=(6.6, 9), gridspec_kw={"wspace": 0.02})

norm_unc = pnn.maps.Normalize(vmin=0, vmax=300)

for j, (data, pnn_type, is_recal, ax_row) in enumerate(zip([iop1, iop1_recal, iop2, iop2_recal], [pnn1, pnn1, pnn2, pnn2], [False, True, False, True], axs)):
    cbar_kwargs = {"label": "Uncertainty [%]" if j == len(axs)-1 else None} | pnn.maps.kw_cbar

    for iop, ax in zip(pnn.c.iops_443, ax_row):
        # Setup cmaps and norms
        unc = f"{iop}_std_pct"

        pnn.maps._plot_with_background(data, unc, ax=ax,
                                       norm=norm_unc, cmap=pnn.maps.cmap_unc, mask_land=False, projected=True, background=background, cbar_kwargs=cbar_kwargs)

        # Label
        pnn.maps.o.label_topleft(ax, f"{pnn_type.label}{' (rec.)' if is_recal else ''}")

pnn.maps.o.label_axes_sequentially(axs)

plt.savefig("Map1_recal.pdf", dpi=600)
plt.show()
plt.close()




### Figure 2: Prisma_2023_09_11_10_13_53_L2W, 675 nm, bnn-mcd and rnn
# Map 2: replace BNN-MCD with MDN
filename_template = "PRISMA_2023_09_11_10_13_53_L2W-{pnn_type}-prisma_gen_l2_iops.nc"
pnn1, pnn2 = pnn.c.mdn, pnn.c.rnn
scene, background, matchups_here, iop1, iop2 = load_data(filename_template, pnn1, pnn2)

fig, axs = create_figure()

shared_kw = {"projected": True, "background": background, "matchups": matchups_here}
plot_Rrs(axs, scene, 674, **shared_kw)

axs1 = axs[1:3]
axs2 = axs[3:]

for data, pnn_type, axs_here in zip([iop1, iop2], [pnn1, pnn2], [axs1, axs2]):
    for iop, axs_iop in zip(pnn.c.iops_675, axs_here.T):
        pnn.maps.plot_IOP_single(data, iop, axs=axs_iop, uncmin=0, uncmax=300, **shared_kw)

        # Labels
        for ax in axs_iop:
            pnn.output.label_topleft(ax, pnn_type.label)

pnn.maps.o.label_axes_sequentially([ax for ax in axs.ravel() if ax.collections])

plt.savefig("Map2.pdf", dpi=600)
plt.show()
plt.close()
