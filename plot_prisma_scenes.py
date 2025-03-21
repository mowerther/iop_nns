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
import pandas as pd
from matplotlib import pyplot as plt
import pnn

### Parse command line arguments
parser = pnn.ArgumentParser(description=__doc__)
parser.add_argument("-f", "--folder", help="folder to load processed scenes from", type=pnn.c.Path, default=pnn.c.map_output_path)
args = parser.parse_args()

### Set up match-ups
matchups = pnn.data.read_prisma_insitu(filter_invalid_dates=True)

### Setup
def load_data(template: str, *pnn_types, use_recalibrated_data=False) -> tuple[pnn.maps.xr.Dataset, pd.DataFrame]:
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


def _matchup_pixels_single(matchup: pd.Series, scene: pnn.maps.xr.Dataset) -> pd.Series:
    distance2 = (scene["lat"] - matchup["lat"])**2 + (scene["lon"] - matchup["lon"])**2  # Ignore geospatial effects for now
    closest = distance2.argmin(dim=["x", "y"])
    closest = {key: int(val) for key, val in closest.items()}
    closest = pd.Series(closest)
    return closest


def _matchup_get_value(matchup: pd.Series, scene: pnn.maps.xr.Dataset) -> pd.Series:
    indices = matchup.to_dict()
    indices = {key: slice(val-1, val+2) for key, val in indices.items()}  # 3x3 square
    pixels = scene[indices]
    return pixels.mean().to_pandas()


def matchup_pixels(matchups: pd.DataFrame, scene: pnn.maps.xr.Dataset, iops: pnn.maps.xr.Dataset) -> None:
    """
    Find pixels in the scene (Rrs) and IOP map that are closest to the matchups and compare the corresponding data.
    """
    # Find coordinates closest to matchups
    matchups_xy = matchups.apply(_matchup_pixels_single, axis=1, args=(scene,))
    matchups_scene = [matchups_xy.apply(_matchup_get_value, axis=1, args=(data,)) for data in [scene, iops]]
    matchups_scene = pd.concat(matchups_scene, axis=1)

    # Filter entries with only land pixels
    water_filter = (matchups_scene["water"] > 0)
    matchups, matchups_scene = matchups.loc[water_filter], matchups_scene.loc[water_filter]
    print(f"Number of matchups: {len(matchups)} in situ  ;  {len(matchups_scene)} scene")

    # Rename Rrs columns to match (different rounding between in situ data and scenes)
    wavelengths_diff = [int(col[4:]) for col in matchups_scene.columns.difference(matchups.columns) if "Rrs" in col]
    matchups_scene = matchups_scene.rename(columns={f"Rrs_{wvl}": f"Rrs_{wvl-1}" for wvl in wavelengths_diff})

    # Calculate statistics
    prisma_rrs_cols = [f"Rrs_{wvl}" for wvl in pnn.c.wavelengths_prisma]

    # MdSA
    for cols, label in zip([prisma_rrs_cols, pnn.c.iops], ["Rrs", "IOPs"]):
        values_insitu, values_scene = matchups[cols], matchups_scene[cols]

        mdsa = pnn.metrics.mdsa(values_insitu, values_scene)
        print(mdsa)
        mdsa_overall = pnn.metrics.mdsa(values_insitu.unstack(), values_scene.unstack())
        print(f"{label} MdSA overall: {mdsa_overall:.1f}%")

    # Coverage (IOPs only)
    values_insitu, values_scene = matchups[pnn.c.iops], matchups_scene[pnn.c.iops]
    uncertainties_scene = matchups_scene[[f"{iop}_std" for iop in pnn.c.iops]].rename(columns={f"{iop}_std": iop for iop in pnn.c.iops})
    coverage = pnn.metrics.coverage(values_insitu, values_scene, uncertainties_scene)
    print("IOP coverage:")
    print(coverage)


def matchup_pixels_map2(matchups: pd.DataFrame, scene: pnn.maps.xr.Dataset, iops: pnn.maps.xr.Dataset, model_name: str) -> None:
    """
    Extended matchup analysis for Map 2 with additional metrics and station-specific analysis.
    """
    # Find coordinates closest to matchups
    matchups_xy = matchups.apply(_matchup_pixels_single, axis=1, args=(scene,))
    matchups_scene = [matchups_xy.apply(_matchup_get_value, axis=1, args=(data,)) for data in [scene, iops]]
    matchups_scene = pd.concat(matchups_scene, axis=1)

    # Filter entries with only land pixels
    water_filter = (matchups_scene["water"] > 0)
    matchups, matchups_scene = matchups.loc[water_filter], matchups_scene.loc[water_filter]
    print(f"\n===== {model_name} MODEL METRICS =====")
    print(f"Number of matchups: {len(matchups)} in situ  ;  {len(matchups_scene)} scene")

    # Rename Rrs columns to match (different rounding between in situ data and scenes)
    wavelengths_diff = [int(col[4:]) for col in matchups_scene.columns.difference(matchups.columns) if "Rrs" in col]
    matchups_scene = matchups_scene.rename(columns={f"Rrs_{wvl}": f"Rrs_{wvl-1}" for wvl in wavelengths_diff})

    # Identify shallow and deep stations
    shallow_stations = ['st2.1', 'st2.2', 'st2.3', 'st2.4']
    has_station_info = 'station' in matchups.columns
    
    # Create masks for shallow and deep stations
    if has_station_info:
        shallow_mask = matchups['station'].isin(shallow_stations)
        deep_mask = ~shallow_mask
        
        shallow_count = shallow_mask.sum()
        deep_count = deep_mask.sum()
        print(f"Found {shallow_count} optically shallow stations and {deep_count} optically deep stations")
    
    # Get values for IOPs only
    values_insitu, values_scene = matchups[pnn.c.iops], matchups_scene[pnn.c.iops]
    
    # 1. COMBINED METRICS (all stations)
    print("\n----- COMBINED (ALL STATIONS) -----")
    
    # MdSA for IOPs
    mdsa = pnn.metrics.mdsa(values_insitu, values_scene)
    print(f"IOPs MdSA:")
    print(mdsa)
    
    mdsa_overall = pnn.metrics.mdsa(values_insitu.unstack(), values_scene.unstack())
    print(f"IOPs MdSA overall: {mdsa_overall:.1f}%")
    
    # Calculate SSPB
    sspb = pnn.metrics.sspb(values_insitu, values_scene)
    print(f"IOPs SSPB:")
    print(sspb)
    
    # Calculate MAE
    print(f"IOPs MAE:")
    for iop in pnn.c.iops:
        insitu_vals = values_insitu[iop].values
        scene_vals = values_scene[iop].values
        # Filter out NaN values
        valid_mask = ~np.isnan(insitu_vals) & ~np.isnan(scene_vals)
        if np.any(valid_mask):
            mae = np.median(np.abs(scene_vals[valid_mask] - insitu_vals[valid_mask]))
            print(f"{iop}: {mae:.3f} m^-1")
    
    # Calculate station-specific metrics if station info is available
    if has_station_info:
        # 2. SHALLOW STATIONS METRICS
        if shallow_count > 0:
            print("\n----- OPTICALLY SHALLOW STATIONS -----")
            shallow_insitu, shallow_scene = values_insitu.loc[shallow_mask], values_scene.loc[shallow_mask]
            
            # MdSA
            shallow_mdsa = pnn.metrics.mdsa(shallow_insitu, shallow_scene)
            print(f"IOPs MdSA (shallow stations):")
            print(shallow_mdsa)
            
            shallow_mdsa_overall = pnn.metrics.mdsa(shallow_insitu.unstack(), shallow_scene.unstack())
            print(f"IOPs MdSA overall (shallow stations): {shallow_mdsa_overall:.1f}%")
            
            # SSPB
            shallow_sspb = pnn.metrics.sspb(shallow_insitu, shallow_scene)
            print(f"IOPs SSPB (shallow stations):")
            print(shallow_sspb)
            
            # MAE
            print(f"IOPs MAE (shallow stations):")
            for iop in pnn.c.iops:
                insitu_vals = shallow_insitu[iop].values
                scene_vals = shallow_scene[iop].values
                # Filter out NaN values
                valid_mask = ~np.isnan(insitu_vals) & ~np.isnan(scene_vals)
                if np.any(valid_mask):
                    mae = np.median(np.abs(scene_vals[valid_mask] - insitu_vals[valid_mask]))
                    print(f"{iop}: {mae:.3f} m^-1")
        
        # 3. DEEP STATIONS METRICS
        if deep_count > 0:
            print("\n----- OPTICALLY DEEP STATIONS -----")
            deep_insitu, deep_scene = values_insitu.loc[deep_mask], values_scene.loc[deep_mask]
            
            # MdSA
            deep_mdsa = pnn.metrics.mdsa(deep_insitu, deep_scene)
            print(f"IOPs MdSA (deep stations):")
            print(deep_mdsa)
            
            deep_mdsa_overall = pnn.metrics.mdsa(deep_insitu.unstack(), deep_scene.unstack())
            print(f"IOPs MdSA overall (deep stations): {deep_mdsa_overall:.1f}%")
            
            # SSPB
            deep_sspb = pnn.metrics.sspb(deep_insitu, deep_scene)
            print(f"IOPs SSPB (deep stations):")
            print(deep_sspb)
            
            # MAE
            print(f"IOPs MAE (deep stations):")
            for iop in pnn.c.iops:
                insitu_vals = deep_insitu[iop].values
                scene_vals = deep_scene[iop].values
                # Filter out NaN values
                valid_mask = ~np.isnan(insitu_vals) & ~np.isnan(scene_vals)
                if np.any(valid_mask):
                    mae = np.median(np.abs(scene_vals[valid_mask] - insitu_vals[valid_mask]))
                    print(f"{iop}: {mae:.3f} m^-1")

    # Coverage (IOPs only)
    uncertainties_scene = matchups_scene[[f"{iop}_std" for iop in pnn.c.iops]].rename(columns={f"{iop}_std": iop for iop in pnn.c.iops})
    coverage = pnn.metrics.coverage(values_insitu, values_scene, uncertainties_scene)
    print("\n----- COVERAGE -----")
    print(f"IOP coverage:")
    print(coverage)

# Custom function to plot matchups with specific colors for Map 2
def plot_custom_matchups_map2(matchups, ax, **kwargs):
    """
    Plot match-ups as scatter points with custom colors for specific stations in Map 2.
    """
    if matchups is None or len(matchups) == 0:
        return
    
    # Create a copy to avoid modifying the original
    matchups = matchups.copy()
    
    # Define special stations
    special_stations = ['st2.1', 'st2.2', 'st2.3', 'st2.4']
    
    # Check if 'station' column exists
    if 'station' not in matchups.columns:
        print("Warning: 'station' column not found in matchups data")
        # Use default plotting
        pnn.maps._plot_matchups(matchups, ax, **kwargs)
        return
    
    # Split into special and regular stations
    is_special = matchups['station'].isin(special_stations)
    special_matchups = matchups[is_special]
    regular_matchups = matchups[~is_special]
    
    # Handle color parameter if provided
    c = kwargs.pop('c', None)
    
    # Plot regular matchups with black outline
    if len(regular_matchups) > 0:
        regular_kwargs = kwargs.copy()
        if c is not None and hasattr(c, '__len__') and len(c) == len(matchups):
            # Extract colors only for regular matchups using the same boolean mask
            regular_kwargs['c'] = c[~is_special]
        
        ax.scatter(regular_matchups["lon"], regular_matchups["lat"], 
                  transform=pnn.maps.projection,
                  s=25, edgecolor="black", marker="D", **regular_kwargs)
    
    # Plot shallow matchups with orange outline
    if len(special_matchups) > 0:
        special_kwargs = kwargs.copy()
        if c is not None and hasattr(c, '__len__') and len(c) == len(matchups):
            # Extract colors only for special matchups using the same boolean mask
            special_kwargs['c'] = c[is_special]
            
        ax.scatter(special_matchups["lon"], special_matchups["lat"], 
                  transform=pnn.maps.projection,
                  s=25, edgecolor="orange", marker="D", **special_kwargs)


### Figure 1: Prisma_2023_05_24_10_17_20_converted L2C, 443 nm, ens-nn and mdn
# Map 1: Exchange MDN with BNN-MCD
filename_template = "PRISMA_2023_05_24_10_17_20_converted_L2C-{pnn_type}-prisma_gen_aco_iops.nc"
pnn1, pnn2 = pnn.c.ensemble, pnn.c.bnn_mcd
scene, background, matchups_here, iop1, iop2 = load_data(filename_template, pnn1, pnn2)

# Compare matchups
print(filename_template)
for pnnx, iopx in zip([pnn1, pnn2], [iop1, iop2]):
    print(f"{pnnx.label} match-ups:")
    matchup_pixels(matchups_here, scene, iopx)
    print()

# Plot
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
print("Saved Map1.pdf\n\n\n")


### Figure 2: Prisma_2023_09_11_10_13_53_L2W, 675 nm, bnn-mcd and rnn
# Map 2: replace BNN-MCD with MDN
filename_template = "PRISMA_2023_09_11_10_13_53_L2W-{pnn_type}-prisma_gen_l2_iops.nc"
pnn1, pnn2 = pnn.c.mdn, pnn.c.rnn
scene, background, matchups_here, iop1, iop2 = load_data(filename_template, pnn1, pnn2)

# Compare matchups
print(filename_template)
for pnnx, iopx in zip([pnn1, pnn2], [iop1, iop2]):
    print(f"{pnnx.label} match-ups:")
    matchup_pixels_map2(matchups_here, scene, iopx, pnnx.label)
    print()

# Print matchup stations for verification
if matchups_here is not None and 'station' in matchups_here.columns:
    print("Matchup stations found:", matchups_here['station'].tolist())

# Plot
fig, axs = create_figure()

# Store the original _plot_matchups function
original_plot_matchups = pnn.maps._plot_matchups

# Custom function for Map 2
pnn.maps._plot_matchups = plot_custom_matchups_map2

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

pnn.maps._plot_matchups = original_plot_matchups

plt.savefig("Map2.pdf", dpi=600)
plt.show()
plt.close()
print("Saved Map2.pdf\n\n\n")


### Figure 2.5: uncertainties of map 2: without and with recalibration just from the two models, without the IOP maps
*_, iop1_recal, iop2_recal = load_data(filename_template, pnn1, pnn2, use_recalibrated_data=True)

# Compare matchups
print("RECALIBRATED", filename_template)
for pnnx, iopx in zip([pnn1, pnn2], [iop1_recal, iop2_recal]):
    print(f"{pnnx.label} match-ups:")
    matchup_pixels(matchups_here, scene, iopx)
    print()

# Plot
fig, axs = pnn.maps._create_map_figure(nrows=4, ncols=3, projected=True, figsize=(6.6, 9), gridspec_kw={"wspace": 0.02})

norm_unc = pnn.maps.Normalize(vmin=0, vmax=300)

for j, (data, pnn_type, is_recal, ax_row) in enumerate(zip([iop1, iop1_recal, iop2, iop2_recal], [pnn1, pnn1, pnn2, pnn2], [False, True, False, True], axs)):
    for iop, ax in zip(pnn.c.iops_675, ax_row):
        # Setup cmaps and norms
        unc = f"{iop}_std_pct"
        cbar_kwargs = {"label": f"Uncertainty in {iop.label} [%]" if j == len(axs)-1 else None} | pnn.maps.kw_cbar

        pnn.maps._plot_with_background(data, unc, ax=ax,
                                       norm=norm_unc, cmap=pnn.maps.cmap_unc, mask_land=False, projected=True, background=background, cbar_kwargs=cbar_kwargs)

        # Label
        pnn.maps.o.label_topleft(ax, f"{pnn_type.label}{' (rec.)' if is_recal else ''}")

pnn.maps.o.label_axes_sequentially(axs)

plt.savefig("Map2_recal.pdf", dpi=600)
plt.show()
plt.close()
print("Saved Map2_recal.pdf\n\n\n")
