"""
Functions for reading and processing spatial (map) data.
"""
from functools import partial
from pathlib import Path
from typing import Iterable, Optional

import h5py
import numpy as np
import xarray as xr

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from cartopy.crs import PlateCarree
from cmcrameri.cm import batlow

from . import constants as c


### CONSTANTS
pattern_prisma_acolite = "PRISMA_*_converted_L2C.nc"
pattern_prisma_l2 = "PRISMA_*_L2W.nc"
projection = PlateCarree()


### DATA LOADING
def _load_general(filename: Path | str) -> xr.Dataset:
    """
    Load and pre-process NetCDF files.
    """
    # Data loading
    data = xr.open_dataset(filename)
    data = data.set_coords(["lon", "lat"])
    return data


def NDWI(data: xr.Dataset) -> xr.DataArray:
    green, nir = data["Rrs_559"], data["Rrs_860"]
    return (green - nir) / (green + nir)


def mask_water(data: xr.Dataset, *, threshold: float=0.) -> xr.DataArray:
    """
    Calculate NDWI and check if it is above a threshold (above -> water) to generate a mask that is True for water pixels and False for land pixels.
    """
    ndwi = NDWI(data)
    ndwi_over_threshold = (ndwi >= threshold)
    return ndwi_over_threshold


def select_prisma_columns(data: xr.Dataset, key: str="Rrs") -> xr.Dataset:
    # Select columns PNNs were trained on
    # Note that there are rounding differences between the data and pnn.constants -- we simply select on the min and max
    lmin, lmax = c.wavelengths_prisma[0], c.wavelengths_prisma[-1]
    def column_in_scope(column_name: str) -> bool:
        is_reflectance = (key in column_name)
        if is_reflectance:
            wavelength = float(column_name.split("_")[-1])
            in_range = (lmin <= wavelength <= lmax)
            return in_range
        else:
            return False

    cols = [col for col in data.keys() if column_in_scope(col)]
    data = data[cols]

    return data


def _load_l2c(filename: Path | str) -> xr.Dataset:
    """
    Load L2C processed data.
    """
    data = _load_general(filename)

    # Filter Rrs
    data_Rrs = select_prisma_columns(data, key="Rrs")

    # Add mask based on original data
    mask = mask_water(data)
    data_Rrs["water"] = mask

    return data_Rrs


def _load_acolite(filename: Path | str) -> xr.Dataset:
    """
    Load ACOLITE-processed data and convert rho_w to R_rs.
    """
    data = _load_general(filename)

    # Convert rho_w to R_rs
    renamer = {rhos: f"Rrs_{rhos[9:]}" for rhos in data.keys()}
    data_Rrs = data.rename(renamer)
    data_Rrs = data_Rrs / np.pi

    # Add mask
    mask = mask_water(data_Rrs)

    # Filter Rrs
    data_Rrs = select_prisma_columns(data_Rrs, key="Rrs")
    data_Rrs["water"] = mask

    return data_Rrs


def load_prisma_map(filename: Path | str, acolite=False) -> xr.Dataset:
    """
    Load a PRISMA scene from a netCDF file.
    If `acolite`, use the ACOLITE file loader (including rho_s -> R_rs conversion), else the L2C one
    """
    load_data = _load_acolite if acolite else _load_l2c
    data_Rrs = load_data(filename)
    return data_Rrs


### MAPS <-> SPECTRA
def map_to_spectra(data: xr.Dataset, mask_land=True) -> tuple[np.ndarray, tuple[int]]:
    """
    Extract the spectrum from each pixel in an (x, y)-shaped image into an (x * y)-length array of spectra.
    Note that this loses information on variable names, coordinates, etc, so take care to keep this around elsewhere.
    If `mask_land`, remove rows that were masked in the data.
    """
    # Simple case: convert to map and get shape
    data_Rrs = select_prisma_columns(data)
    data_as_numpy = data_Rrs.to_array().values
    map_shape = data_as_numpy.shape
    data_as_numpy = data_as_numpy.reshape((map_shape[0], -1))
    data_as_numpy = data_as_numpy.T

    # Apply mask if desired
    if mask_land:
        mask_as_numpy = data["water"].values.ravel()
        data_as_numpy = data_as_numpy[mask_as_numpy]

    return data_as_numpy, map_shape


def _list_to_dataset_shape(data_list: np.ndarray, reference_scene: xr.Dataset, *, mask_land=True) -> np.ndarray:
    """
    Convert a list into a map corresponding to the dimensions of a given xarray Dataset.
    The first dimension of data_list is converted into 2D spatial dimensions.
    The list is first transposed so that other variables become indices in the result.
    If `mask_land`, apply the mask from `reference_scene` to the data.
    """
    n_variables = data_list.shape[1]
    new_shape = tuple(reference_scene.sizes.values())

    # Masked case: create an empty array simulating the original scene, then fill up the relevant pixels only
    if mask_land:
        # Setup
        full_length = np.prod(new_shape)
        data_list_full = np.tile(np.nan, (full_length, n_variables))
        mask_as_numpy = reference_scene["water"].values.ravel()

        # Assign values corresponding to mask
        data_list_full[mask_as_numpy] = data_list
        data_list = data_list_full

    # Reshape the data
    data_as_map = data_list.T.reshape(n_variables, *new_shape)

    return data_as_map


def spectra_to_map(data: np.ndarray, map_shape: tuple[int] | xr.Dataset, *, mask_land=True) -> np.ndarray | xr.Dataset:
    """
    Reshape a list of spectra back into a pre-defined map shape.
    If `map_shape` is an xarray Dataset, copy its georeferencing etc.
    If `mask_land`, apply the mask from `map_shape` to the data.
    """
    if isinstance(map_shape, xr.Dataset):
        data_as_map = _list_to_dataset_shape(data, map_shape, mask_land=mask_land)
        data_as_dict = {var: (map_shape.dims, arr) for var, arr in zip(map_shape.variables, data_as_map)}
        new_scene = xr.Dataset(data_as_dict, coords=map_shape.coords)
        data_as_map = new_scene

    elif isinstance(map_shape, tuple):
        data_as_map = data.T.reshape(map_shape)

    return data_as_map


def create_iop_map(iop_mean: np.ndarray, iop_variance: np.ndarray, reference_scene: xr.Dataset, *,
                   iop_labels: Optional[Iterable[c.Parameter]]=c.iops, mask_land=True) -> xr.Dataset:
    """
    Convert IOP estimates (mean and variance -> uncertainty) into an xarray Dataset like a provided scene.
    If `mask_land`, assume the means/variances only apply to masked pixels in the reference scene.
    """
    # Reshape to 2D
    iop_mean = _list_to_dataset_shape(iop_mean, reference_scene, mask_land=mask_land)
    iop_variance = _list_to_dataset_shape(iop_variance, reference_scene, mask_land=mask_land)

    # Calculate uncertainty
    iop_std = np.sqrt(iop_variance)
    iop_std_pct = iop_std / iop_mean * 100

    # Cast into xarray
    mean_dict = {f"{iop}": (reference_scene.dims, arr) for iop, arr in zip(iop_labels, iop_mean)}
    std_dict = {f"{iop}_std": (reference_scene.dims, arr) for iop, arr in zip(iop_labels, iop_std)}
    std_pct_dict = {f"{iop}_std_pct": (reference_scene.dims, arr) for iop, arr in zip(iop_labels, iop_std_pct)}
    combined_dict = mean_dict | std_dict | std_pct_dict
    iop_map = xr.Dataset(combined_dict, coords=reference_scene.coords)

    return iop_map


### OUTPUT
def save_iop_map(data: xr.Dataset, saveto: Path | str, **kwargs) -> None:
    """
    Save a Dataset to file.
    Thin wrapper to allow future functionality to be added.
    """
    data.to_netcdf(saveto, **kwargs)


### PLOTTING
_create_map_figure = partial(plt.subplots, subplot_kw={"projection": projection})


def _plot_land_RGB(data: xr.Dataset, ax: plt.Axes, **kwargs) -> None:
    """
    Plot RGB layers as a colour image.
    NOT compatible with projections.
    """
    # Combine into RGB cube
    data_rgb = xr.concat([data[c] for c in "rgb"], dim="c")

    # Plot
    data_rgb.plot.imshow(ax=ax, **kwargs) # robust=True,


def _plot_land_brightness(data: xr.Dataset, ax: plt.Axes, *, rasterized=True, **kwargs) -> None:
    """
    Plot a brightness layer in greyscale.
    Compatible with projections.
    If no brightness layer exists, try to create it from RGB layers.
    """
    # Find or create brightness layer
    try:
        brightness = data["brightness"]
    except KeyError:
        brightness = 0.3 * data["r"] + 0.6 * data["g"] + 0.1 * data["b"]

    # Plot
    brightness.plot.pcolormesh(ax=ax, cmap="gray", robust=True, add_colorbar=False, rasterized=rasterized, **kwargs)


def _plot_land(data: xr.Dataset, ax: plt.Axes, *, use_rgb=True, mask_water=True, **kwargs) -> None:
    """
    Plot a background image contained in `data`.
    If `use_rgb`, use _plot_land_RGB to plot an un-projected RGB image.
    Else, use _plot_land_brightness to plot a projected greyscale image.
    If `mask_water`, only show RGB where `water` is False.
    """
    # Optional: mask
    if mask_water:
        data = data.where(~data["water"])

    # Plot into ax
    if use_rgb:
        _plot_land_RGB(data, ax, **kwargs)
    else:
        _plot_land_brightness(data, ax, transform=projection, x="lon", y="lat", **kwargs)


def plot_Rrs(data: xr.Dataset, *, col: str="Rrs_446",
             projected=True, mask_land=True,
             background: Optional[xr.Dataset]=None, background_rgb=True,
             title: Optional[str]=None, **kwargs) -> None:
    """
    Plot Rrs (default: 446 nm) for the given dataset.
    Mask land if desired.
    If `projected`, project to PlateCarree; else, just plot in data coordinates.
    """
    # Create figure
    if projected:
        fig, ax = _create_map_figure(1, 1, figsize=(14, 6))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    # Mask
    if mask_land:
        data_to_plot = data.where(data["water"])[col]
    else:
        data_to_plot = data[col]

    # Plot
    kw = {"vmin": 0, "vmax": 0.04, "cmap": batlow} | kwargs
    if projected:
        data_to_plot.plot.pcolormesh(ax=ax, transform=projection, x="lon", y="lat", **kw)
    else:
        data_to_plot.plot.imshow(ax=ax, add_labels=False, yincrease=False, **kw)

    # Plot land if desired
    if background is not None:
        # Check for setting compatibility
        if projected and background_rgb:
            print("Cannot plot a projected RGB background; changing to projected brightness background instead.")
            background_rgb = False
        _plot_land(background, ax, mask_water=mask_land, use_rgb=background_rgb, add_labels=False, yincrease=projected)

    # Plot parameters
    ax.set_title(title)
    ax.grid(False)

    plt.show()
    plt.close()


def plot_IOP_single(data: xr.Dataset, iop=c.aph_443, *,
                    background: Optional[xr.Dataset]=None, background_rgb=True,
                    axs: Optional[Iterable[plt.Axes]]=None,
                    title: Optional[str]=None,
                    saveto: Optional[Path | str]=None, **kwargs) -> None:
    """
    For one IOP (default: aph at 443 nm), plot the mean prediction and % uncertainty.
    """
    # Create new figure if no axs are given
    newfig = (axs is None)
    if newfig:
        fig, axs = _create_map_figure(ncols=2, figsize=(14, 6), layout="constrained")

    # Setup
    norm_mean = LogNorm(vmin=1e-5, vmax=1e1)
    norm_std_pct = Normalize(vmin=c.total_unc_pct.vmin, vmax=c.total_unc_pct.vmax)

    # Plot data
    map_kwargs = {"transform": projection, "x": "lon", "y": "lat", "cmap": "cividis", "robust": True, "rasterized": True}
    data[iop].plot.pcolormesh(ax=axs[0], norm=norm_mean, **map_kwargs)
    data[f"{iop}_std_pct"].plot.pcolormesh(ax=axs[1], norm=norm_std_pct, **map_kwargs)

    # Plot land if desired
    if background is not None:
        for ax in axs:
            _plot_land(background, ax, mask_water=True, use_rgb=background_rgb)

    # Plot parameters
    axs[0].set_title(f"{iop.label}: mean")
    axs[1].set_title(f"{iop.label}: uncertainty [%]")
    if newfig:
        fig.suptitle(title)

    if newfig:
        if saveto:
            plt.savefig(saveto)

        plt.show()
        plt.close()


def plot_IOP_all(data: xr.Dataset, *,
                 iops=c.iops,
                 title: Optional[str]=None,
                 saveto: Optional[Path | str]=None, **kwargs) -> None:
    """
    For all IOPs, plot the mean prediction and % uncertainty.
    """
    # Create figure
    nrows = len(iops)
    fig, axs = _create_map_figure(ncols=2, nrows=nrows, figsize=(14, 6*nrows), layout="constrained")

    # Plot individual rows
    for ax_row, iop in zip(axs, iops):
        plot_IOP_single(data, iop=iop, axs=ax_row, **kwargs)

    # Plot parameters
    fig.suptitle(title)

    if saveto:
        plt.savefig(saveto)

    plt.show()
    plt.close()


def get_h5_filename(filename_nc: Path | str) -> Path:
    """
    For a level-2 .nc file, get the corresponding level-1 .h5 filename.
    Note that this depends entirely on the specific file structure for this project; it does not generalise.
    """
    # Set up prefix
    filename_nc = Path(filename_nc)
    date = filename_nc.stem.split("_")[1:7]  # Standard format is PRISMA_yyyy_mm_dd_hh_mm_ss_...nc
    prefix = "PRS_L1_STD_OFFL_"
    date_prefix = "".join([prefix, *date])

    # Look for matching files
    matching_filenames = filename_nc.parent.glob(f"{date_prefix}*.he5")
    matching_filenames = list(matching_filenames)
    assert len(matching_filenames) == 1, f"Did not find exactly 1 match; instead found {len(matching_filenames)}:\n{matching_filenames}"
    filename_h5 = matching_filenames[0]

    return filename_h5

"""
    def normalize_band(band):
        p2, p98 = np.percentile(band, (2, 98))
        return np.clip((band - p2) / (p98 - p2), 0, 1)
"""

def load_h5_as_rgb(filename: Path | str, *,
                   vnir_cube_address: str="/HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/VNIR_Cube",
                   bands: Iterable[int]=(32, 45, 61),
                   normalise: str="pct") -> np.ndarray:
    """
    Load a PRISMA HDF5 file from the given filename and extract the desired bands for an RGB image.
    """
    # Extract relevant bands from HDF5 file
    with h5py.File(filename, "r") as f:
        vnir_cube = f.get(vnir_cube_address)
        rgb_cube = vnir_cube[:, bands]  # [x, c, y]

    # Reorder to RGB cube
    rgb_cube = np.swapaxes(rgb_cube, 1, 0)  # [c, x, y]
    rgb_cube = np.rot90(rgb_cube, k=-1, axes=(1, 2))  # Rotate image counter-clockwise to match level 2

    # Normalisation
    if normalise == "max":
        # Normalise to [0..1] range
        rgb_max = rgb_cube.max(axis=(1, 2))  # [c]
        rgb_cube = rgb_cube / rgb_max[:, np.newaxis, np.newaxis]

    elif normalise == "pct":
        # Normalise to [2..98] percentile range
        lower, upper = np.percentile(rgb_cube, (2, 98), axis=(1, 2))
        lower, upper = lower[:, np.newaxis, np.newaxis], upper[:, np.newaxis, np.newaxis]
        rgb_cube = np.clip((rgb_cube - lower) / (upper - lower), 0, 1)

    return rgb_cube


def rgb_to_xarray(scene: xr.Dataset, rgb_cube: np.ndarray) -> xr.Dataset:
    """
    Convert a numpy-format RGB cube to an xarray Dataset with variables "r", "g", "b", "brightness", and matching coordinates.
    """
    rgb_dict = {c: (scene.dims, arr) for c, arr in zip("rgb", rgb_cube)}
    brightness = 0.3 * rgb_cube[0] + 0.6 * rgb_cube[1] + 0.1 * rgb_cube[2]
    brightness = {"brightness": (scene.dims, brightness)}

    scene_rgb = rgb_dict | brightness
    scene_rgb = xr.Dataset(scene_rgb, coords=scene.coords)

    # Add mask if available
    if "water" in scene.keys():
        scene_rgb = scene_rgb.assign({"water": scene["water"]})

    return scene_rgb
