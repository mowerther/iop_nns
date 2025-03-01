"""
Functions for reading and processing spatial (map) data.
Currently contains plotting functions which would ideally be moved to pnn.output.
"""
import datetime as dt
from functools import partial
from pathlib import Path
from typing import Hashable, Iterable, Optional

import h5py
import numpy as np
import pandas as pd
import xarray as xr

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.image import AxesImage
from cartopy.crs import PlateCarree
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.geocollection import GeoQuadMesh
from cmcrameri import cm as cmc

from . import constants as c
from . import output as o


### CONSTANTS
pattern_prisma_acolite = "PRISMA_*_converted_L2C.nc"
pattern_prisma_l2 = "PRISMA_*_L2W.nc"

projection = PlateCarree()
kw_projection = {"transform": projection, "x": "lon", "y": "lat"}
kw_xy = {"add_labels": False, "yincrease": False}
kw_cbar = {"location": "bottom", "fraction": 0.1, "pad": 0.05, "extend": "both"}

N_colours = 12
cmap_iop = cmc.navia.resampled(N_colours)
cmap_unc = cmc.turku.resampled(N_colours)
cmap_Rrs = LinearSegmentedColormap.from_list("devon_discrete", cmc.devon.colors[20:-20]).resampled(N_colours)  # Devon without white


### DATA LOADING
def load_map(filename: Path | str) -> xr.Dataset:
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


def select_prisma_columns(data: xr.Dataset, key: Hashable="Rrs") -> xr.Dataset:
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
    data = load_map(filename)

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
    data = load_map(filename)

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
                   iop_labels: Optional[Iterable[Hashable]]=c.iops, mask_land=True) -> xr.Dataset:
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
def _create_map_figure(*args, projected=True, **kwargs) -> tuple[plt.Figure, Iterable[GeoAxes | plt.Axes]]:
    """
    Generate a Figure and Axes, with or without projection setup.
    """
    # Set up subplot kwargs
    subplot_kw = {"projection": projection} if projected else {}
    try:  # Allow the user to pass additional subplot kwargs
        subplot_kw = subplot_kw | kwargs.pop("subplot_kw")
    except KeyError:
        pass

    return plt.subplots(*args, subplot_kw=subplot_kw, layout="constrained", **kwargs)


def _plot_land_RGB(data: xr.Dataset, ax: plt.Axes, **kwargs) -> None:
    """
    Plot RGB layers as a colour image.
    NOT compatible with projections.
    """
    # Combine into RGB cube
    data_rgb = xr.concat([data[c] for c in "rgb"], dim="c")  # data[*"rgb"] or similar does not work

    # Plot
    data_rgb.plot.imshow(ax=ax, **kwargs)


def _plot_land_brightness(data: xr.Dataset, ax: plt.Axes, **kwargs) -> None:
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
    brightness.plot.pcolormesh(ax=ax, cmap=cmc.grayC, add_colorbar=False, **kwargs)


def _plot_with_background(data: xr.Dataset, col: Hashable, ax: GeoAxes | plt.Axes, *,
                          projected=True,
                          mask_land=True, background: Optional[xr.Dataset]=None, background_rgb=False,
                          rasterized=True, **kwargs) -> GeoQuadMesh | AxesImage:
    """
    Plot a variable `col` in the dataset `data` spatially, with masking and a greyscale/RGB background.
    Returns the image of the variable, so it can be re-used (e.g. for colour bars).
    """
    # Set up axes-related kwargs
    kw_ax = kw_projection if projected else kw_xy
    kw_ax = kw_ax | {"ax": ax, "rasterized": rasterized}

    # Plot background (note - unmasked, behind main image)
    if background is not None:
        plot_func = _plot_land_brightness if projected else _plot_land_RGB
        plot_func(background, **kw_ax)

    # Set up mask, plot function according to kwargs
    variable = data.where(data["water"])[col] if mask_land else data[col]
    plot_func = variable.plot.pcolormesh if projected else variable.plot.imshow  # xr.DataArray.plot... does not work

    # Plot data
    kwargs = kw_ax | kwargs
    im = plot_func(**kwargs)

    return im


def _plot_matchups(matchups: pd.DataFrame, ax: GeoAxes,
                   **kwargs) -> None:
    """
    Plot match-ups as scatter points on top of a GeoAxes.
    Only works on projected maps.
    color, cmap, and norm are handled through **kwargs so they can stay optional.
    """
    # Checks
    assert isinstance(ax, GeoAxes), f"Axes object `{ax}` is not a projected GeoAxes, cannot plot match-ups."

    # Determine colours
    ax.scatter(matchups["lon"], matchups["lat"], transform=projection,
               s=25, edgecolor="black", marker="D",
               **kwargs)




def plot_Rrs(data: xr.Dataset, wavelength: int=446, *,
             ax: Optional[plt.Axes]=None, projected=True,
             mask_land=True, background: Optional[xr.Dataset]=None, background_rgb=False,
             matchups: Optional[pd.DataFrame]=None,
             title: Optional[str]=None, **kwargs) -> None:
    """
    Plot Rrs (default: 446 nm) for the given dataset.
    Masking, projection are handled by _plot_with_background.
    """
    # Select data
    col = f"Rrs_{wavelength}"

    # Create new figure if no ax was given
    newfig = (ax is None)
    if newfig:
        fig, ax = _create_map_figure(1, 1, projected=projected, figsize=(8, 6))
    # Get `projected` from ax type?

    # Set up colours
    norm = Normalize(0, 0.02)

    # Plot
    cbar_kwargs = {"label": r"$R_{rs}$" + f"({wavelength}) " + r"[sr$^{-1}$]", "ticks": np.linspace(0, 0.02, N_colours//3+1)} | kw_cbar
    im = _plot_with_background(data, col,
                               ax=ax, projected=projected,
                               mask_land=mask_land, background=background, background_rgb=background_rgb,
                               cmap=cmap_Rrs, norm=norm, cbar_kwargs=cbar_kwargs, **kwargs)

    # Plot match-ups
    if projected and matchups is not None:
        _plot_matchups(matchups, ax=ax, c=matchups[col], cmap=cmap_Rrs, norm=norm)

    # Plot parameters
    if newfig:
        fig.suptitle(title)

    if newfig:
        plt.show()
        plt.close()


def plot_IOP_single(data: xr.Dataset, iop: c.Parameter=c.aph_443, *,
                    uncmin: Optional[float]=None, uncmax: Optional[float]=None,
                    axs: Optional[Iterable[plt.Axes]]=None, projected=True,
                    background: Optional[xr.Dataset]=None, background_rgb=False,
                    matchups: Optional[pd.DataFrame]=None,
                    title: Optional[str]=None,
                    saveto: Optional[Path | str]=None, **kwargs) -> None:
    """
    For one IOP (default: aph at 443 nm), plot the mean prediction and % uncertainty.
    Masking, projection are handled by _plot_with_background.
    """
    # Create new figure if no axs are given
    newfig = (axs is None)
    if newfig:
        fig, axs = _create_map_figure(ncols=2, projected=projected, figsize=(14, 6))

    # Setup cmaps and norms
    unc = f"{iop}_std_pct"

    f = 10**(2/4)  # ticks beyond 1e-2  /  ticks between 1e-2 and 1e-1
    norm_mean = LogNorm(vmin=1e-2/f, vmax=1e0*f)
    norm_unc = Normalize(vmin=uncmin, vmax=uncmax)

    # Plot data
    kw_shared = {"mask_land": False, "background": background, "background_rgb": background_rgb, "robust": True}

    cbar_kwargs = {"label": f"Mean {iop.label} [{iop.unit}]"} | kw_cbar
    _plot_with_background(data, iop, ax=axs[0],
                          norm=norm_mean, cmap=cmap_iop, cbar_kwargs=cbar_kwargs,
                          **kw_shared)

    cbar_kwargs = {"label": f"Uncertainty in {iop.label} [%]"} | kw_cbar
    _plot_with_background(data, unc, ax=axs[1],
                          norm=norm_unc, cmap=cmap_unc, cbar_kwargs=cbar_kwargs, **kw_shared)

    # Plot match-ups
    if projected and matchups is not None:
        _plot_matchups(matchups, ax=axs[0], c=matchups[iop], cmap=cmap_iop, norm=norm_mean)

    # Plot parameters
    if newfig:
        fig.suptitle(title)

    if newfig:
        if saveto:
            plt.savefig(saveto)

        plt.show()
        plt.close()


def plot_IOP_all(data: xr.Dataset, *, iops=c.iops,
                 projected=True,
                 background: Optional[xr.Dataset]=None, background_rgb=False,
                 title: Optional[str]=None,
                 saveto: Optional[Path | str]=None, **kwargs) -> None:
    """
    For all IOPs, plot the mean prediction and % uncertainty.
    """
    # Create figure
    nrows = len(iops)
    fig, axs = _create_map_figure(ncols=2, nrows=nrows, projected=projected, figsize=(14, 6*nrows))

    # Plot individual rows
    for ax_row, iop in zip(axs, iops):
        plot_IOP_single(data, iop=iop, axs=ax_row, projected=projected, background=background, background_rgb=background_rgb, **kwargs)

    # Plot parameters
    fig.suptitle(title)

    if saveto:
        plt.savefig(saveto)

    plt.show()
    plt.close()


def plot_Rrs_and_IOP(reflectance: xr.Dataset, iop_map: xr.Dataset, *, wavelength: int=446, iop: c.Parameter=c.aph_443,
                     projected=True,
                     background: Optional[xr.Dataset]=None, background_rgb=False,
                     title: Optional[str]=None,
                     saveto: Optional[Path | str]=None, **kwargs) -> None:
    """
    Plot Rrs and one IOP (mean/uncertainty) next to each other.
    kwargs are passed to both plot_Rrs and plot_IOP_single.
    """
    # Create figure
    fig, axs = _create_map_figure(ncols=3, nrows=1, projected=projected, figsize=(15, 5))

    # Plot data
    kw_shared = {"projected": projected, "background": background, "background_rgb": background_rgb} | kwargs
    plot_Rrs(reflectance, wavelength, ax=axs[0], **kw_shared)
    plot_IOP_single(iop_map, iop, axs=axs[1:], **kw_shared)

    # Plot parameters
    fig.suptitle(title)

    if saveto:
        plt.savefig(saveto, dpi=600)

    plt.show()
    plt.close()


def plot_Rrs_and_IOP_overview(reflectance: xr.Dataset, iop_map: xr.Dataset, *,
                              projected=True,
                              background: Optional[xr.Dataset]=None, background_rgb=False,
                              title: Optional[str]=None,
                              saveto: Optional[Path | str]=None, **kwargs) -> None:
    """
    Plot all IOPs with their uncertainties and the closest wavelengths.
    Hard-coded for PRISMA.
    """
    # Create figure
    fig, axs = _create_map_figure(ncols=4, nrows=4, projected=projected, figsize=(15, 12))
    axs_rrs = axs[::2, 0]
    axs_empty = axs[1::2, 0]
    axs_iop_443 = axs[:2, 1:]
    axs_iop_675 = axs[2:, 1:]

    # Kwarg setup
    kw_shared = {"projected": projected, "background": background, "background_rgb": background_rgb} | kwargs

    # Plot Rrs
    for ax, wvl in zip(axs_rrs, [446, 674]):
        plot_Rrs(reflectance, wvl, ax=ax, **kw_shared)

    for ax in axs_empty:
        ax.axis("off")

    # Plot IOPs
    for axs_iop_wvl, iop_series in zip([axs_iop_443, axs_iop_675], [c.iops_443, c.iops_675]):
        for axs_iop, iop in zip(axs_iop_wvl.T, iop_series):
            # Plot data
            plot_IOP_single(iop_map, iop, axs=axs_iop, **kw_shared)

            # Labels
            for ax in axs_iop:
                o.label_topleft(ax, iop.label)

    # Plot parameters
    fig.suptitle(title)

    if saveto:
        plt.savefig(saveto, dpi=600)

    plt.show()
    plt.close()


def filename_to_date(filename: Path | str) -> tuple[str, dt.datetime]:
    """
    Take a PRISMA NetCDF filename and extract its date.
    Returns both the string format and a datetime object.
    """
    filename = Path(filename)
    date = filename.stem.split("_")[1:7]  # Standard format is PRISMA_yyyy_mm_dd_hh_mm_ss_...nc

    date_str = "".join(date)
    date_dt = dt.datetime.fromisoformat(date_str[:8])  # Date only for now

    return date_str, date_dt


def get_h5_filename(filename_nc: Path | str, *, prefix: str="PRS_L1_STD_OFFL_") -> Path:
    """
    For a level-2 .nc file, get the corresponding level-1 .h5 filename.
    Note that this depends entirely on the specific file structure for this project; it does not generalise.
    """
    # Set up prefix
    filename_nc = Path(filename_nc)
    date, _ = filename_to_date(filename_nc)
    date_prefix = f"{prefix}{date}"

    # Look for matching files
    matching_filenames = filename_nc.parent.glob(f"{date_prefix}*.he5")
    matching_filenames = list(matching_filenames)
    assert len(matching_filenames) == 1, f"Did not find exactly 1 match; instead found {len(matching_filenames)}:\n{matching_filenames}"
    filename_h5 = matching_filenames[0]

    return filename_h5


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


def _rgb_to_luminance(rgb: Iterable[float]) -> Iterable[float]:
    """
    Convert an RGB array [c, x, ...] to a luminance array [x, ...].
    Note that this is functionality also appears in pnn.output.common; it is repeated here to allow independent development.
    """
    r, g, b = rgb
    return 0.3*r + 0.6*g + 0.1*b


def rgb_to_xarray(scene: xr.Dataset, rgb_cube: np.ndarray) -> xr.Dataset:
    """
    Convert a numpy-format RGB cube to an xarray Dataset with variables "r", "g", "b", "brightness", and matching coordinates.
    """
    rgb_dict = {c: (scene.dims, arr) for c, arr in zip("rgb", rgb_cube)}
    brightness = _rgb_to_luminance(rgb_cube)
    brightness = {"brightness": (scene.dims, brightness)}

    scene_rgb = rgb_dict | brightness
    scene_rgb = xr.Dataset(scene_rgb, coords=scene.coords)

    # Add mask if available
    if "water" in scene.keys():
        scene_rgb = scene_rgb.assign({"water": scene["water"]})

    return scene_rgb


def find_matchups_on_date(matchups: pd.DataFrame, date: dt.datetime, *, col: Hashable="date") -> pd.DataFrame:
    """
    Find rows within `matchups` (column `col`) that are on the same day as `date`.
    First converts `matchups[col]` to Pandas datetimes.
    Currently assumes everything is a day, no hours etc.
    """
    matchups_dt = pd.to_datetime(matchups[col], dayfirst=True)
    matching = matchups[matchups_dt == date]
    return matching
