"""
Functions for reading and processing spatial (map) data.
"""
from functools import partial
from pathlib import Path
from typing import Iterable, Optional

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


def _load_acolite(filename: Path | str) -> xr.Dataset:
    """
    Load ACOLITE-processed data and convert rho_w to R_rs.
    """
    data = _load_general(filename)

    # Convert rho_w to R_rs
    data_Rrs = select_prisma_columns(data, key="rhos_l2c")
    data_Rrs = data_Rrs / np.pi
    renamer = {rhos: f"Rrs_{rhos[9:]}" for rhos in data_Rrs.keys()}
    data_Rrs = data_Rrs.rename(renamer)

    return data_Rrs


def _load_l2c(filename: Path | str) -> xr.Dataset:
    """
    Load L2C processed data.
    """
    data = _load_general(filename)
    data_Rrs = select_prisma_columns(data, key="Rrs")
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
def map_to_spectra(data: xr.Dataset) -> np.ndarray:
    """
    Extract the spectrum from each pixel in an (x, y)-shaped image into an (x * y)-length array of spectra.
    Note that this loses information on variable names, coordinates, etc, so take care to keep this around elsewhere.
    """
    data_as_numpy = data.to_array().values
    map_shape = data_as_numpy.shape
    data_as_numpy = data_as_numpy.reshape((map_shape[0], -1))
    data_as_numpy = data_as_numpy.T
    return data_as_numpy, map_shape


def _list_to_dataset_shape(data_list: np.ndarray, reference_scene: xr.Dataset) -> np.ndarray:
    """
    Convert a list into a map corresponding to the dimensions of a given xarray Dataset.
    The first dimension of data_list is converted into 2D spatial dimensions.
    The list is first transposed so that other variables become indices in the result.
    """
    new_shape = tuple(reference_scene.sizes.values())
    data_as_map = data_list.T.reshape(-1, *new_shape)
    return data_as_map


def spectra_to_map(data: np.ndarray, map_shape: tuple[int] | xr.Dataset) -> np.ndarray | xr.Dataset:
    """
    Reshape a list of spectra back into a pre-defined map shape.
    If `map_shape` is an xarray Dataset, copy its georeferencing etc.
    """
    if isinstance(map_shape, xr.Dataset):
        data_as_map = _list_to_dataset_shape(data, map_shape)
        data_as_dict = {var: (map_shape.dims, arr) for var, arr in zip(map_shape.variables, data_as_map)}
        new_scene = xr.Dataset(data_as_dict, coords=map_shape.coords)
        data_as_map = new_scene

    elif isinstance(map_shape, tuple):
        data_as_map = data.T.reshape(map_shape)

    return data_as_map


def create_iop_map(iop_mean: np.ndarray, iop_variance: np.ndarray, reference_scene: xr.Dataset, *,
                   iop_labels: Optional[Iterable[c.Parameter]]=c.iops) -> xr.Dataset:
    """
    Convert IOP estimates (mean and variance -> uncertainty) into an xarray Dataset like a provided scene.
    """
    # Reshape to 2D
    iop_mean = _list_to_dataset_shape(iop_mean, reference_scene)
    iop_variance = _list_to_dataset_shape(iop_variance, reference_scene)

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

def plot_Rrs(data: xr.Dataset, *, col: str="Rrs_446",
             title: Optional[str]=None, **kwargs) -> None:
    """
    Plot Rrs (default: 446 nm) for the given dataset.
    """
    # Create figure
    fig, ax = _create_map_figure(1, 1, figsize=(14, 6))

    # Plot data
    data[col].plot.pcolormesh(ax=ax, transform=projection, x="lon", y="lat", vmin=0, vmax=0.10, cmap=batlow)

    # Plot parameters
    ax.set_title(title)

    plt.show()
    plt.close()


def plot_IOP_single(data: xr.Dataset, iop=c.aph_443, *,
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
        plot_IOP_single(data, iop=iop, axs=ax_row)

    # Plot parameters
    fig.suptitle(title)

    if saveto:
        plt.savefig(saveto)

    plt.show()
    plt.close()
