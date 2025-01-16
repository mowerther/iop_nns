"""
Functions for reading and processing spatial (map) data.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

from matplotlib import pyplot as plt
from cartopy.crs import PlateCarree
from cmcrameri.cm import batlow as default_cmap

from . import constants as c


### CONSTANTS
pattern_prisma_acolite = "PRISMA_*_converted_L2C.nc"
pattern_prisma_l2 = "PRISMA_*_L2W.nc"


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


### PLOTTING
def plot_Rrs(data: xr.Dataset, *, col: str="Rrs_446",
             title: Optional[str]=None, **kwargs) -> None:
    """
    Plot Rrs (default: 446 nm) for the given dataset.
    """
    # Create figure
    fig = plt.figure(figsize=(14, 6))
    ax = plt.axes(projection=PlateCarree())

    # Plot data
    data[col].plot.pcolormesh(ax=ax, transform=PlateCarree(), x="lon", y="lat", vmin=0, cmap=default_cmap)

    # Plot parameters
    ax.set_title(title)

    plt.show()
    plt.close()
