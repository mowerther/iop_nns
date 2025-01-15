"""
Functions for reading and processing spatial (map) data.
"""
from pathlib import Path

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


### REFLECTANCE
def select_prisma_columns(data: xr.Dataset) -> xr.Dataset:
    # Select columns PNNs were trained on
    # Note that there are rounding differences between the data and pnn.constants -- we simply select on the min and max
    lmin, lmax = c.wavelengths_prisma[0], c.wavelengths_prisma[-1]
    def column_in_scope(column_name: str) -> bool:
        is_Rrs = ("Rrs_" in column_name)
        if is_Rrs:
            wavelength = float(column_name.split("_")[1])
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
    data /= np.pi

    return data


def _load_l2(filename: Path | str) -> xr.Dataset:
    """
    Load L2C processed data.
    """
    data = _load_general(filename)
    data = select_prisma_columns(data)
    return data


### PLOTTING
def plot_Rrs(data: xr.Dataset, *, col: str="Rrs_446", **kwargs) -> None:
    """
    Plot Rrs (default: 446 nm) for the given dataset.
    """
    # Create figure
    fig = plt.figure(figsize=(14, 6))
    ax = plt.axes(projection=PlateCarree())

    # Plot data
    data[col].plot.pcolormesh(ax=ax, transform=PlateCarree(), x="lon", y="lat", vmin=0, cmap=default_cmap)

    plt.show()
    plt.close()
