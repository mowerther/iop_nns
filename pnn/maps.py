"""
Functions for reading and processing spatial (map) data.
"""
from pathlib import Path

import xarray as xr
from cartopy.crs import PlateCarree

from cmcrameri.cm import batlow as default_cmap
from matplotlib import pyplot as plt

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

    # Select columns PNNs were trained on

    return data


def _load_acolite(filename: Path | str) -> xr.Dataset:
    """
    Load ACOLITE-processed data and convert rho_w to R_rs.
    """
    data = _load_general(filename)

    # Convert rho_w to R_rs; division by pi

    return data

def _load_l2(filename: Path | str) -> xr.Dataset:
    """
    Load L2C processed data.
    """
    data = _load_general(filename)
    return data


### PLOTTING
def plot_Rrs(data: xr.Dataset, **kwargs) -> None:
    """
    Plot Rrs (default: 446 nm) for the given dataset.
    """
    # Create figure
    fig = plt.figure(figsize=(14, 6))
    ax = plt.axes(projection=PlateCarree())

    # Plot data
    data.Rrs_446.plot.pcolormesh(ax=ax, transform=PlateCarree(), x="lon", y="lat", vmin=0, cmap=default_cmap)

    plt.show()
    plt.close()
