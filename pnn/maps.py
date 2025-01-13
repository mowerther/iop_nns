"""
Functions for reading and processing spatial (map) data.
"""
from pathlib import Path

import xarray as xr

from . import constants as c


### CONSTANTS
pattern_prisma_acolite = "PRISMA_*_converted_L2C.nc"
pattern_prisma_l2 = "PRISMA_*_L2W.nc"


### DATA LOADING
def _load_general(filename: Path | str) -> xr.Dataset:
    """
    Load and pre-process NetCDF files.
    """
    data = xr.open_dataset(filename)
    data = data.set_coords(["lon", "lat"])
    return data


def _load_acolite(filename: Path | str) -> xr.Dataset:
    """
    Load ACOLITE-processed data and convert rho_w to R_rs.
    """
    data = _load_general(filename)


def _load_l2(filename: Path | str) -> xr.Dataset:
    """
    Load L2C processed data.
    """
    data = _load_general(filename)


# plt.figure(figsize=(14, 6))
# ax = plt.axes(projection=ccrs.PlateCarree())
# data.Rrs_446.plot.pcolormesh(
#     ax=ax, transform=ccrs.PlateCarree(), x="lon", y="lat", add_colorbar=False
# )
# ax.coastlines()
