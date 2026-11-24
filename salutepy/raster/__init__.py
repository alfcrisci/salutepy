"""
salutepy.raster - biometeorological indices of biometeo.js on gridded data (GRIB, NetCDF, GeoTIFF),
computed in parallel.

Typical use::

    from salutepy import raster
    ds = raster.open_raster("run.grb2")                 # GRIB, NetCDF or GeoTIFF
    std = raster.standardize(ds)                         # t (degC), rh (%), wind (m/s), ...
    out = raster.compute(std, ["utci", "pet", "humidex"], workers=8)
    raster.write_netcdf(out, "indices.nc")

Command line: ``python -m salutepy.raster run.grb2 -i utci,pet -o indices.nc -w 8``
"""

from .engine import INDICES, compute, list_indices
from .io import open_raster, open_rasters, write_geotiff, write_netcdf
from .variables import standardize

__all__ = [
    "INDICES",
    "compute",
    "list_indices",
    "open_raster",
    "open_rasters",
    "standardize",
    "write_geotiff",
    "write_netcdf",
]
