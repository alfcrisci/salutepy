"""
Reading and writing gridded data: GRIB/GRIB2 (cfgrib), NetCDF (xarray) and GeoTIFF (rasterio).

All readers return an ``xarray.Dataset``. Spatial dimensions are kept as in the file
(``latitude``/``longitude`` for GRIB and most NetCDF, ``y``/``x`` for GeoTIFF); for GeoTIFF the
affine transform and the CRS are stored in the attributes ``transform`` and ``crs`` and 2D
``lat``/``lon`` coordinates are added, so that sun position and WBGT can be computed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

GRIB_EXT = {".grb", ".grib", ".grb2", ".grib2", ".gb2"}
NETCDF_EXT = {".nc", ".nc4", ".netcdf", ".cdf"}
TIFF_EXT = {".tif", ".tiff", ".gtiff"}


def _kind(path: Path) -> str:
    suffixes = [s.lower() for s in path.suffixes]
    for s in reversed(suffixes):
        if s in GRIB_EXT:
            return "grib"
        if s in NETCDF_EXT:
            return "netcdf"
        if s in TIFF_EXT:
            return "geotiff"
    raise ValueError(f"Unknown raster format for {path} (expected GRIB, NetCDF or GeoTIFF)")


def open_grib(path, filter_by_keys: dict | None = None, **kwargs) -> xr.Dataset:
    """
    Open a GRIB/GRIB2 file with cfgrib. A GRIB file often contains several "hypercubes" (e.g. 2 m
    and 10 m fields): all of them are read and merged; the vertical coordinates that differ
    (heightAboveGround, surface, ...) are dropped so that t2m, d2m, u10, v10, ... end up in one
    Dataset. Use ``filter_by_keys`` (e.g. ``{"typeOfLevel": "heightAboveGround"}``) to select.
    """
    import cfgrib

    backend = {"indexpath": ""}
    if filter_by_keys:
        backend["filter_by_keys"] = filter_by_keys
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        parts = cfgrib.open_datasets(str(path), backend_kwargs=backend, **kwargs)
    level_coords = {"heightAboveGround", "surface", "meanSea", "isobaricInhPa", "level", "depthBelowLandLayer",
                    "entireAtmosphere", "nominalTop"}
    cleaned = []
    for p in parts:
        drop = [c for c in p.coords if c in level_coords and p[c].ndim == 0]
        p = p.drop_vars(drop + (["valid_time"] if "valid_time" in p.coords else []))
        # the analysis (step 0) and the forecast steps come as separate hypercubes: give all a step dimension
        if "step" in p.coords and "step" not in p.dims:
            p = p.expand_dims("step")
        cleaned.append(p)
    ds = xr.merge(cleaned, compat="no_conflicts", join="outer", combine_attrs="override")
    if "step" in ds.dims and "time" in ds.coords and ds["time"].ndim == 0:
        ds = ds.assign_coords(valid_time=ds["time"] + ds["step"])
        ds = ds.swap_dims({"step": "valid_time"})
    ds.attrs["source_file"] = str(path)
    return ds


def open_geotiff(path, names: list[str] | None = None, units: dict | None = None) -> xr.Dataset:
    """
    Open a GeoTIFF with rasterio: each band becomes a variable, named after ``names`` (in band
    order), the band description, or ``band<i>``. ``units`` (dict name -> units) sets the units
    attribute (e.g. {"t": "degC"}), used by standardize() to convert Kelvin.
    """
    import rasterio
    from rasterio.warp import transform as warp_transform

    with rasterio.open(path) as src:
        data = src.read().astype(float)
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        h, w = src.height, src.width
        tr = src.transform
        crs = src.crs
        descr = list(src.descriptions)
    cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    xs = tr.c + cols * tr.a + rows * tr.b
    ys = tr.f + cols * tr.d + rows * tr.e
    x1d = xs[0, :]
    y1d = ys[:, 0]
    if crs is not None and not crs.is_geographic:
        lon, lat = warp_transform(crs, "EPSG:4326", xs.ravel(), ys.ravel())
        lon = np.asarray(lon).reshape(h, w)
        lat = np.asarray(lat).reshape(h, w)
    else:
        lon, lat = xs, ys
    variables = {}
    for i in range(data.shape[0]):
        if names and i < len(names):
            name = names[i]
        elif descr[i]:
            name = descr[i]
        else:
            name = f"band{i + 1}"
        da = xr.DataArray(data[i], dims=("y", "x"))
        if units and name in units:
            da.attrs["units"] = units[name]
        variables[name] = da
    ds = xr.Dataset(variables, coords={"y": y1d, "x": x1d, "lat": (("y", "x"), lat), "lon": (("y", "x"), lon)})
    ds.attrs["transform"] = tuple(tr)[:6]
    ds.attrs["crs"] = crs.to_wkt() if crs is not None else ""
    ds.attrs["source_file"] = str(path)
    return ds


def open_raster(path, **kwargs) -> xr.Dataset:
    """Open a GRIB, NetCDF or GeoTIFF file (chosen from the extension) as an xarray Dataset."""
    path = Path(path)
    kind = _kind(path)
    if kind == "grib":
        return open_grib(path, **kwargs)
    if kind == "geotiff":
        return open_geotiff(path, **kwargs)
    ds = xr.open_dataset(path, **kwargs)
    ds.attrs.setdefault("source_file", str(path))
    return ds


def open_rasters(files: dict, units: dict | None = None) -> xr.Dataset:
    """
    Open several single-variable files and merge them, e.g.
    ``open_rasters({"t": "tair.tif", "rh": "rh.tif", "tmrt": "mrt.tif"})``. The key is the name given
    to the first variable (or band) of each file. ``units`` sets the units of GeoTIFF variables.
    """
    parts = []
    for name, path in files.items():
        p = Path(path)
        if _kind(p) == "geotiff":
            full = open_geotiff(p, names=[name], units=units)
            ds = full[[name]]
            ds.attrs.update(full.attrs)
        else:
            full = open_raster(p)
            first = list(full.data_vars)[0]
            ds = full[[first]].rename({first: name})
        parts.append(ds)
    out = xr.merge(parts, compat="override", join="override", combine_attrs="override")
    return out


def write_netcdf(ds: xr.Dataset, path, compress: bool = True) -> Path:
    """Write a Dataset to NetCDF4, with zlib compression of the numeric variables."""
    path = Path(path)
    ds = ds.copy()
    for k in ("transform",):
        if k in ds.attrs:
            ds.attrs[k] = list(ds.attrs[k])
    encoding = {}
    if compress:
        for v in ds.data_vars:
            if np.issubdtype(ds[v].dtype, np.number):
                encoding[v] = {"zlib": True, "complevel": 4}
    ds.to_netcdf(path, encoding=encoding)
    return path


def _grid_transform(ds: xr.Dataset):
    """Affine transform and CRS for a Dataset: from the attributes (GeoTIFF) or a regular lat/lon grid."""
    from rasterio.transform import Affine

    if "transform" in ds.attrs:
        t = list(ds.attrs["transform"])
        return Affine(*t[:6]), ds.attrs.get("crs") or None
    for ydim, xdim in (("latitude", "longitude"), ("lat", "lon"), ("y", "x")):
        if ydim in ds.coords and xdim in ds.coords and ds[ydim].ndim == 1 and ds[xdim].ndim == 1:
            y = ds[ydim].values
            x = ds[xdim].values
            dx = float(x[1] - x[0])
            dy = float(y[1] - y[0])
            return Affine(dx, 0, float(x[0]) - dx / 2, 0, dy, float(y[0]) - dy / 2), "EPSG:4326"
    raise ValueError("Cannot derive the georeferencing: no transform attribute and no regular lat/lon grid")


def write_geotiff(da: xr.DataArray, path, nodata: float = -9999.0) -> Path:
    """
    Write a 2D DataArray, or a 3D one (one band per element of the first dimension, e.g. time),
    to a GeoTIFF. Georeferencing comes from the ``transform``/``crs`` attributes or from a regular
    lat/lon grid.
    """
    import rasterio

    path = Path(path)
    arr = np.asarray(da.values, dtype="float32")
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError("write_geotiff needs a 2D or 3D DataArray")
    ref = da.to_dataset(name="v") if isinstance(da, xr.DataArray) else da
    ref.attrs.update(da.attrs)
    transform, crs = _grid_transform(ref)
    if transform.e > 0:  # latitude increasing (south-up grid, e.g. GRIB/cfgrib): flip to north-up
        from rasterio.transform import Affine

        arr = arr[:, ::-1, :]
        transform = Affine(transform.a, transform.b, transform.c, transform.d, -transform.e,
                           transform.f + transform.e * arr.shape[1])
    arr = np.where(np.isnan(arr), nodata, arr)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
                       dtype="float32", crs=crs, transform=transform, nodata=nodata, compress="deflate") as dst:
        dst.write(arr)
        if da.ndim == 3:
            first = da.dims[0]
            for i, v in enumerate(da[first].values):
                dst.set_band_description(i + 1, f"{da.name}_{first}_{v}")
        else:
            dst.set_band_description(1, str(da.name))
    return path
