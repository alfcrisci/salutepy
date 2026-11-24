"""
Standardisation of the input variables for the indices.

``standardize`` looks for the usual names of GRIB (ecCodes/cfgrib short names), NetCDF (CF,
CMIP, ERA5, ICON, GFS) and GeoTIFF band names, converts the units and returns a Dataset with:

=========  =========================================  ======
name       meaning                                    units
=========  =========================================  ======
t          air temperature (2 m)                      degC
rh         relative humidity                          %
td         dew-point temperature                      degC
wind       wind speed (10 m, or as given)             m/s
pair       air pressure                               hPa
solar      global solar radiation (flux)              W/m2
tmrt       mean radiant temperature                   degC
tg         globe temperature                          degC
=========  =========================================  ======

Missing variables are derived when possible (rh from t and td, td from t and rh, wind from u/v,
pair from msl or the 1013.25 hPa default). Accumulated radiation in J/m2 (e.g. ERA5 ``ssrd``)
is de-accumulated along time only if ``deaccumulate=True``.
"""

from __future__ import annotations

import warnings

import numpy as np
import xarray as xr

from ..indices.biometeo_np import es, rh_from_dewpoint, wind_speed

ALIASES = {
    "t": ["t", "t2m", "2t", "tas", "tair", "ta", "air_temperature", "T_2M", "TMP_2maboveground", "TMP", "temperature"],
    "td": ["td", "d2m", "2d", "dewp", "dpt", "tdps", "TD_2M", "DPT_2maboveground", "dewpoint", "dew_point_temperature"],
    "rh": ["rh", "r2", "2r", "rh2m", "hurs", "relative_humidity", "RELHUM_2M", "RH_2maboveground", "RH", "r"],
    "u": ["u10", "10u", "uas", "U_10M", "UGRD_10maboveground", "u"],
    "v": ["v10", "10v", "vas", "V_10M", "VGRD_10maboveground", "v"],
    "wind": ["wind", "ws", "si10", "10si", "wspd", "sfcWind", "wind_speed", "WIND_10M", "ws10"],
    "pair": ["pair", "sp", "pres", "ps", "surface_pressure", "PS", "PRES_surface"],
    "msl": ["msl", "prmsl", "pmsl", "psl", "PMSL", "PRMSL_meansealevel"],
    "solar": ["solar", "rad", "radiation", "dswrf", "sdswrf", "rsds", "ssrd", "glob", "ghi", "ASOB_S", "SWDOWN",
              "DSWRF_surface"],
    "tmrt": ["tmrt", "mrt", "Tmrt", "mean_radiant_temperature"],
    "tg": ["tg", "globe", "tglobe", "globe_temperature"],
}


def _find(ds: xr.Dataset, key: str, mapping: dict):
    if key in mapping:
        name = mapping[key]
        if name not in ds:
            raise KeyError(f"Variable '{name}' given for '{key}' is not in the dataset")
        return ds[name]
    for name in ALIASES.get(key, []):
        if name in ds.data_vars:
            return ds[name]
    lower = {v.lower(): v for v in ds.data_vars}
    for name in ALIASES.get(key, []):
        if name.lower() in lower:
            return ds[lower[name.lower()]]
    return None


def _units(da) -> str:
    return str(da.attrs.get("units", da.attrs.get("GRIB_units", ""))).strip()


def _to_celsius(da: xr.DataArray) -> xr.DataArray:
    u = _units(da)
    if u in ("K", "kelvin", "Kelvin", "degK") or (u == "" and float(np.nanmean(da.values)) > 150):
        da = da - 273.15
    elif u in ("degF", "F", "fahrenheit"):
        da = (da - 32) * 5 / 9
    return da.assign_attrs(units="degC")


def _to_hpa(da: xr.DataArray) -> xr.DataArray:
    u = _units(da)
    if u in ("Pa", "pascal") or (u == "" and float(np.nanmean(da.values)) > 2000):
        da = da / 100.0
    elif u in ("kPa",):
        da = da * 10.0
    return da.assign_attrs(units="hPa")


def _to_percent(da: xr.DataArray) -> xr.DataArray:
    u = _units(da)
    if u in ("1", "fraction", "0-1") or (u == "" and float(np.nanmax(da.values)) <= 1.5):
        da = da * 100.0
    return da.assign_attrs(units="%")


def _solar(da: xr.DataArray, deaccumulate: bool, time_dim: str | None) -> xr.DataArray:
    u = _units(da)
    if "J" in u:
        if not deaccumulate:
            raise ValueError(
                f"Radiation '{da.name}' is accumulated ({u}): pass deaccumulate=True (difference along "
                "time divided by the interval in seconds) or give a flux in W/m2"
            )
        if time_dim is None:
            raise ValueError("Cannot de-accumulate radiation without a time dimension")
        tt = da[time_dim].values
        dt = np.diff(tt).astype("timedelta64[s]").astype(float)
        diff = da.diff(time_dim) / xr.DataArray(dt, dims=time_dim, coords={time_dim: da[time_dim].values[1:]})
        first = (da.isel({time_dim: [0]}) * np.nan)
        da = xr.concat([first, diff], dim=time_dim).clip(min=0)
    return da.assign_attrs(units="W m-2")


def time_dimension(ds: xr.Dataset) -> str | None:
    """Name of the time dimension of a Dataset (valid_time, step, time), or None."""
    for d in ("valid_time", "time", "step"):
        if d in ds.dims:
            return d
    return None


def valid_times(ds: xr.Dataset):
    """numpy datetime64 array of the valid times along the time dimension (or the scalar time)."""
    if "valid_time" in ds.coords:
        return ds["valid_time"]
    if "time" in ds.coords:
        if "step" in ds.coords and "step" in ds.dims:
            return ds["time"] + ds["step"]
        return ds["time"]
    return None


def lat_lon(ds: xr.Dataset):
    """2D latitude and longitude DataArrays (broadcast on the spatial dimensions)."""
    for la, lo in (("latitude", "longitude"), ("lat", "lon")):
        if la in ds.coords and lo in ds.coords:
            lat, lon = xr.broadcast(ds[la], ds[lo])
            return lat, lon
    raise ValueError("No latitude/longitude coordinates in the dataset")


def standardize(ds: xr.Dataset, mapping: dict | None = None, deaccumulate: bool = False,
                default_pair: float = 1013.25) -> xr.Dataset:
    """
    Return a Dataset with the standard variables (t, rh, td, wind, pair, solar, tmrt, tg) in the
    units used by biometeo.js. ``mapping`` forces the name of a variable, e.g. {"t": "TMP_2m"}.
    Coordinates (time, latitude/longitude or y/x, lat/lon) and the georeferencing attributes are kept.
    """
    mapping = mapping or {}
    found = {k: _find(ds, k, mapping) for k in ALIASES}
    out = {}
    time_dim = time_dimension(ds)
    if found["t"] is not None:
        out["t"] = _to_celsius(found["t"])
    if found["td"] is not None:
        out["td"] = _to_celsius(found["td"])
    if found["rh"] is not None:
        out["rh"] = _to_percent(found["rh"]).clip(0, 100)
    if "rh" not in out and "t" in out and "td" in out:
        out["rh"] = out["t"].copy(data=np.clip(rh_from_dewpoint(out["t"].values, out["td"].values), 0, 100)).assign_attrs(units="%")
    if "td" not in out and "t" in out and "rh" in out:
        # inversion of es() by Newton iterations (Hardy 1998), consistent with rh_from_dewpoint
        e = es(out["t"].values) * out["rh"].values / 100.0
        td = out["t"].values - (100 - out["rh"].values) / 5.0
        for _ in range(20):
            f = es(td) - e
            d = (es(td + 0.01) - es(td - 0.01)) / 0.02
            td = td - f / d
        out["td"] = out["t"].copy(data=td).assign_attrs(units="degC")
    if found["wind"] is not None:
        out["wind"] = found["wind"].assign_attrs(units="m s-1")
    elif found["u"] is not None and found["v"] is not None:
        out["wind"] = found["u"].copy(data=wind_speed(found["u"].values, found["v"].values)).assign_attrs(units="m s-1")
    if found["pair"] is not None:
        out["pair"] = _to_hpa(found["pair"])
    elif found["msl"] is not None:
        out["pair"] = _to_hpa(found["msl"])
        warnings.warn("Surface pressure not found: mean sea level pressure is used", stacklevel=2)
    elif "t" in out:
        out["pair"] = xr.full_like(out["t"], default_pair).assign_attrs(units="hPa")
    if found["solar"] is not None:
        out["solar"] = _solar(found["solar"], deaccumulate, time_dim)
    if found["tmrt"] is not None:
        out["tmrt"] = _to_celsius(found["tmrt"])
    if found["tg"] is not None:
        out["tg"] = _to_celsius(found["tg"])
    if not out:
        raise ValueError(f"No known variable found. Variables in the dataset: {list(ds.data_vars)}")
    std = xr.Dataset({k: v.rename(k) for k, v in out.items()})
    std.attrs.update(ds.attrs)
    return std
