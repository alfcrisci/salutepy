"""
Parallel computation of the indices on gridded data.

``compute(std, indices, workers=N)`` takes a Dataset prepared by ``standardize`` and returns a
Dataset with one variable per index (same dimensions and coordinates as the air temperature) plus,
when available, the class of the index (``<index>_class``) with Italian and English labels in the
attributes.

Parallelism: the grid (all time steps and pixels) is flattened and split into blocks that are
processed by a pool of processes (``concurrent.futures.ProcessPoolExecutor``). PET is computed
with numba in parallel threads (``workers`` threads) instead of processes.
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from ..indices import biometeo_np as bnp
from .variables import lat_lon, valid_times


@dataclass(frozen=True)
class IndexSpec:
    name: str
    inputs: tuple
    units: str
    long_name: str
    references: str
    classes: str | None = None
    params: dict = field(default_factory=dict)


INDICES = {
    "utci": IndexSpec("utci", ("t", "rh", "wind", "tmrt"), "degC", "Universal Thermal Climate Index",
                      "Broede et al. (2012) Int. J. Biometeorol. 56: 481-494", "utci"),
    "pet": IndexSpec("pet", ("t", "rh", "wind", "tmrt", "pair"), "degC",
                     "Physiological Equivalent Temperature (Hoeppe, VDI 3787)",
                     "Hoeppe (1999) Int. J. Biometeorol. 43: 71-75; VDI 3787 Part 2", "pet"),
    "wbgt": IndexSpec("wbgt", ("t", "td", "wind", "solar", "time", "lat", "lon"), "degC",
                      "Wet Bulb Globe Temperature outdoor (Liljegren, HeatStress)",
                      "Liljegren et al. (2008) J. Occup. Environ. Hyg. 5: 645-655; R package HeatStress",
                      None, {"hour": True, "solar_time_correction": False}),
    "wbgt_bernard": IndexSpec("wbgt_bernard", ("t", "td"), "degC", "Wet Bulb Globe Temperature in shade/indoor (Bernard, HeatStress)",
                              "Bernard & Pourmoghani (1999); Lemke & Kjellstrom (2012) Ind. Health 50: 267-278"),
    "swbgt": IndexSpec("swbgt", ("t", "rh"), "degC", "Simplified WBGT (ABM, Lemke & Kjellstrom constants; HeatStress)",
                       "Lemke & Kjellstrom (2012) Ind. Health 50: 267-278"),
    "wbt_stull": IndexSpec("wbt_stull", ("t", "rh"), "degC", "Wet-bulb temperature (Stull 2011; HeatStress)",
                           "Stull (2011) J. Appl. Meteor. Climatol. 50: 2267-2269"),
    "apparent_temp": IndexSpec("apparent_temp", ("t", "rh", "wind"), "degC", "Apparent temperature without radiation (Steadman 1994; HeatStress)",
                               "Steadman (1994) Aust. Meteor. Mag. 43: 1-16"),
    "discomfort_index": IndexSpec("discomfort_index", ("t", "rh"), "degC", "Thom discomfort index (Giles form; HeatStress discomInd)",
                                  "Giles et al. (1990) Int. J. Biometeorol. 34: 98-104", "thi"),
    "effective_temp": IndexSpec("effective_temp", ("t", "rh", "wind"), "degC", "Normal effective temperature (HeatStress effectiveTemp)",
                                "Gregorczuk & Cena (1967); Li & Chan (2000)", "net"),
    "heatindex_hs": IndexSpec("heatindex_hs", ("t", "rh"), "degC", "Heat index (HeatStress hi, converted from degF to degC)",
                              "Rothfusz (1990) NWS SR 90-23; R package HeatStress", "heatindex"),
    "humidex_hs": IndexSpec("humidex_hs", ("t", "rh"), "degC", "Humidex (HeatStress, Dosseger vapour pressure)",
                            "Masterton & Richardson (1979)", "humidex"),
    "heatindex": IndexSpec("heatindex", ("t", "rh"), "degC", "Heat index (NWS)",
                           "Rothfusz (1990) NWS SR 90-23; Steadman (1979)", "heatindex"),
    "humidex": IndexSpec("humidex", ("t", "rh"), "degC", "Humidex",
                         "Masterton & Richardson (1979) Environment Canada CLI 1-79", "humidex"),
    "net": IndexSpec("net", ("t", "rh", "wind"), "degC", "Normal Effective Temperature",
                     "Li & Chan (2000) Meteorol. Appl. 7: 369-375", "net"),
    "thi": IndexSpec("thi", ("t", "rh"), "degC", "Thom discomfort index (Giles form)",
                     "Thom (1959) Weatherwise 12: 57-61; Giles et al. (1990) Int. J. Biometeorol. 34: 98-104", "thi"),
    "windchill": IndexSpec("windchill", ("t", "wind"), "degC", "Wind chill equivalent temperature",
                           "Osczevski & Bluestein (2005) Bull. Amer. Meteor. Soc. 86: 1453-1458", "windchill"),
    "mrt_globe": IndexSpec("mrt_globe", ("t", "tg", "wind"), "degC", "Mean radiant temperature from globe temperature",
                           "ISO 7726:1998", None, {"diam_globe": 0.15, "emis_globe": 0.95}),
    "mrt_solar": IndexSpec("mrt_solar", ("t", "rh", "solar", "sunelev"), "degC",
                           "Mean radiant temperature from solar radiation",
                           "Jendritzky et al. (1990); mrt_solar_proj of biometeo.js", None,
                           {"albedo": 0.3, "emis_sfc": 0.97, "fdir": 0.8}),
    "vpd": IndexSpec("vpd", ("t", "rh"), "hPa", "Vapour pressure deficit (saturation deficit)",
                     "IAPWS saturation vapour pressure over water/ice; deficitsat of biometeo.js", None,
                     {"unit": "hPa"}),
    "sunelev": IndexSpec("sunelev", ("time", "lat", "lon"), "degree", "Solar elevation",
                         "Blanco-Muriel et al. (2001) Solar Energy 70: 431-441"),
}


def vpd_factor(unit: str) -> float:
    """Divisor from hPa to the requested VPD unit ("hPa" or "kPa")."""
    u = str(unit).strip().lower()
    if u == "hpa":
        return 1.0
    if u == "kpa":
        return 10.0
    raise ValueError("VPD unit must be 'hPa' or 'kPa'")


def list_indices() -> list[str]:
    """Names of the available indices."""
    return list(INDICES)


# ---------------------------------------------------------------------------
# Kernels (module level, so that they can be sent to worker processes)
# ---------------------------------------------------------------------------

def _kernel(name: str, a: dict, params: dict) -> np.ndarray:
    if name == "utci":
        return bnp.utci(a["t"], a["rh"], a["wind"], a["tmrt"])
    if name == "heatindex":
        return bnp.heatindex(a["t"], a["rh"])
    if name == "humidex":
        return bnp.humidex(a["t"], a["rh"])
    if name == "net":
        return bnp.net_index(a["t"], a["rh"], a["wind"])
    if name == "thi":
        return bnp.thi(a["t"], a["rh"])
    if name == "windchill":
        return bnp.windchill(a["t"], a["wind"])
    if name == "mrt_globe":
        return bnp.mrt_globe(a["t"], a["tg"], a["wind"], params.get("diam_globe", 0.15), params.get("emis_globe", 0.95))
    if name == "mrt_solar":
        return bnp.mrt_solar_proj(a["t"], a["rh"], a["solar"], a["sunelev"], params.get("albedo", 0.3),
                                  params.get("emis_sfc", 0.97), params.get("fdir", 0.8))
    if name == "sunelev":
        return bnp.sun_elevation(a["time"], a["lat"], a["lon"])
    if name == "vpd":
        from ..indices.psychrometrics import deficitsat

        return deficitsat(a["t"], a["rh"]) / vpd_factor(params.get("unit", "hPa"))
    if name in ("swbgt", "wbt_stull", "apparent_temp", "discomfort_index", "effective_temp", "heatindex_hs", "humidex_hs"):
        from ..indices import heatstress_np as hs

        if name == "swbgt":
            return hs.swbgt(a["t"], a["rh"])
        if name == "wbt_stull":
            return hs.wbt_stull(a["t"], a["rh"])
        if name == "apparent_temp":
            return hs.apparent_temp(a["t"], a["rh"], a["wind"])
        if name == "discomfort_index":
            return hs.discomfort_index(a["t"], a["rh"])
        if name == "effective_temp":
            return hs.effective_temp(a["t"], a["rh"], a["wind"])
        if name == "heatindex_hs":
            return (hs.hi(a["t"], a["rh"]) - 32) * 5 / 9
        return hs.humidex(a["t"], a["rh"])
    raise KeyError(name)


def _run_block(job):
    name, arrays, params = job
    return _kernel(name, arrays, params)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _template(std: xr.Dataset) -> xr.DataArray:
    for v in ("t", "rh", "wind", "tmrt", "tg", "solar"):
        if v in std:
            return std[v]
    raise ValueError("The dataset has no input variable")


def _full_inputs(std: xr.Dataset, template: xr.DataArray, needed: set, cache: dict) -> dict:
    """Numpy arrays with the full shape of the template for the needed inputs."""
    out = {}
    for k in needed:
        if k in cache:
            out[k] = cache[k]
            continue
        if k in ("lat", "lon"):
            lat, lon = lat_lon(std if ("lat" in std.coords or "latitude" in std.coords) else template.to_dataset())
            arr = (lat if k == "lat" else lon)
            arr = xr.broadcast(arr, template)[0].transpose(*template.dims).values
        elif k == "time":
            vt = valid_times(std)
            if vt is None:
                raise ValueError("No time coordinate: needed for sun position and WBGT (pass time= to compute)")
            arr = xr.broadcast(vt, template)[0].transpose(*template.dims).values.astype("datetime64[ns]")
        else:
            if k not in std:
                raise KeyError(k)
            arr = xr.broadcast(std[k], template)[0].transpose(*template.dims).values.astype(float)
        cache[k] = arr
        out[k] = arr
    return out


def _parallel(name: str, arrays: dict, params: dict, workers: int, min_block: int) -> np.ndarray:
    shape = next(iter(arrays.values())).shape
    flat = {k: np.ascontiguousarray(v).ravel() for k, v in arrays.items()}
    n = next(iter(flat.values())).size
    heavy = False
    if workers <= 1 or (n < min_block and not heavy):
        return _kernel(name, flat, params).reshape(shape)
    nblocks = max(1, min(n, workers * (8 if heavy else 2)))
    bounds = np.linspace(0, n, nblocks + 1).astype(int)
    jobs = [(name, {k: v[bounds[i]:bounds[i + 1]] for k, v in flat.items()}, params) for i in range(nblocks)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(_run_block, jobs))
    return np.concatenate(parts).reshape(shape)


def compute(std: xr.Dataset, indices, workers: int | None = None, classes: bool = True,
            params: dict | None = None, time=None, min_block: int = 500_000) -> xr.Dataset:
    """
    Compute ``indices`` (names from ``list_indices()``) on a standardised Dataset.

    Parameters
    ----------
    std : Dataset from ``standardize``.
    indices : list of index names, or "all".
    workers : number of processes (and numba threads for PET); default: number of CPUs.
    classes : also return ``<index>_class`` (int8, 0 = not computable) with the labels as attributes.
    params : per-index parameters, e.g. {"mrt_globe": {"diam_globe": 0.05}}.
    time : valid time (datetime64) for data without a time coordinate (e.g. a single GeoTIFF).
    min_block : below this number of grid points the vectorised indices are computed in one process.

    If ``tmrt`` is missing, UTCI and PET use the mean radiant temperature from solar radiation
    (``mrt_solar``) when ``solar`` and the time are available, otherwise the air temperature
    (shade / no radiation), with a warning.
    """
    if isinstance(indices, str):
        indices = list(INDICES) if indices == "all" else [i.strip() for i in indices.split(",") if i.strip()]
    unknown = [i for i in indices if i not in INDICES]
    if unknown:
        raise KeyError(f"Unknown indices {unknown}; available: {list_indices()}")
    workers = workers or os.cpu_count() or 1
    params = params or {}
    std = std.copy()
    if time is not None and valid_times(std) is None:
        std = std.assign_coords(time=np.datetime64(time, "ns"))
    template = _template(std)
    cache: dict = {}
    results: dict = {}

    def need_sunelev():
        if "sunelev" not in results:
            arrs = _full_inputs(std, template, {"time", "lat", "lon"}, cache)
            results["sunelev"] = _parallel("sunelev", arrs, {}, workers, min_block)
        return results["sunelev"]

    def tmrt_array():
        if "tmrt" in std:
            return _full_inputs(std, template, {"tmrt"}, cache)["tmrt"]
        if "tmrt_derived" in cache:
            return cache["tmrt_derived"]
        if "solar" in std and valid_times(std) is not None:
            warnings.warn("tmrt not given: computed from solar radiation (mrt_solar)", stacklevel=3)
            arrs = _full_inputs(std, template, {"t", "rh", "solar"}, cache)
            arrs["sunelev"] = need_sunelev()
            p = {**INDICES["mrt_solar"].params, **params.get("mrt_solar", {})}
            cache["tmrt_derived"] = _parallel("mrt_solar", arrs, p, workers, min_block)
        else:
            warnings.warn("tmrt and solar radiation not given: tmrt = air temperature", stacklevel=3)
            cache["tmrt_derived"] = _full_inputs(std, template, {"t"}, cache)["t"]
        return cache["tmrt_derived"]

    for name in indices:
        spec = INDICES[name]
        p = {**spec.params, **params.get(name, {})}
        if name == "sunelev":
            need_sunelev()
            continue
        needed = set(spec.inputs) - {"tmrt", "sunelev"}
        missing = [k for k in needed if k not in ("time", "lat", "lon") and k not in std]
        if missing:
            raise KeyError(f"Index '{name}' needs {missing}, not found in the dataset")
        arrs = _full_inputs(std, template, needed, cache)
        if "tmrt" in spec.inputs:
            arrs["tmrt"] = tmrt_array()
        if "sunelev" in spec.inputs:
            arrs["sunelev"] = need_sunelev()
        if name in ("pet", "wbgt", "wbgt_bernard"):
            # compiled (numba) kernels, parallel in threads
            from ..indices import pet as petmod
            from ..indices import wbgt_fast

            if petmod.HAVE_NUMBA:
                import numba

                numba.set_num_threads(min(workers, numba.config.NUMBA_NUM_THREADS))
            if name == "pet":
                results[name] = petmod.pet_hoppe(arrs["t"], arrs["rh"], arrs["wind"], arrs["tmrt"], arrs["pair"])
            elif name == "wbgt":
                results[name] = wbgt_fast.wbgt_liljegren(arrs["t"], arrs["td"], arrs["wind"], arrs["solar"], arrs["time"],
                                                         arrs["lon"], arrs["lat"], hour=p["hour"],
                                                         solar_time_correction=p["solar_time_correction"])["data"]
            else:
                results[name] = wbgt_fast.wbgt_bernard(arrs["t"], arrs["td"])["data"]
        else:
            results[name] = _parallel(name, arrs, p, workers, min_block)

    out = xr.Dataset(coords=template.coords)
    for name in indices:
        spec = INDICES[name]
        da = xr.DataArray(results[name], dims=template.dims, coords=template.coords, name=name)
        units = spec.units
        if name == "vpd":
            units = "kPa" if vpd_factor({**spec.params, **params.get(name, {})}["unit"]) == 10.0 else "hPa"
        da.attrs = {"units": units, "long_name": spec.long_name, "references": spec.references,
                    "source": "salutepy.raster (port of biometeo.js, rBiometeo 0.4)"}
        out[name] = da
        if classes and spec.classes:
            cfun = {"utci": bnp.utci_class, "pet": bnp.pet_class, "heatindex": bnp.heatindex_class,
                    "humidex": bnp.humidex_class, "net": bnp.net_class, "thi": bnp.thi_class,
                    "windchill": bnp.windchill_class}[spec.classes]
            c = cfun(results[name])
            c = np.where(np.isnan(c), 0, c).astype("int8")
            labels_it = bnp.class_labels(spec.classes, "it")
            labels_en = bnp.class_labels(spec.classes, "en")
            cda = xr.DataArray(c, dims=template.dims, coords=template.coords, name=f"{name}_class")
            cda.attrs = {"long_name": f"class of {spec.long_name}", "flag_values": np.arange(1, len(labels_en) + 1, dtype="int8"),
                         "flag_meanings": " ".join(s.replace(" ", "_").replace(";", "").replace(":", "")
                                                   for s in labels_en),
                         "labels_it": "; ".join(f"{i + 1}={s}" for i, s in enumerate(labels_it)),
                         "labels_en": "; ".join(f"{i + 1}={s}" for i, s in enumerate(labels_en)),
                         "missing_value_meaning": "0 = not computable"}
            out[f"{name}_class"] = cda
    for k in ("transform", "crs", "source_file"):
        if k in std.attrs:
            out.attrs[k] = std.attrs[k]
    return out
