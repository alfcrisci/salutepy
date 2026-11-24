"""
WBGT al sole (wbgtsun) e all'ombra (wbgtsha) direttamente da un GRIB2 MOLOCH, senza lo split cdo.

Riproduce le righe 189-211 di meteosalute_op_dir_mol01.r:

    wind_speed   = sqrt(u^2 + v^2)                    # vento a 10 m
    wind_speed2m = reducewind(wind_speed)             # vento a 2 m
    w = wbgt.Liljegren(T, TD, wind_speed2m, dswrf, dates, lon, lat)
    wbgtsun = w$data
    wbgtsha = 0.7 * w$Tnwb + 0.3 * T                  # Tnwb del calcolo al sole
    tglob   = w$Tg

con il WBGT di Liljegren compilato (indices.wbgt_fast, identico a HeatStress entro 1e-6 degC) e
calcolato in parallelo su tutta la griglia e tutti i passi.

Tempo usato per la posizione del sole (``time_mode``):

* ``"valid"`` (default): l'ora di validità di ogni passo letta dal GRIB, con ``hour=True``
  (angolo zenitale all'ora del dato).
* ``"r"``: come lo script R: data del run alle 00 UTC + (i - 1) ore + 1 s, e ``hour=False``
  (wbgt.Liljegren usa sempre le 12 UTC di quel giorno, perché lo script non passa ``hour``).
  Serve per confrontare i risultati con quelli prodotti in R.

Uso::

    from salutepy.pipeline.wbgt_grib import wbgt_from_grib
    ds = wbgt_from_grib("2026112312_mol01gfs.grb2")          # passi 11..83 come cdo -seltimestep,11/83
    ds.to_netcdf("wbgt.nc")

    python -m salutepy.pipeline.wbgt_grib 2026112312_mol01gfs.grb2 -o wbgt.nc -w 8
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import xarray as xr

from ..indices.misc import reducewind
from ..indices.wbgt_fast import HAVE_NUMBA, wbgt_liljegren
from ..raster.io import open_raster
from ..raster.variables import standardize


def wbgt_from_grib(path, timesteps=(11, 83), time_mode: str = "valid", hour: bool | None = None,
                   workers: int | None = None, solar_time_correction: bool = False,
                   valid_time=None, r_first_step: int = 11) -> xr.Dataset:
    """
    Calcola wbgtsun, wbgtsha, tglob e tnwb (degC) per ogni passo e ogni pixel di un GRIB2 MOLOCH.

    Parameters
    ----------
    path : file GRIB2 (es. ``2026112312_mol01gfs.grb2``) oppure Dataset già aperto con ``open_raster``.
    timesteps : (primo, ultimo) passo, contati da 1 come ``cdo -seltimestep,11/83``; None = tutti.
    time_mode : "valid" (ora di validità, default) oppure "r" (come lo script R, vedi il modulo).
    hour : True usa l'ora del dato per l'angolo zenitale; default True con "valid", False con "r".
    workers : thread di calcolo (default: tutti i processori).
    solar_time_correction : corregge l'angolo orario con longitudine ed equazione del tempo
        (HeatStress non lo fa); default False, come in R.
    valid_time : uno o più tempi validi (UTC) da calcolare; se dato, sostituisce ``timesteps``.
    r_first_step : con time_mode="r", il passo cdo (da 1) che nello script R è i = 1 (default 11).
    """
    if time_mode not in ("valid", "r"):
        raise ValueError("time_mode deve essere 'valid' oppure 'r'")
    if hour is None:
        hour = time_mode == "valid"
    if HAVE_NUMBA and workers:
        import numba

        numba.set_num_threads(min(workers, numba.config.NUMBA_NUM_THREADS))

    ds = path if isinstance(path, xr.Dataset) else open_raster(path)
    if valid_time is not None:
        ds = ds.sel(valid_time=np.atleast_1d(np.asarray(valid_time, dtype="datetime64[ns]")))
    elif "valid_time" in ds.dims and timesteps is not None:
        first, last = timesteps
        ds = ds.isel(valid_time=slice(first - 1, last))
    std = standardize(ds)
    for v in ("t", "td", "wind", "solar"):
        if v not in std:
            raise KeyError(f"Nel GRIB manca la variabile '{v}' (trovate: {list(ds.data_vars)})")

    t = std["t"].transpose("valid_time", ...) if "valid_time" in std["t"].dims else std["t"].expand_dims("valid_time")
    dims = t.dims
    td = std["td"].transpose(*dims).values
    wind2m = reducewind(std["wind"].transpose(*dims).values)
    solar = std["solar"].transpose(*dims).values
    lat, lon = xr.broadcast(std["latitude"], std["longitude"])
    lat2 = np.broadcast_to(lat.transpose("latitude", "longitude").values, t.shape)
    lon2 = np.broadcast_to(lon.transpose("latitude", "longitude").values, t.shape)

    vt = t["valid_time"].values.astype("datetime64[s]")
    if time_mode == "valid":
        times = vt
    else:
        # come lo script R: as.POSIXct(dateboldate) + (i - 1) * 3600 + 1, dove i = 1 e' il passo cdo r_first_step
        run = np.asarray(ds["time"].values).astype("datetime64[s]")
        run_day = run.astype("datetime64[D]").astype("datetime64[s]")
        lead_h = ((vt - run) // np.timedelta64(1, "h")).astype(int)
        i = lead_h - (r_first_step - 1) + 1
        times = run_day + (i - 1).astype("timedelta64[h]").astype("timedelta64[s]") + np.timedelta64(1, "s")
    times3 = np.broadcast_to(times.reshape(-1, 1, 1), t.shape)

    res = wbgt_liljegren(t.values, td, wind2m, solar, times3, lon2, lat2, hour=hour,
                         solar_time_correction=solar_time_correction)

    coords = t.coords
    out = xr.Dataset(
        {
            "wbgtsun": (dims, res["data"], {"units": "degC", "long_name": "WBGT al sole (Liljegren)"}),
            "wbgtsha": (dims, 0.7 * res["Tnwb"] + 0.3 * t.values,
                        {"units": "degC", "long_name": "WBGT all'ombra: 0.7 Tnwb + 0.3 T (come meteosalute_op_dir_mol01.r)"}),
            "tglob": (dims, res["Tg"], {"units": "degC", "long_name": "temperatura del globo (Liljegren)"}),
            "tnwb": (dims, res["Tnwb"], {"units": "degC", "long_name": "temperatura di bulbo umido naturale (Liljegren)"}),
            "ta2m": (dims, t.values, {"units": "degC", "long_name": "temperatura a 2 m"}),
            "wind2m": (dims, wind2m, {"units": "m s-1", "long_name": "vento a 2 m (reducewind del vento a 10 m)"}),
            "td2m": (dims, td, {"units": "degC", "long_name": "temperatura di rugiada a 2 m"}),
            "wind10m": (dims, std["wind"].transpose(*dims).values, {"units": "m s-1", "long_name": "vento a 10 m"}),
            "solar": (dims, solar, {"units": "W m-2", "long_name": "radiazione globale (dswrf)"}),
        },
        coords=coords,
    )
    out.attrs = {
        "source_file": str(ds.attrs.get("source_file", path)) if isinstance(path, xr.Dataset) else str(path),
        "method": "HeatStress::wbgt.Liljegren (salutepy.indices.wbgt_fast)",
        "time_mode": time_mode,
        "hour": str(hour),
        "solar_time_correction": str(solar_time_correction),
    }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m salutepy.pipeline.wbgt_grib",
                                 description="wbgtsun e wbgtsha da un GRIB2 MOLOCH")
    ap.add_argument("grib")
    ap.add_argument("-o", "--output", required=True, help="file NetCDF di uscita")
    ap.add_argument("-w", "--workers", type=int, default=None)
    ap.add_argument("--steps", default="11,83", help="primo,ultimo passo (da 1) oppure 'all'")
    ap.add_argument("--time-mode", default="valid", choices=["valid", "r"])
    args = ap.parse_args(argv)
    steps = None if args.steps == "all" else tuple(int(x) for x in args.steps.split(","))
    ds = wbgt_from_grib(args.grib, timesteps=steps, time_mode=args.time_mode, workers=args.workers)
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(args.output, encoding=enc)
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
