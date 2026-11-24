"""
Produce i NetCDF orari di meteosalute_op_dir_mol01.r (solo il blocco WBGT) da un GRIB2 MOLOCH,
con gli stessi nomi di file, dimensioni (record, latitude, longitude), tempo e fill value di R.

Variabili scritte (nomi come nei NetCDF di R):
    ta2m, urel, v10m, rcorte, wbgtsun, wbgtsha, tglob, temprad

Uso::

    python -m salutepy.pipeline.meteosalute_wbgt_nc 2026112312_mol01gfs.grb2 -d esempio
    python -m salutepy.pipeline.meteosalute_wbgt_nc GRIB -d esempio --time-mode valid -w 8 --compress

* ``--time-mode r`` (default): come lo script operativo (sole alle 12 UTC del giorno del run,
  tempo del NetCDF = giorno run 00 UTC + (i-1) s, mrt_globe con 150 mm = 0.15 m) -> confrontabile con R.
* ``--time-mode valid``: come meteosalute_op_dir_mol01_fix.r (tempo valido UTC, hour=TRUE,
  tempo del NetCDF = tempo valido, mrt_globe con diam=0.05).

Vengono scritti 72 file (passi cdo 11..82, perché il ciclo R salta l'ultimo dei 73 estratti),
e per ognuno si stampa il tempo impiegato.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

from ..indices.biometeo_np import mrt_globe
from ..indices.heatstress_np import dewp2hurs
from ..raster.io import open_raster
from .wbgt_grib import wbgt_from_grib

FILL = -999.9
EPOCH_1992 = np.datetime64("1992-01-01T00:00:00", "s")

ATTRS = {
    "ta2m": "degC", "urel": "%", "v10m": "m/s", "rcorte": "W m-2",
    "wbgtsun": "degC", "wbgtsha": "degC", "tglob": "degC", "temprad": "degC",
}


def run_of(grib, ds) -> np.datetime64:
    m = re.search(r"(\d{10})_mol01", Path(grib).name)
    if m:
        r = m[1]
        return np.datetime64(f"{r[:4]}-{r[4:6]}-{r[6:8]}T{r[8:]}:00", "s")
    return np.asarray(ds["time"].values).astype("datetime64[s]")


def crea_netcdf(grib, outdir="esempio", first_step=11, n_files=72, time_mode="r", workers=None,
                compress=False, prefix="mol01gfs12_", diam=None, verbose=True) -> list[Path]:
    t_start = time.perf_counter()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ds = open_raster(grib)
    run = run_of(grib, ds)
    run_day = run.astype("datetime64[D]").astype("datetime64[s]")
    if diam is None:
        diam = 0.15 if time_mode == "r" else 0.05     # R operativo: 150 mm = 0.15 m (vecchio biometeo.js in mm)
    if verbose:
        print(f"GRIB aperto in {time.perf_counter() - t_start:.1f} s: run {run}, "
              f"{ds.sizes.get('valid_time', 1)} passi, griglia {ds.sizes['latitude']}x{ds.sizes['longitude']}")
    if not n_files:
        n_files = ds.sizes.get("valid_time", 1) - (first_step - 1)
    files = []
    for i in range(1, n_files + 1):
        t0 = time.perf_counter()
        valid = run + np.timedelta64(first_step - 1 + i - 1, "h")
        label = run_day + np.timedelta64(1, "D") + np.timedelta64(first_step - 11 + i - 1, "h")  # = valido + 2 h      # nome come in R
        out = outdir / f"{prefix}{label.astype(object):%Y%m%d%H}.nc"
        w = wbgt_from_grib(ds, valid_time=valid, time_mode=time_mode, workers=workers,
                           r_first_step=first_step).isel(valid_time=0)
        w = w.sortby("latitude", ascending=False)                                  # nord -> sud come R
        t = w.ta2m.values
        fields = {
            "ta2m": t,
            "urel": dewp2hurs(t, w.td2m.values),
            "v10m": w.wind10m.values,
            "rcorte": w.solar.values,
            "wbgtsun": w.wbgtsun.values,
            "wbgtsha": w.wbgtsha.values,
            "tglob": w.tglob.values,
            "temprad": mrt_globe(t, w.tglob.values, w.wind2m.values, diam),
        }
        if time_mode == "r":
            tnc = run_day + np.timedelta64(i - 1, "s")       # riga 161 dello script R
        else:
            tnc = valid
        hours = (tnc - EPOCH_1992) / np.timedelta64(1, "h")
        dims = ("record", "latitude", "longitude")
        nc = xr.Dataset(
            {k: (dims, np.where(np.isfinite(v), v, FILL).astype("float32")[None], {"units": ATTRS[k]})
             for k, v in fields.items()},
            coords={
                "record": ("record", [hours], {"units": "hours since 1992-1-1 00:00:00", "long_name": "record"}),
                "latitude": ("latitude", w.latitude.values.astype(float), {"units": "degrees_north"}),
                "longitude": ("longitude", w.longitude.values.astype(float), {"units": "degrees_east"}),
            },
            attrs={"source": str(grib), "valid_time_utc": str(valid), "time_mode": time_mode,
                   "producer": "salutepy.pipeline.meteosalute_wbgt_nc"},
        )
        enc = {k: {"_FillValue": np.float32(FILL), "dtype": "float32", **({"zlib": True, "complevel": 1} if compress else {})}
               for k in fields}
        enc["record"] = {"dtype": "float64", "_FillValue": None}
        enc["latitude"] = enc["longitude"] = {"_FillValue": None}
        nc.to_netcdf(out, format="NETCDF4", encoding=enc, unlimited_dims=["record"])
        files.append(out)
        if verbose:
            print(f"{i:2d}/{n_files} {out.name}  valido {valid} UTC  {time.perf_counter() - t0:5.1f} s", flush=True)
    if verbose:
        tot = time.perf_counter() - t_start
        print(f"Totale: {len(files)} file in {tot:.1f} s ({tot / max(len(files), 1):.1f} s/file) -> {outdir}")
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m salutepy.pipeline.meteosalute_wbgt_nc")
    ap.add_argument("grib")
    ap.add_argument("-d", "--outdir", default="esempio")
    ap.add_argument("-w", "--workers", type=int)
    ap.add_argument("-n", "--n-files", type=int, default=72, help="numero di file (0 = fino all'ultimo passo)")
    ap.add_argument("--all", action="store_true", help="tutte le ore del GRIB (= --first-step 1 -n 0)")
    ap.add_argument("--first-step", type=int, default=11)
    ap.add_argument("--time-mode", default="r", choices=["r", "valid"])
    ap.add_argument("--compress", action="store_true", help="compressione zlib (file più piccoli)")
    a = ap.parse_args(argv)
    if a.all:
        a.first_step, a.n_files = 1, 0
    crea_netcdf(a.grib, a.outdir, a.first_step, a.n_files, a.time_mode, a.workers, a.compress)
    return 0


if __name__ == "__main__":
    sys.exit(main())
