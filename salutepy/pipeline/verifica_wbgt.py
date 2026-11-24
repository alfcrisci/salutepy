"""
Ricalcola wbgtsun / wbgtsha / tglob di un NetCDF prodotto da meteosalute_op_dir_mol01.r
partendo dal GRIB2 MOLOCH e confronta i valori.

Nome del NetCDF:  mol01gfs12_YYYYMMDDHH.nc  ->  i = ore da (giorno run + 1) 00 + 1
Passo GRIB usato: run 12 UTC + (10 + i - 1) ore  (cdo -seltimestep,11/83, i-esimo file)

Uso::

    python -m salutepy.pipeline.verifica_wbgt 2026112312_mol01gfs.grb2 mol01gfs12_2026112412.nc
    python -m salutepy.pipeline.verifica_wbgt GRIB NC --time-mode valid    # tempo valido corretto
    python -m salutepy.pipeline.verifica_wbgt GRIB NC -o confronto.nc      # salva anche le differenze
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from .wbgt_grib import wbgt_from_grib

VARS = ("ta2m", "wbgtsun", "wbgtsha", "tglob")


def tempi_da_nomi(grib, nc, first_step=11):
    """(run 12 UTC, indice i dello script R, tempo valido UTC) dai nomi dei file."""
    run = re.search(r"(\d{10})_mol01", Path(grib).name)
    lab = re.search(r"_(\d{10})\.nc$", Path(nc).name)
    if not run or not lab:
        raise ValueError("nomi attesi: YYYYMMDDHH_mol01*.grb2 e *_YYYYMMDDHH.nc")
    run = np.datetime64(f"{run[1][:4]}-{run[1][4:6]}-{run[1][6:8]}T{run[1][8:]}:00", "s")
    lab = np.datetime64(f"{lab[1][:4]}-{lab[1][4:6]}-{lab[1][6:8]}T{lab[1][8:]}:00", "s")
    day1 = run.astype("datetime64[D]").astype("datetime64[s]") + np.timedelta64(1, "D")
    i = int((lab - day1) // np.timedelta64(1, "h")) + 1
    valid = run + np.timedelta64(first_step - 1 + i - 1, "h")
    return run, i, valid


def confronta(grib, nc, time_mode="r", workers=None):
    run, i, valid = tempi_da_nomi(grib, nc)
    py = wbgt_from_grib(grib, valid_time=valid, time_mode=time_mode, workers=workers).isel(valid_time=0)
    r = xr.open_dataset(nc, decode_times=False)
    r = r.isel({d: 0 for d in ("record", "time") if d in r.dims})
    # stessa griglia (lat/lon arrotondati per evitare differenze di virgola mobile)
    py = py.assign_coords(latitude=py.latitude.round(4), longitude=py.longitude.round(4))
    r = r.assign_coords(latitude=r.latitude.round(4), longitude=r.longitude.round(4))
    py, r = xr.align(py, r, join="inner")
    righe = []
    diff = xr.Dataset()
    for v in VARS:
        a, b = py[v].values.astype(float), r[v].values.astype(float)
        ok = np.isfinite(a) & np.isfinite(b) & (b > -999)
        d = a[ok] - b[ok]
        diff[f"d_{v}"] = (("latitude", "longitude"), np.where(ok, a - b, np.nan))
        righe.append((v, ok.sum(), d.mean(), np.abs(d).mean(), np.sqrt((d ** 2).mean()), np.abs(d).max(),
                      100 * (np.abs(d) <= 0.1).mean(), np.corrcoef(a[ok], b[ok])[0, 1]))
    info = dict(run=str(run), i=i, valid_utc=str(valid), time_mode=time_mode)
    return info, righe, diff.assign_coords(latitude=py.latitude, longitude=py.longitude), py, r


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m salutepy.pipeline.verifica_wbgt")
    ap.add_argument("grib")
    ap.add_argument("nc")
    ap.add_argument("--time-mode", default="r", choices=["r", "valid"])
    ap.add_argument("-w", "--workers", type=int)
    ap.add_argument("-o", "--output", help="NetCDF con i campi ricalcolati e le differenze")
    a = ap.parse_args(argv)
    info, righe, diff, py, _ = confronta(a.grib, a.nc, a.time_mode, a.workers)
    print(" ".join(f"{k}={v}" for k, v in info.items()))
    print(f"{'var':8s} {'n':>7s} {'bias':>9s} {'MAE':>9s} {'RMSE':>9s} {'max|d|':>9s} {'%<=0.1':>7s} {'r':>8s}")
    for v, n, b, mae, rmse, mx, p, cc in righe:
        print(f"{v:8s} {n:7d} {b:9.4f} {mae:9.4f} {rmse:9.4f} {mx:9.4f} {p:7.1f} {cc:8.5f}")
    if a.output:
        xr.merge([py[list(VARS)].drop_vars("valid_time", errors="ignore"), diff]).to_netcdf(a.output)
        print(a.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
