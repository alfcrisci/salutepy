"""
I NetCDF orari completi di meteosalute_op_dir_mol01.r (42 variabili) da un GRIB2 MOLOCH.

Stessi nomi di file, variabili, dimensioni (record, latitude, longitude), tempo e fill -999.9
dei NetCDF operativi prodotti in R. Ogni indice è calcolato su tutta la griglia in modo
vettoriale/compilato (numba) invece che pixel per pixel.

Uso::

    python -m salutepy.pipeline.meteosalute_nc 2026112312_mol01gfs.grb2 -d esempio
    python -m salutepy.pipeline.meteosalute_nc GRIB -d esempio --time-mode valid -w 8 --compress
    python -m salutepy.pipeline.meteosalute_nc GRIB -d esempio -n 3          # solo i primi 3 file

``--time-mode r`` (default) riproduce lo script operativo, comprese le sue particolarità:

* sole (WBGT e Steadman) calcolato con giorno del run 00 UTC + (i-1) h + 1 s; WBGT con hour=FALSE;
* tempo del NetCDF = giorno del run 00 UTC + (i-1) secondi;
* mrt_globe con 150 mm (= 0.15 m, come il biometeo.js usato in produzione);
* clomin/clomax con il metabolismo del primo pixel per tutta la griglia (il wrapper R passa M[0]).

``--time-mode valid`` usa invece il tempo valido UTC per il sole e per il NetCDF, hour=TRUE
e il metabolismo di ogni pixel per clomin/clomax (mrt_globe sempre con 0.15 m).

Il file mol01gfs12_2026112412.nc prodotto in R è riprodotto entro 0.01 su tutte le variabili
(arrotondamenti a 2 decimali), tranne una manciata di pixel sui limiti di classe.
"""

from __future__ import annotations

import argparse
import sys
import time
from importlib import resources
from pathlib import Path

import numpy as np
import xarray as xr

from ..indices import biometeo_np as B
from ..indices import comfort_fast as CF
from ..indices import compat as R
from ..indices.misc import metabolism, windchill_class_wk
from ..indices.sun_position import sunposition_rbiometeo
from ..raster.io import open_raster
from .meteosalute_wbgt_nc import EPOCH_1992, FILL, run_of
from .wbgt_grib import wbgt_from_grib

# ordine e unità delle variabili come in nc_create() dello script R
VARIABLES = [
    ("ta2m", "degC"), ("urel", "%"), ("v10m", "m/s"), ("gust", "m/s"), ("pmsl", "hPa"), ("pvap", "hPa"),
    ("defsat", "hPa"), ("csky", "dimensionless"), ("pmvsha", "dimensionless"), ("clomin", "dimensionless"),
    ("clomax", "dimensionless"), ("utci", "degC"), ("utcisha", "degC"), ("utciclasha", "dimensionless"),
    ("stosun", "degC"), ("stosha", "degC"), ("stoind", "degC"), ("temprad", "degC"), ("rcorte", "W m-2"),
    ("stosuncla", "dimensionless"), ("stoshacla", "dimensionless"), ("utcicla", "dimensionless"),
    ("wbgtsha", "degC"), ("wbgtsun", "degC"), ("tglob", "degC"), ("rkcla_int", "class"),
    ("rkcla_nint", "class"), ("rkshacla_int", "class"), ("rkshacla_nint", "class"), ("lprodsha", "perc"),
    ("lprod", "perc"), ("wbgttiwtlsha", "perc"), ("wbgttwtlsha", "hours"), ("wbgttiwtlsun", "perc"),
    ("wbgttwtlsun", "hours"), ("airtiwtl", "perc"), ("airtwtl", "hours"), ("windchill", "degC"),
    ("windcla", "dimensionless"), ("frostime", "minutes"), ("thi", "degC"), ("lapr", "dimensionless"),
]

GRIB_EXTRA = {"gust": ["i10fg", "gust", "fg10", "10fg"], "pmsl": ["pmsl", "msl", "prmsl"],
              "csky": ["clct", "tcc", "N_atm"], "lapr": ["lapr"]}


def load_fillrk(path=None) -> np.ndarray:
    """Vettore fillrkmoloch (1 ovunque, 2 nei pixel che evitano il bug GrADS della classe unica)."""
    if path is None:
        with resources.files("salutepy.data").joinpath("fillrkmoloch02.npz").open("rb") as f:
            return np.load(f)["fill"].astype(float)
    path = Path(path)
    if path.suffix == ".rds":
        import pyreadr

        return np.asarray(list(pyreadr.read_r(str(path)).values())[0].iloc[:, 0], dtype=float)
    return np.load(path)["fill"].astype(float)


def _extra(raw, key):
    for name in GRIB_EXTRA[key]:
        if name in raw.data_vars:
            return np.asarray(raw[name].values, dtype=float).ravel()
    return np.full(raw.sizes["latitude"] * raw.sizes["longitude"], np.nan)


def compute_step(ds, run, i, time_mode="r", first_step=11, fill=None, workers=None):
    """Tutte le 42 variabili (vettori in ordine R: nord->sud, ovest->est) per l'i-esimo file."""
    run_day = run.astype("datetime64[D]").astype("datetime64[s]")
    valid = run + np.timedelta64(first_step - 1 + i - 1, "h")
    w = wbgt_from_grib(ds, valid_time=valid, time_mode=time_mode, workers=workers,
                       r_first_step=first_step).isel(valid_time=0).sortby("latitude", ascending=False)
    raw = ds.sel(valid_time=valid).sortby("latitude", ascending=False)
    f = lambda a: np.asarray(a, dtype=float).ravel()  # noqa: E731
    T, TD, ws, w2, sol = f(w.ta2m), f(w.td2m), f(w.wind10m), f(w.wind2m), f(w.solar)
    lat2, lon2 = np.meshgrid(w.latitude.values, w.longitude.values, indexing="ij")
    if fill is None or fill.size != T.size:
        fill = np.ones(T.size)

    rh = R.dewp2hurs(T, TD)
    if time_mode == "r":
        tsun = run_day + np.timedelta64(i - 1, "h") + np.timedelta64(1, "s")
    else:
        tsun = valid
    diam = 0.15  # globo da 150 mm, come i NetCDF operativi
    elev = sunposition_rbiometeo(tsun, f(lat2), f(lon2))
    tglob, wsun, wsha = f(w.tglob), f(w.wbgtsun), f(w.wbgtsha)
    mrt = B.mrt_globe(T, tglob, w2, diam)

    met = metabolism(T) / 1.8
    met = np.where(met > 58.1 * 1.1, met, 70.0)
    met_clo = met[0] if time_mode == "r" else met

    o = {"ta2m": T, "urel": rh, "v10m": ws, "gust": _extra(raw, "gust"), "pmsl": _extra(raw, "pmsl") / 100,
         "csky": _extra(raw, "csky"), "lapr": _extra(raw, "lapr"), "rcorte": sol,
         "wbgtsun": wsun, "wbgtsha": wsha, "tglob": tglob, "temprad": mrt}
    o["pvap"] = R.pvap(T, rh)
    o["defsat"] = R.deficitsat(T, rh)
    o["pmvsha"] = CF.pmv_iso7730(T, rh, ws, T, met, 0.0, 1.1)
    o["clomin"] = CF.clomin_7730(T, rh, ws, T, met_clo, 0.0)
    o["clomax"] = CF.clomax_7730(T, rh, ws, T, met_clo, 0.0)
    u = B.utci(T, rh, ws, mrt)
    ush = B.utci(T, rh, ws, T)
    o["utci"], o["utcisha"] = u, ush
    o["utcicla"], o["utciclasha"] = B.utci_class(u), B.utci_class(ush)
    stosun = np.asarray(R.steadman_outdoor_sun(T, rh, ws, sol, elev), dtype=float)
    stosha = np.asarray(R.steadman_outdoor_shade(T, rh, ws), dtype=float)
    o["stoind"] = R.steadman_indoor(T, rh)
    o["stosuncla"] = R.steadman_class(stosun)
    o["stoshacla"] = R.steadman_class(stosha)
    o["stosun"] = np.where(np.isnan(stosun), stosha, stosun)
    o["stosha"] = stosha
    two = fill == 2
    o["rkcla_int"] = np.where(two, 2, R.heat_risk_level(wsun, 0, 23.5))
    o["rkcla_nint"] = np.where(two, 2, R.heat_risk_level(wsun, 0, 25.3))
    o["rkshacla_int"] = np.where(two, 2, R.heat_risk_level(wsha, 0, 23.5))
    o["rkshacla_nint"] = np.where(two, 2, R.heat_risk_level(wsha, 0, 25.3))
    o["lprod"], o["lprodsha"] = R.lost_productivity(wsun), R.lost_productivity(wsha)
    o["wbgttiwtlsha"], o["wbgttwtlsha"] = R.mean_wbgtTI_WTL(wsha), R.mean_wbgtWTL(wsha)
    o["wbgttiwtlsun"], o["wbgttwtlsun"] = R.mean_wbgtTI_WTL(wsun), R.mean_wbgtWTL(wsun)
    o["airtiwtl"], o["airtwtl"] = R.mean_airTI_WTL(T), R.mean_airTWTL(T)
    wc = np.asarray(R.windchill(T, w2), dtype=float)
    wc = np.where(wc > 10, FILL, wc)
    wcc = np.asarray(windchill_class_wk(wc), dtype=float)
    wcc = np.where(wcc > 5.1, FILL, wcc)
    if np.sum(wcc > 1) < 2:
        wcc = fill
    o["windchill"], o["windcla"] = wc, wcc
    o["frostime"] = B.frostime(T, w2)
    o["thi"] = B.thi(T, rh)
    tnc = run_day + np.timedelta64(i - 1, "s") if time_mode == "r" else valid
    return o, w.latitude.values, w.longitude.values, valid, tnc


def write_step(out, fields, lat, lon, tnc, attrs=None, compress=False):
    hours = (tnc - EPOCH_1992) / np.timedelta64(1, "h")
    shape = (1, lat.size, lon.size)
    dims = ("record", "latitude", "longitude")
    data = {}
    for name, units in VARIABLES:
        v = np.asarray(fields[name], dtype=float).reshape(shape)
        data[name] = (dims, np.where(np.isfinite(v), v, FILL).astype("float32"), {"units": units})
    nc = xr.Dataset(data, coords={
        "record": ("record", [hours], {"units": "hours since 1992-1-1 00:00:00", "long_name": "record"}),
        "latitude": ("latitude", lat.astype(float), {"units": "degrees_north"}),
        "longitude": ("longitude", lon.astype(float), {"units": "degrees_east"}),
    }, attrs=attrs or {})
    enc = {k: {"_FillValue": np.float32(FILL), "dtype": "float32",
               **({"zlib": True, "complevel": 1} if compress else {})} for k, _ in VARIABLES}
    enc["record"] = {"dtype": "float64", "_FillValue": None}
    enc["latitude"] = enc["longitude"] = {"_FillValue": None}
    nc.to_netcdf(out, format="NETCDF4", encoding=enc, unlimited_dims=["record"])


def crea_netcdf(grib, outdir="esempio", first_step=11, n_files=72, time_mode="r", workers=None,
                compress=False, prefix="mol01gfs12_", fillrk=None, verbose=True) -> list[Path]:
    t_start = time.perf_counter()
    if workers:
        try:
            import numba

            numba.set_num_threads(min(workers, numba.config.NUMBA_NUM_THREADS))
        except ImportError:
            pass
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ds = open_raster(grib)
    run = run_of(grib, ds)
    run_day = run.astype("datetime64[D]").astype("datetime64[s]")
    fill = load_fillrk(fillrk)
    if verbose:
        print(f"GRIB aperto in {time.perf_counter() - t_start:.1f} s: run {run}, "
              f"{ds.sizes.get('valid_time', 1)} passi, griglia {ds.sizes['latitude']}x{ds.sizes['longitude']}",
              flush=True)
    if not n_files:
        n_files = ds.sizes.get("valid_time", 1) - (first_step - 1)
    files = []
    for i in range(1, n_files + 1):
        t0 = time.perf_counter()
        label = run_day + np.timedelta64(1, "D") + np.timedelta64(first_step - 11 + i - 1, "h")  # = valido + 2 h
        out = outdir / f"{prefix}{label.astype(object):%Y%m%d%H}.nc"
        fields, lat, lon, valid, tnc = compute_step(ds, run, i, time_mode, first_step, fill, workers)
        write_step(out, fields, lat, lon, tnc, compress=compress, attrs={
            "source": str(grib), "valid_time_utc": str(valid), "time_mode": time_mode,
            "producer": "salutepy.pipeline.meteosalute_nc"})
        files.append(out)
        if verbose:
            print(f"{i:2d}/{n_files} {out.name}  valido {valid} UTC  {time.perf_counter() - t0:5.1f} s", flush=True)
    if verbose:
        tot = time.perf_counter() - t_start
        print(f"Totale: {len(files)} file in {tot:.1f} s ({tot / max(len(files), 1):.1f} s/file) -> {outdir}")
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m salutepy.pipeline.meteosalute_nc")
    ap.add_argument("grib")
    ap.add_argument("-d", "--outdir", default="esempio")
    ap.add_argument("-w", "--workers", type=int)
    ap.add_argument("-n", "--n-files", type=int, default=72, help="numero di file (0 = fino all'ultimo passo)")
    ap.add_argument("--all", action="store_true", help="tutte le ore del GRIB (= --first-step 1 -n 0)")
    ap.add_argument("--first-step", type=int, default=11)
    ap.add_argument("--time-mode", default="r", choices=["r", "valid"])
    ap.add_argument("--fillrk", help="fillrkmoloch02.rds (default: copia inclusa nel pacchetto)")
    ap.add_argument("--compress", action="store_true", help="compressione zlib (file più piccoli)")
    a = ap.parse_args(argv)
    if a.all:
        a.first_step, a.n_files = 1, 0
    crea_netcdf(a.grib, a.outdir, a.first_step, a.n_files, a.time_mode, a.workers, a.compress, fillrk=a.fillrk)
    return 0


if __name__ == "__main__":
    sys.exit(main())
