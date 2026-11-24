"""
Porting di salute_aux.r::estrai_acclimdays / estrai_param (le versioni
consolidate/corrette, non le vecchie copie duplicate rimosse da
crea_matrici.r e crea_archive_moloch_grid.R durante il refactoring R).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from .shell import run_cmd

PIXELNUMBER_MOL01 = 358701


def estrai_acclimdays(nc_path: str | Path, pixelnumber: int = PIXELNUMBER_MOL01) -> np.ndarray:
    """
    Porting di salute_aux.r::estrai_acclimdays(x).
    Massimo giornaliero di wbgtsun estratto via cdo da un file NetCDF
    giornaliero d'archivio. Ritorna un vettore di missing (-999) se il
    file non esiste o l'estrazione cdo fallisce, invece di propagare
    l'errore (stessa semantica R).
    """
    nc_path = Path(nc_path)
    if not nc_path.exists():
        return np.full(pixelnumber, -999.0)

    out = nc_path.parent / "temp_acclimdays.nc"
    if out.exists():
        out.unlink()

    result = run_cmd(f"cdo -selvar,wbgtsun -daymax {nc_path} {out}", stop_on_error=False)
    if result.returncode != 0 or not out.exists():
        warnings.warn(f"estrazione acclimdays fallita su {nc_path} - restituisco vettore di missing")
        return np.full(pixelnumber, -999.0)

    with Dataset(out) as ds:
        varname = [v for v in ds.variables if v not in ("lon", "lat", "time", "longitude", "latitude", "record")][0]
        values = np.asarray(ds.variables[varname][:]).ravel()

    out.unlink()
    return np.round(values, 1)


def estrai_param(nc_path: str | Path, par: str = "wbgtsun", pixelnumber: int = PIXELNUMBER_MOL01) -> np.ndarray:
    """
    Porting di salute_aux.r::estrai_param(x, par).
    Estrae TUTTI i timestep della variabile richiesta via cdo da un file
    NetCDF (equivalente a `as.data.frame(brick(...))` in R: una colonna
    per ogni layer temporale presente nel file, non un singolo valore).
    Per un file giornaliero orario (24 record) ritorna una matrice
    (pixelnumber, 24).

    Ritorna una matrice/vettore di missing (-999) in caso di errore, con
    1 sola colonna (a differenza del caso di successo, che ne ha una per
    timestep) — stessa inconsistenza di tipo presente nella funzione R
    originale quando il file manca, riprodotta identicamente per fedeltà.
    """
    nc_path = Path(nc_path)
    if not nc_path.exists():
        return np.full((pixelnumber, 1), -999.0)

    out = nc_path.parent / "temp_par.nc"
    if out.exists():
        out.unlink()

    result = run_cmd(f"cdo -selvar,{par} {nc_path} {out}", stop_on_error=False)
    if result.returncode != 0 or not out.exists():
        warnings.warn(f"estrazione param {par} fallita su {nc_path} - restituisco vettore di missing")
        return np.full((pixelnumber, 1), -999.0)

    with Dataset(out) as ds:
        var = ds.variables[par]  # dims attese: (longitude, latitude, record)
        n_lon, n_lat, n_time = var.shape
        raw = np.asarray(var[:])  # (n_lon, n_lat, n_time)
        # stesso ordine "riga-maggiore" (lat varia lentamente, lon velocemente)
        # usato ovunque in questo package per il vettore-pixel 1D, coerente
        # con terra::as.data.frame() in R
        cols = [raw[:, :, t].T.ravel() for t in range(n_time)]
        values = np.column_stack(cols)

    out.unlink()
    return np.round(values, 1)
