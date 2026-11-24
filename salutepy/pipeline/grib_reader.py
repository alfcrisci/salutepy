"""
Lettura dei file GRIB2 splittati da cdo (un timestep, 9 variabili) usando
`rasterio`.

*** SCELTA TECNICA IMPORTANTE ***
Il codice R usa `terra::rast(fname)` per leggere il GRIB2, che sotto il
cofano si appoggia a GDAL. `datafile=as.data.frame(grib_r)` poi assegna i
nomi delle variabili PER POSIZIONE (`names(datafile)=names_meteos`, un
vettore fisso di 9 nomi), assumendo che l'ordine delle bande nel file
GRIB2 sia sempre lo stesso.

Per essere fedeli a questo comportamento (posizionale, non basato su
introspezione dei metadati GRIB), qui si usa `rasterio` — anch'esso un
binding diretto di GDAL — invece di `cfgrib`/`xarray`, che leggerebbe il
GRIB2 in un paradigma completamente diverso (un DataArray per ogni
combinazione di shortName/level/step, con auto-discovery delle variabili,
NON necessariamente nello stesso ordine posizionale del file). Usare
cfgrib qui rischierebbe di leggere le variabili in un ordine diverso da
quello assunto da names_meteos, con conseguente assegnazione errata dei
nomi.

*** NON TESTATO SU UN FILE GRIB2 REALE ***
Questa sessione non dispone di un file .grb2 di esempio prodotto da
`cdo -f grb2 -splitsel,1 -seltimestep,11/83`: il codice sotto e' scritto
per essere logicamente equivalente a `terra::rast()` + `as.data.frame()`,
ma va validato al primo utilizzo su un file reale (in particolare:
verificare che rasterio esponga le bande nello stesso ordine di terra,
e che `dataset.xy(row, col)` produca le stesse coordinate pixel-centro
di `terra::crds()`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

# Ordine fisso delle 9 variabili nel file GRIB2 splittato, identico a
# names_meteos in meteosalute_op_dir_mol01.r. Posizionale: dipende
# dall'ordine con cui MOLOCH scrive le variabili nel GRIB2 originale.
NAMES_METEOS = ["pmsl", "N_atm", "dswrf_sfc", "lapr", "T_fhg", "TD_fhg", "u_fhg", "v_fhg", "gust_sfc"]


@dataclass
class MolochGribData:
    """Equivalente del data.frame `datafile` + i vettori lon/lat/lonvect/latvect in R."""

    variables: dict  # nome variabile -> np.ndarray 1D (un valore per pixel, ordine riga-maggiore)
    lon: np.ndarray  # longitudini uniche delle colonne
    lat: np.ndarray  # latitudini uniche delle righe
    lonvect: np.ndarray  # longitudine di ogni pixel (stessa lunghezza delle variabili)
    latvect: np.ndarray  # latitudine di ogni pixel


def read_moloch_split(path: str | Path, names: list[str] | None = None) -> MolochGribData:
    """
    Legge un file GRIB2 a singolo timestep con 9 bande (output di
    `cdo -f grb2 -splitsel,1 ...`), equivalente a:

        grib_r = rast(fname)
        datafile = as.data.frame(grib_r)
        names(datafile) = names_meteos
        lon = unique(crds(grib_r)[,1]); lat = unique(crds(grib_r)[,2])
        lonvect = crds(grib_r)[,1]; latvect = crds(grib_r)[,2]

    Parameters
    ----------
    path : percorso del file GRIB2
    names : nomi delle variabili in ordine di banda (default NAMES_METEOS)

    Returns
    -------
    MolochGribData
    """
    names = names or NAMES_METEOS

    with rasterio.open(path) as ds:
        n_bands = ds.count
        if n_bands != len(names):
            raise ValueError(
                f"Il file {path} ha {n_bands} bande, attese {len(names)} ({names}). "
                "Verificare che l'ordine/numero di variabili nel GRIB2 sia quello atteso."
            )

        height, width = ds.height, ds.width

        # Coordinate pixel-centro, stesso concetto di terra::crds()
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        xs, ys = rasterio.transform.xy(ds.transform, rows.ravel(), cols.ravel())
        lonvect = np.asarray(xs, dtype=float)
        latvect = np.asarray(ys, dtype=float)

        # coordinate uniche di riga/colonna (assumendo griglia regolare, come in R)
        col_ids = np.arange(width)
        row_ids = np.arange(height)
        lon_unique = np.asarray(rasterio.transform.xy(ds.transform, [0] * width, col_ids)[0], dtype=float)
        lat_unique = np.asarray(rasterio.transform.xy(ds.transform, row_ids, [0] * height)[1], dtype=float)

        variables = {}
        for i, name in enumerate(names, start=1):
            band = ds.read(i).astype(float).ravel()
            variables[name] = band

    return MolochGribData(
        variables=variables,
        lon=lon_unique,
        lat=lat_unique,
        lonvect=lonvect,
        latvect=latvect,
    )
