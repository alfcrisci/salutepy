"""
Scrittura del NetCDF orario "meteosalute", equivalente alla sezione
"Nedtcdf creation" di meteosalute_op_dir_mol01.r (ncdim_def/ncvar_def/
nc_create/ncvar_put via libreria R ncdf4).

Stesse dimensioni (longitude, latitude, record[unlimited]), stessi nomi
di variabile, stesse unita', stesso fillvalue (-999.9) del file R
originale, per garantire compatibilita' con gli script a valle
(crea_matrici.r, i visualizzatori GrADS, ecc.) che si aspettano quello
schema esatto.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from netCDF4 import Dataset

FILLVALUE = -999.9

# (nome_variabile, unita') nello stesso ordine di dichiarazione del file R,
# per facilita' di confronto/manutenzione — l'ordine nel file NetCDF non è
# comunque significativo per la lettura via cdo/xarray a valle.
VARIABLES = [
    ("ta2m", "degC"),
    ("urel", "%"),
    ("v10m", "m/s"),
    ("gust", "m/s"),
    ("pmsl", "hPa"),
    ("pvap", "hPa"),
    ("defsat", "hPa"),
    ("csky", "dimensionless"),
    ("pmvsha", "dimensionless"),
    ("clomin", "dimensionless"),
    ("clomax", "dimensionless"),
    ("utci", "degC"),
    ("utcisha", "degC"),
    ("utciclasha", "dimensionless"),
    ("stosun", "degC"),
    ("stosha", "degC"),
    ("stoind", "degC"),
    ("temprad", "degC"),
    ("rcorte", "W m-2"),
    ("stosuncla", "dimensionless"),
    ("stoshacla", "dimensionless"),
    ("utcicla", "dimensionless"),
    ("wbgtsha", "degC"),
    ("wbgtsun", "degC"),
    ("tglob", "degC"),
    ("rkcla_int", "class"),
    ("rkcla_nint", "class"),
    ("rkshacla_int", "class"),
    ("rkshacla_nint", "class"),
    ("lprodsha", "perc"),
    ("lprod", "perc"),
    ("wbgttiwtlsha", "perc"),
    ("wbgttwtlsha", "hours"),
    ("wbgttiwtlsun", "perc"),
    ("wbgttwtlsun", "hours"),
    ("airtiwtl", "perc"),
    ("airtwtl", "hours"),
    ("windchill", "degC"),
    ("windcla", "dimensionless"),
    ("frostime", "minutes"),
    ("thi", "degC"),
    ("lapr", "dimensionless"),
]


def write_meteosalute_nc(
    path: str | Path,
    lon: np.ndarray,
    lat: np.ndarray,
    time_hours_since_1992: float,
    n_lon: int,
    n_lat: int,
    data: dict[str, np.ndarray],
) -> Path:
    """
    Scrive il file NetCDF orario, equivalente alla sezione "Nedtcdf
    creation" + "Put variables" del codice R.

    Parameters
    ----------
    path : percorso di output
    lon, lat : vettori di coordinate uniche (lunghezza n_lon, n_lat)
    time_hours_since_1992 : valore scalare del timestep, in ore da
        "1992-1-1 00:00:00" (stessa origine usata da ud.convert in R)
    n_lon, n_lat : dimensioni della griglia (per il reshape dei dati
        1D-per-pixel in 2D longitude x latitude)
    data : dict {nome_variabile: array 1D di lunghezza n_lon*n_lat},
        con le chiavi che devono coincidere con i nomi in VARIABLES
        (mancanti -> riempiti con FILLVALUE e un warning)

    Returns
    -------
    Path del file scritto
    """
    path = Path(path)

    with Dataset(path, "w", format="NETCDF4") as ncout:
        ncout.createDimension("longitude", n_lon)
        ncout.createDimension("latitude", n_lat)
        ncout.createDimension("record", None)  # unlimited, come in R

        time_var = ncout.createVariable("time", "f4", ("record",))
        time_var.units = "hours since 1992-01-01 00:00:00 UTC"
        time_var[:] = [time_hours_since_1992]

        lat_var = ncout.createVariable("lat", "f4", ("latitude",))
        lat_var.units = "degrees_north"
        lat_var.long_name = "latitude"
        lat_var[:] = lat

        lon_var = ncout.createVariable("lon", "f4", ("longitude",))
        lon_var.units = "degrees_east"
        lon_var.long_name = "longitude"
        lon_var[:] = lon

        for name, units in VARIABLES:
            var = ncout.createVariable(
                name, "f4", ("longitude", "latitude", "record"), fill_value=FILLVALUE
            )
            var.units = units

            if name in data:
                arr = np.asarray(data[name], dtype=float).reshape(n_lat, n_lon).T
                arr = np.where(np.isnan(arr), FILLVALUE, arr)
                var[:, :, 0] = arr
            else:
                import warnings

                warnings.warn(f"Variabile '{name}' mancante in data: scritta come fillvalue.")
                var[:, :, 0] = FILLVALUE

    return path
