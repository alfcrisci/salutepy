"""
Porting di salute_aux.r::sunposition_norefrac.

*** SOSTITUZIONE DOCUMENTATA ***
Il codice R chiama `sunposition(posixtimefile, latvect[x], lonvect[x])`,
che risolve a `suntools::sunposition` (il pacchetto e' caricato con
`library(suntools)` in testa a meteosalute_op_dir_mol01.r). Non ho il
sorgente di `suntools` in questa sessione. Tuttavia `salute_aux.r` — che
fa parte del codice originale della pipeline, non un pacchetto esterno —
contiene gia' una funzione locale `sunposition_norefrac(timenow, lat, long)`
con la STESSA firma (tempo, lat, lon) e la STESSA struttura di output
(`list(elevation=.., azimuth=.., solarzen=..)`) richiesta dal chiamante:

    sunpos$elevation, sunpos$azimuth, sunpos$solarzen

Uso quindi `sunposition_norefrac` (algoritmo NOAA-style, senza
correzione di rifrazione atmosferica) come sostituto trasparente di
`suntools::sunposition`, dato che e' codice della stessa codebase con
identica interfaccia. Se `suntools::sunposition` applica una formula
diversa o una correzione di rifrazione, l'elevazione solare calcolata
qui potrebbe differire leggermente (tipicamente <1° vicino all'orizzonte,
dove la rifrazione atmosferica ha l'effetto maggiore) — da verificare
con un confronto numerico diretto se necessario.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np


def _is_leapyear(year):
    return ((year % 4 == 0) & (year % 400 == 0)) | ((year % 4 == 0) & (year % 100 != 0))


def sunposition_norefrac(timenow: datetime, lat: float, lon: float) -> dict:
    """
    Porting di salute_aux.r::sunposition_norefrac(timenow, lat, long).
    Posizione solare (senza correzione di rifrazione atmosferica).

    Parameters
    ----------
    timenow : datetime.datetime (assunto in UTC/GMT, come as.POSIXlt(...,"GMT") in R)
    lat, lon : float o array-like, gradi

    Returns
    -------
    dict con 'elevation', 'azimuth', 'solarzen' (gradi)
    """
    if timenow.tzinfo is None:
        timenow = timenow.replace(tzinfo=timezone.utc)
    else:
        timenow = timenow.astimezone(timezone.utc)

    year = timenow.year
    hour = timenow.hour
    day = timenow.day
    month = timenow.month
    minute = timenow.minute
    sec = timenow.second

    twopi = 2 * np.pi
    deg2rad = np.pi / 180

    month_days = np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])
    day = day + np.cumsum(month_days)[month - 1]

    leapday = (
        (year % 4 == 0)
        and (year % 400 == 0 or year % 100 != 0)
        and day >= 60
        and not (month == 2 and day == 60)
    )
    if leapday:
        day = day + 1

    hour = hour + minute / 60 + sec / 3600
    delta = year - 1949
    leap = int(delta / 4)
    jd = 32916.5 + delta * 365 + leap + day + hour / 24
    time = jd - 51545.0

    mnlong = (280.46 + 0.9856474 * time) % 360
    mnlong = mnlong + 360 if mnlong < 0 else mnlong

    mnanom = (357.528 + 0.9856003 * time) % 360
    mnanom = mnanom + 360 if mnanom < 0 else mnanom
    mnanom = mnanom * deg2rad

    eclong = mnlong + 1.915 * np.sin(mnanom) + 0.02 * np.sin(2 * mnanom)
    eclong = eclong % 360
    eclong = eclong + 360 if eclong < 0 else eclong
    oblqec = 23.439 - 4e-07 * time
    eclong = eclong * deg2rad
    oblqec = oblqec * deg2rad

    num = np.cos(oblqec) * np.sin(eclong)
    den = np.cos(eclong)
    ra = np.arctan(num / den)
    if den < 0:
        ra = ra + np.pi
    elif num < 0:
        ra = ra + twopi
    dec = np.arcsin(np.sin(oblqec) * np.sin(eclong))

    gmst = (6.697375 + 0.0657098242 * time + hour) % 24
    gmst = gmst + 24 if gmst < 0 else gmst

    lat_arr = np.atleast_1d(np.asarray(lat, dtype=float))
    lon_arr = np.atleast_1d(np.asarray(lon, dtype=float))

    lmst = (gmst + lon_arr / 15.0) % 24.0
    lmst = np.where(lmst < 0, lmst + 24.0, lmst)
    lmst = lmst * 15.0 * deg2rad

    ha = lmst - ra
    ha = np.where(ha < -np.pi, ha + twopi, ha)
    ha = np.where(ha > np.pi, ha - twopi, ha)

    lat_rad = lat_arr * deg2rad

    el = np.arcsin(np.sin(dec) * np.sin(lat_rad) + np.cos(dec) * np.cos(lat_rad) * np.cos(ha))
    az = np.arcsin(-np.cos(dec) * np.sin(ha) / np.cos(el))

    cos_az_pos = 0 <= (np.sin(dec) - np.sin(el) * np.sin(lat_rad))
    sin_az_neg = np.sin(az) < 0
    az = np.where(cos_az_pos & sin_az_neg, az + twopi, az)
    az = np.where(~cos_az_pos, np.pi - az, az)

    el_deg = el / deg2rad
    az_deg = az / deg2rad

    solar_zen = 90 - el_deg

    return {"elevation": el_deg, "azimuth": az_deg, "solarzen": solar_zen}
