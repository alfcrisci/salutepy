"""
Porting di rBiometeo (biometeo.js): Steadman apparent temperature, THI,
heat_risk_level, mrt_globe, windchill, frostime.

Ogni funzione R in rBiometeo e' un thin wrapper verso una funzione
JavaScript omonima eseguita via V8 (oggetto `ct` nel sorgente R). Le
formule qui sotto sono portate dalle funzioni JS in
rBiometeo-master/inst/js/biometeo.js (non dai wrapper R, che non
contengono la formula).
"""

from __future__ import annotations

import numpy as np


def two_dec(x):
    """Porting di TwoDec() in biometeo.js: arrotondamento a 2 decimali."""
    return np.round(x, 2)


def one_dec(x):
    """Porting di OneDec() in biometeo.js: arrotondamento a 1 decimale."""
    return np.round(x, 1)


# ---------------------------------------------------------------------------
# Steadman apparent temperature
# ---------------------------------------------------------------------------

def steadman_outdoor_shade(t, rh, wind):
    """
    Porting di biometeo.js: steadman_outdoor_shade(t,rh,wind).
    Fuori dai range di validita' (rh 0-100.1%, wind 0-130 m/s, t -100/100 degC)
    ritorna NaN (era il sentinella 9999 in R/JS).
    """
    t = np.asarray(t, dtype=float)
    rh = np.asarray(rh, dtype=float)
    wind = np.asarray(wind, dtype=float)

    e = (rh / 100.0) * (6.105 * np.exp((t * 17.27) / (237.7 + t)))
    result = t + (0.33 * e) - (0.7 * wind) - 4.0

    invalid = (rh > 100.1) | (rh < 0.0) | (wind > 130.0) | (wind < 0.0) | (t > 100.0) | (t < -100.0)
    result = np.where(invalid, np.nan, result)

    return two_dec(result)


def steadman_outdoor_sun(t, rh, wind, solar, sunelev):
    """
    Porting di biometeo.js: steadman_outdoor(t,rh,wind,solar,sunelev).
    Se solar < 50 W/m2 ricade su steadman_outdoor_shade (come nel JS
    originale: "if (solar < 50) return steadman_outdoor_shade(...)").
    """
    t = np.asarray(t, dtype=float)
    rh = np.asarray(rh, dtype=float)
    wind = np.asarray(wind, dtype=float)
    solar = np.asarray(solar, dtype=float)
    sunelev = np.asarray(sunelev, dtype=float)

    invalid = (rh > 100.1) | (rh < 0.0) | (t > 100.0) | (t < -100.0)

    shade_fallback = solar < 50

    ee = (rh / 1000.0) * (6.105 * np.exp((t * 17.27) / (237.7 + t)))
    q_glob = (
        0.56 * (0.386 - (0.0032 * sunelev)) * solar
        + 0.224 * (0.1 * solar)
        + 0.028 * solar
        - 150.0 * (0.38 - 0.16 * np.sqrt(ee))
    )

    with np.errstate(invalid="ignore"):
        sun_result = np.where(q_glob > 0.0, t + 3.48 * ee - 0.7 * wind + 0.7 * q_glob / (wind + 10.0) - 4.25, np.nan)

    shade_result = steadman_outdoor_shade(t, rh, wind)

    result = np.where(shade_fallback, shade_result, sun_result)
    result = np.where(invalid, np.nan, result)

    return two_dec(result)


def steadman_indoor(t, rh):
    """Porting di biometeo.js: steadman_indoor(t,rh)."""
    t = np.asarray(t, dtype=float)
    rh = np.asarray(rh, dtype=float)

    e = (rh / 100.0) * (6.105 * np.exp((t * 17.27) / (237.7 + t)))
    result = -2.56 + (0.89 * t) + (0.382 * e)

    invalid = (rh > 100.1) | (rh < 0.0) | (t > 100.0) | (t < -100.0)
    result = np.where(invalid, np.nan, result)

    return two_dec(result)


def steadman_class(steadman_index):
    """
    Porting di rBiometeo::steadman_class.R.
    Classi 1-9 secondo i breakpoint:
    (-100,-4.99],(-4.99,0.01],(0.01,5.01],(5.01,10.01],(10.01,27.01],
    (27.01,30.01],(30.01,35.01],(35.01,40.01],(40.01,100]
    """
    x = np.asarray(steadman_index, dtype=float)
    bins = [-100, -4.99, 0.01, 5.01, 10.01, 27.01, 30.01, 35.01, 40.01, 100]
    # np.digitize con bins crescenti, right=True corrisponde a cut(...,right=TRUE) di R
    classes = np.digitize(x, bins[1:], right=True) + 1
    classes = classes.astype(float)
    classes[np.isnan(x)] = np.nan
    classes[(x < bins[0]) | (x > bins[-1])] = np.nan
    return classes


# ---------------------------------------------------------------------------
# THI, heat risk, mrt, windchill, frostime
# ---------------------------------------------------------------------------

def thi(t, rh):
    """
    Porting di rBiometeo::THI.R (biometeo.js THI, Giles 1990).
    Temperature Humidity Index discomfort.
    """
    t = np.asarray(t, dtype=float)
    rh = np.asarray(rh, dtype=float)
    return 0.8 * t - 0.55 * (0.01 * rh) * (t - 14.4) + 46.4


def heat_risk_level(wbgt, cav, tresh=28.5):
    """
    Porting di rBiometeo::heat_risk_level.R (biometeo.js heat_risk_level).
    Classe di rischio da caldo per il lavoratore: 1 (<=0.8), 2 (<1),
    3 (<1.2), 4 (>=1.2), sul rapporto (wbgt+cav)/tresh.
    """
    wbgt = np.asarray(wbgt, dtype=float)
    cav = np.asarray(cav, dtype=float)
    risk = (wbgt + cav) / tresh

    classes = np.where(risk <= 0.8, 1, np.where(risk < 1, 2, np.where(risk < 1.2, 3, 4)))
    return classes.astype(float)


def mrt_globe(t, tg, wind, diam_globe=0.05, emis_globe=0.97):
    """
    Porting di rBiometeo (biometeo.js mrt_globe(t,tg,wind,diam_globe,emis_globe)).
    Temperatura media radiante (degC) secondo ISO 7726:1998.

    diam_globe e' in METRI (default 0.05 = 50mm), coerente con la formula
    JS originale che fa Math.pow(diam_globe, 0.4) direttamente.

    RISOLTO: nel codice R della pipeline (meteosalute_op_dir_mol01.r) la
    chiamata passava in precedenza 150 invece di 0.15 come diametro
    (commento "150 mm" ma valore in millimetri passato come numero puro,
    anziche' convertito in metri) — un fattore ~27x sul termine di scambio
    convettivo. Corretto nel sorgente R a
    `mrt_globe(datafile$T_fhg, tglob, wind_speed2m, 0.15)`. Per riprodurre
    il vecchio output bug-compatibile (solo per confronto/debug), chiamare
    esplicitamente mrt_globe(t, tg, wind, diam_globe=150).
    """
    t = np.asarray(t, dtype=float)
    tg = np.asarray(tg, dtype=float)
    wind = np.asarray(wind, dtype=float)
    stefanb = 0.0000000567

    return (
        (tg + 273.15) ** 4
        + ((1.1e8 * wind ** 0.6) / (emis_globe * diam_globe ** 0.4)) * (tg - t)
    ) ** 0.25 - 273.15


def windchill(t, wind):
    """
    Porting di rBiometeo::windchill.R (biometeo.js windchill, formula NWS
    metrica con vento in km/h). Il wrapper R applica inoltre
    `res[which(t>10)]=t[which(t>10)]`: per t>10 degC il windchill non ha
    senso fisico e viene sostituito dalla temperatura stessa — riprodotto
    qui identicamente.
    """
    t = np.asarray(t, dtype=float)
    wind = np.asarray(wind, dtype=float)

    wind_kmh = np.maximum(wind, 1.3) * 3.6
    twc = 13.12 + 0.6215 * t - 11.37 * wind_kmh ** 0.16 + 0.3965 * t * wind_kmh ** 0.16

    return np.where(t > 10, t, twc)


def frostime(t, wind):
    """
    Porting di rBiometeo::frostime.R (biometeo.js frostime).
    Tempo di congelamento (minuti) della pelle esposta.

    Valido solo per t in [-60,-10] degC e wind in [0,100.1] m/s (la
    condizione OR nel JS originale, `t > -10 || t < -60`, e' la negazione
    corretta dell'intervallo valido [-60,-10]: fuori da quel range di
    temperature "fredde" la formula non e' definita e ritorna NaN, come nel
    sentinella 9999 originale).
    """
    t = np.asarray(t, dtype=float)
    wind = np.asarray(wind, dtype=float)

    invalid = (wind > 100.1) | (wind < 0.0) | (t > -10.0) | (t < -60.0)

    # nel JS originale: ft=(formula)/60; return OneDec(ft*60) -- la divisione
    # e la moltiplicazione per 60 si annullano matematicamente, quindi il
    # risultato e' semplicemente OneDec(formula), in MINUTI (convenzione
    # della wind chill frostbite chart, non secondi)
    with np.errstate(invalid="ignore"):
        ft = (-24.5 * ((0.667 * wind) + 4.8) + 2111) * ((-t - 4.8) ** -1.668)
    ft = one_dec(ft)

    return np.where(invalid, np.nan, ft)
