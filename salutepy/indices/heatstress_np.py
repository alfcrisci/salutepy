"""
Vectorised NumPy ports of the indices of the R package HeatStress (A. Casanueva, MeteoSwiss,
version 1.0.8.1) that were not yet in salutepy:

swbgt, wbt.Stull, apparentTemp, discomInd, effectiveTemp, hi, humidex, tashurs2dewp,
tashurs2vap.pres, dewp2hurs. WBGT of Bernard and of Liljegren are in ``wbgt_fast.py``.

Each function reproduces the R code (same constants, same order of operations, same handling of
humidity above 100 percent) and is checked against the R package in tests/test_heatstress.py.
Inputs: temperatures in degC, relative humidity (hurs) in percent, wind in m/s.

References
----------
Casanueva A. et al. (2019). Climate projections of a multivariate heat stress index: the role of
downscaling and bias correction. Geosci. Model Dev. 12: 3419-3438. doi:10.5194/gmd-12-3419-2019
(R package HeatStress: https://github.com/anacv/HeatStress)
"""

from __future__ import annotations

import numpy as np


def _f(x):
    return np.asarray(x, dtype=float)


def tashurs2vap_pres(tas, hurs):
    """
    Water vapour pressure (hPa) from temperature (degC) and relative humidity (%), over water for
    tas >= 0 and over ice below (Dosseger et al., 1992); HeatStress::tashurs2vap.pres.
    """
    tas, hurs = np.broadcast_arrays(_f(tas), _f(hurs))
    t10 = 10 * tas
    h = np.where(hurs > 100, 100.0, hurs)
    water = h * 0.06107 * np.exp((17.368 * t10) / (2388.3 + t10))
    ice = h * 0.06108 * np.exp((17.856 * t10) / (2455.2 + t10))
    out = np.where(t10 < 0, ice, water)
    return np.where(np.isnan(tas), np.nan, out)


def tashurs2dewp(tas, hurs):
    """Dew-point temperature (degC) from temperature and relative humidity (Dosseger et al., 1992); HeatStress::tashurs2dewp."""
    tas, hurs = np.broadcast_arrays(_f(tas), _f(hurs))
    h = np.where(hurs > 100, 100.0, hurs) / 100
    with np.errstate(divide="ignore", invalid="ignore"):
        lw = np.log(h * 0.06107 * 100 * np.exp((17.368 * tas) / (238.83 + tas)) / (0.06107 * 100))
        li = np.log(h * 0.06108 * 100 * np.exp((17.856 * tas) / (245.52 + tas)) / (0.06108 * 100))
        water = 238.83 * lw / (17.368 - lw)
        ice = 245.52 * li / (17.856 - li)
    out = np.where(tas < 0, ice, water)
    return np.where(np.isnan(tas), np.nan, out)


def dewp2hurs(tas, dewp):
    """Relative humidity (%) from temperature and dew point (degC), limited to 0-100; HeatStress::dewp2hurs."""
    tas, dewp = np.broadcast_arrays(_f(tas), _f(dewp))
    water = 100 * np.exp(((17.368 * dewp) / (238.83 + dewp)) - ((17.368 * tas) / (238.83 + tas)))
    ice = 100 * np.exp(((17.856 * dewp) / (245.52 + dewp)) - ((17.856 * tas) / (245.52 + tas)))
    h = np.where(tas < 0, ice, water)
    h = np.where(h > 100, 100.0, h)
    h = np.where(h < 0, 0.0, h)
    return np.where(np.isnan(tas), np.nan, h)


def swbgt(tas, hurs):
    """
    Simplified WBGT (degC) of the Australian Bureau of Meteorology with the constants of Lemke &
    Kjellstrom (2012): 0.567 tas + 0.216 e + 3.38; HeatStress::swbgt.
    """
    return 0.567 * _f(tas) + 0.216 * tashurs2vap_pres(tas, hurs) + 3.38


def wbt_stull(tas, hurs):
    """Wet-bulb temperature (degC), Stull (2011); HeatStress::wbt.Stull."""
    tas, hurs = np.broadcast_arrays(_f(tas), _f(hurs))
    return (tas * np.arctan(0.151977 * np.sqrt(hurs + 8.313659)) + np.arctan(tas + hurs) - np.arctan(hurs - 1.676331)
            + 0.00391838 * (hurs ** (3 / 2)) * np.arctan(0.023101 * hurs) - 4.686035)


def apparent_temp(tas, hurs, wind):
    """Apparent temperature (degC), Steadman (1994) without radiation: tas + 0.33 e - 0.7 wind - 4; HeatStress::apparentTemp."""
    return _f(tas) + 0.33 * tashurs2vap_pres(tas, hurs) - 0.7 * _f(wind) - 4


def discomfort_index(tas, hurs):
    """Thom discomfort index (degC), Giles et al. (1990) form; HeatStress::discomInd (same as THI of biometeo.js)."""
    tas, hurs = np.broadcast_arrays(_f(tas), _f(hurs))
    return tas - 0.55 * (1 - 0.01 * hurs) * (tas - 14.5)


def effective_temp(tas, hurs, wind):
    """Normal effective temperature (degC), Missenard / Gregorczuk & Cena; HeatStress::effectiveTemp (= net_index of biometeo.js, unrounded)."""
    tas, hurs, wind = np.broadcast_arrays(_f(tas), _f(hurs), _f(wind))
    return 37 - (37 - tas) / (0.68 - 0.0014 * hurs + 1 / (1.76 + 1.4 * (wind ** 0.75))) - 0.29 * tas * (1 - 0.01 * hurs)


def hi(tas, hurs):
    """
    Heat index (degF!) of the US NWS, as in HeatStress::hi: Rothfusz regression with the two
    adjustments and the simple formula when T or the simple HI are below 80 F. The result is in
    degrees Fahrenheit, as in the R package.

    Note: HeatStress computes the low-humidity adjustment as ((13 - RH) / 4) * sqrt(17 - |T - 95| / 17),
    while the NWS formula is ((13 - RH) / 4) * sqrt((17 - |T - 95|) / 17). The R behaviour is reproduced
    here; heatindex() of biometeo.js follows the NWS formula.
    """
    tas, hurs = np.broadcast_arrays(_f(tas), _f(hurs))
    tasf = tas * 1.8 + 32
    simple = 0.5 * (tasf + 61.0 + ((tasf - 68.0) * 1.2) + (hurs * 0.094))
    res = (-42.379 + 2.04901523 * tasf + 10.14333127 * hurs - 0.22475541 * tasf * hurs - 6.83783e-3 * tasf ** 2
           - 5.481717e-2 * hurs ** 2 + 1.22874e-3 * tasf ** 2 * hurs + 8.5282e-4 * tasf * hurs ** 2
           - 1.99e-6 * tasf ** 2 * hurs ** 2)
    f1 = (tasf >= 80) & (tasf <= 112) & (hurs <= 13)
    f2 = (tasf >= 80) & (tasf <= 87) & (hurs > 85)
    with np.errstate(invalid="ignore"):
        adj1 = (13 - hurs) / 4 * np.sqrt(17 - np.abs(tasf - 95) / 17)
    adj2 = (hurs - 85) / 10 * ((87 - tasf) / 5)
    res = np.where(f1, res - adj1, res)
    res = np.where(f2, res + adj2, res)
    res = np.where(tasf < 80, simple, res)
    res = np.where(((simple + tasf) / 2) < 80, simple, res)
    return res


def humidex(tas, hurs):
    """Humidex (degC): tas + 5/9 (e - 10) with e from tashurs2vap.pres; HeatStress::humidex."""
    return _f(tas) + 5 / 9 * (tashurs2vap_pres(tas, hurs) - 10)
