"""
Vectorised NumPy ports of the main indices of biometeo.js (rBiometeo 0.4).

Every function accepts scalars or NumPy arrays (broadcast together) and returns float arrays.
The "not computable" codes of biometeo.js (9999 / -9999) become NaN. Results are checked against
biometeo.js itself in tests/test_biometeo_np.py (same values within 1e-9, or the same rounding).

Indices: UTCI, heat index, humidex, NET, THI (Thom/Giles), wind chill, MRT from globe temperature
(ISO 7726) and from solar radiation, solar elevation, PET (Hoppe / VDI 3787, see pet.py),
WBGT (Liljegren, salutepy.indices.wbgt). Classes with Italian/English labels: see CLASSES.

References are given in each function and in biometeo.js.
"""

from __future__ import annotations

import numpy as np

RAD = np.pi / 180.0


def _f(x):
    return np.asarray(x, dtype=float)


def js_round(x, decimals=0):
    """Math.round of JavaScript (halves rounded towards +inf), applied with 10^decimals scaling."""
    m = 10.0 ** decimals
    return np.floor(_f(x) * m + 0.5) / m


def two_dec(x):
    """TwoDec() of biometeo.js."""
    return js_round(x, 2)


def one_dec(x):
    """OneDec() of biometeo.js."""
    return js_round(x, 1)


# ---------------------------------------------------------------------------
# Psychrometrics
# ---------------------------------------------------------------------------

_G = (-2.8365744e3, -6.028076559e3, 1.954263612e1, -2.737830188e-2, 1.6261698e-5, 7.0229056e-10, -1.8680009e-13)


def es(t):
    """Saturation vapour pressure over water (hPa), Hardy (1998) ITS-90; es() of biometeo.js."""
    tk = _f(t) + 273.15
    e = 2.7150305 * np.log(tk)
    for i, g in enumerate(_G):
        e = e + g * tk ** (i - 2)
    return np.exp(e) * 0.01


def esat(tk):
    """Saturation vapour pressure (hPa) for tk in K, Buck (1981); esat() of biometeo.js."""
    tk = _f(tk)
    return 1.004 * (6.1121 * np.exp((17.502 * (tk - 273.15)) / (tk - 32.18)))


def emis_atm(tk, rh_fraction):
    """Atmospheric emissivity (Oke 1987), tk in K and relative humidity as a fraction."""
    return 0.575 * (_f(rh_fraction) * esat(tk)) ** 0.143


def rh_from_dewpoint(t, td):
    """Relative humidity (%) from air and dew-point temperature (degC), with es() (Hardy 1998)."""
    return 100.0 * es(td) / es(t)


def wind_speed(u, v):
    """Wind speed from its u and v components."""
    return np.hypot(_f(u), _f(v))


def _utci_polynomial(ta, va, dtm, pa):
    """Sum of the 211 terms of the UTCI 6th-order polynomial, generated from UTCI() in biometeo.js."""
    total = ta + 0.0
    total = total + (6.07562052e-1)
    total = total + (-2.27712343e-2 * ta)
    total = total + (8.06470249e-4 * ta * ta)
    total = total + (-1.54271372e-4 * ta * ta * ta)
    total = total + (-3.24651735e-6 * ta * ta * ta * ta)
    total = total + (7.32602852e-8 * ta * ta * ta * ta * ta)
    total = total + (1.35959073e-9 * ta * ta * ta * ta * ta * ta)
    total = total + (-2.2583652 * va)
    total = total + (8.80326035e-2 * ta * va)
    total = total + (2.16844454e-3 * ta * ta * va)
    total = total + (-1.53347087e-5 * ta * ta * ta * va)
    total = total + (-5.72983704e-7 * ta * ta * ta * ta * va)
    total = total + (-2.55090145e-9 * ta * ta * ta * ta * ta * va)
    total = total + (-7.51269505e-1 * va * va)
    total = total + (-4.08350271e-3 * ta * va * va)
    total = total + (-5.21670675e-5 * ta * ta * va * va)
    total = total + (1.94544667e-6 * ta * ta * ta * va * va)
    total = total + (1.14099531e-8 * ta * ta * ta * ta * va * va)
    total = total + (1.58137256e-1 * va * va * va)
    total = total + (-6.57263143e-5 * ta * va * va * va)
    total = total + (2.22697524e-7 * ta * ta * va * va * va)
    total = total + (-4.16117031e-8 * ta * ta * ta * va * va * va)
    total = total + (-1.27762753e-2 * va * va * va * va)
    total = total + (9.66891875e-6 * ta * va * va * va * va)
    total = total + (2.52785852e-9 * ta * ta * va * va * va * va)
    total = total + (4.56306672e-4 * va * va * va * va * va)
    total = total + (-1.74202546e-7 * ta * va * va * va * va * va)
    total = total + (-5.91491269e-6 * va * va * va * va * va * va)
    total = total + (3.98374029e-1 * dtm)
    total = total + (1.83945314e-4 * ta * dtm)
    total = total + (-1.7375451e-4 * ta * ta * dtm)
    total = total + (-7.60781159e-7 * ta * ta * ta * dtm)
    total = total + (3.77830287e-8 * ta * ta * ta * ta * dtm)
    total = total + (5.43079673e-10 * ta * ta * ta * ta * ta * dtm)
    total = total + (-2.00518269e-2 * va * dtm)
    total = total + (8.92859837e-4 * ta * va * dtm)
    total = total + (3.45433048e-6 * ta * ta * va * dtm)
    total = total + (-3.77925774e-7 * ta * ta * ta * va * dtm)
    total = total + (-1.69699377e-9 * ta * ta * ta * ta * va * dtm)
    total = total + (1.69992415e-4 * va * va * dtm)
    total = total + (-4.99204314e-5 * ta * va * va * dtm)
    total = total + (2.47417178e-7 * ta * ta * va * va * dtm)
    total = total + (1.07596466e-8 * ta * ta * ta * va * va * dtm)
    total = total + (8.49242932e-5 * va * va * va * dtm)
    total = total + (1.35191328e-6 * ta * va * va * va * dtm)
    total = total + (-6.21531254e-9 * ta * ta * va * va * va * dtm)
    total = total + (-4.99410301e-6 * va * va * va * va * dtm)
    total = total + (-1.89489258e-8 * ta * va * va * va * va * dtm)
    total = total + (8.15300114e-8 * va * va * va * va * va * dtm)
    total = total + (7.5504309e-4 * dtm * dtm)
    total = total + (-5.65095215e-5 * ta * dtm * dtm)
    total = total + (-4.52166564e-7 * ta * ta * dtm * dtm)
    total = total + (2.46688878e-8 * ta * ta * ta * dtm * dtm)
    total = total + (2.42674348e-10 * ta * ta * ta * ta * dtm * dtm)
    total = total + (1.5454725e-4 * va * dtm * dtm)
    total = total + (5.2411097e-6 * ta * va * dtm * dtm)
    total = total + (-8.75874982e-8 * ta * ta * va * dtm * dtm)
    total = total + (-1.50743064e-9 * ta * ta * ta * va * dtm * dtm)
    total = total + (-1.56236307e-5 * va * va * dtm * dtm)
    total = total + (-1.33895614e-7 * ta * va * va * dtm * dtm)
    total = total + (2.49709824e-9 * ta * ta * va * va * dtm * dtm)
    total = total + (6.51711721e-7 * va * va * va * dtm * dtm)
    total = total + (1.94960053e-9 * ta * va * va * va * dtm * dtm)
    total = total + (-1.00361113e-8 * va * va * va * va * dtm * dtm)
    total = total + (-1.21206673e-5 * dtm * dtm * dtm)
    total = total + (-2.1820366e-7 * ta * dtm * dtm * dtm)
    total = total + (7.51269482e-9 * ta * ta * dtm * dtm * dtm)
    total = total + (9.79063848e-11 * ta * ta * ta * dtm * dtm * dtm)
    total = total + (1.25006734e-6 * va * dtm * dtm * dtm)
    total = total + (-1.81584736e-9 * ta * va * dtm * dtm * dtm)
    total = total + (-3.52197671e-10 * ta * ta * va * dtm * dtm * dtm)
    total = total + (-3.3651463e-8 * va * va * dtm * dtm * dtm)
    total = total + (1.35908359e-10 * ta * va * va * dtm * dtm * dtm)
    total = total + (4.1703262e-10 * va * va * va * dtm * dtm * dtm)
    total = total + (-1.30369025e-9 * dtm * dtm * dtm * dtm)
    total = total + (4.13908461e-10 * ta * dtm * dtm * dtm * dtm)
    total = total + (9.22652254e-12 * ta * ta * dtm * dtm * dtm * dtm)
    total = total + (-5.08220384e-9 * va * dtm * dtm * dtm * dtm)
    total = total + (-2.24730961e-11 * ta * va * dtm * dtm * dtm * dtm)
    total = total + (1.17139133e-10 * va * va * dtm * dtm * dtm * dtm)
    total = total + (6.62154879e-10 * dtm * dtm * dtm * dtm * dtm)
    total = total + (4.0386326e-13 * ta * dtm * dtm * dtm * dtm * dtm)
    total = total + (1.95087203e-12 * va * dtm * dtm * dtm * dtm * dtm)
    total = total + (-4.73602469e-12 * dtm * dtm * dtm * dtm * dtm * dtm)
    total = total + (5.12733497 * pa)
    total = total + (-3.12788561e-1 * ta * pa)
    total = total + (-1.96701861e-2 * ta * ta * pa)
    total = total + (9.9969087e-4 * ta * ta * ta * pa)
    total = total + (9.51738512e-6 * ta * ta * ta * ta * pa)
    total = total + (-4.66426341e-7 * ta * ta * ta * ta * ta * pa)
    total = total + (5.48050612e-1 * va * pa)
    total = total + (-3.30552823e-3 * ta * va * pa)
    total = total + (-1.6411944e-3 * ta * ta * va * pa)
    total = total + (-5.16670694e-6 * ta * ta * ta * va * pa)
    total = total + (9.52692432e-7 * ta * ta * ta * ta * va * pa)
    total = total + (-4.29223622e-2 * va * va * pa)
    total = total + (5.00845667e-3 * ta * va * va * pa)
    total = total + (1.00601257e-6 * ta * ta * va * va * pa)
    total = total + (-1.81748644e-6 * ta * ta * ta * va * va * pa)
    total = total + (-1.25813502e-3 * va * va * va * pa)
    total = total + (-1.79330391e-4 * ta * va * va * va * pa)
    total = total + (2.34994441e-6 * ta * ta * va * va * va * pa)
    total = total + (1.29735808e-4 * va * va * va * va * pa)
    total = total + (1.2906487e-6 * ta * va * va * va * va * pa)
    total = total + (-2.28558686e-6 * va * va * va * va * va * pa)
    total = total + (-3.69476348e-2 * dtm * pa)
    total = total + (1.62325322e-3 * ta * dtm * pa)
    total = total + (-3.1427968e-5 * ta * ta * dtm * pa)
    total = total + (2.59835559e-6 * ta * ta * ta * dtm * pa)
    total = total + (-4.77136523e-8 * ta * ta * ta * ta * dtm * pa)
    total = total + (8.6420339e-3 * va * dtm * pa)
    total = total + (-6.87405181e-4 * ta * va * dtm * pa)
    total = total + (-9.13863872e-6 * ta * ta * va * dtm * pa)
    total = total + (5.15916806e-7 * ta * ta * ta * va * dtm * pa)
    total = total + (-3.59217476e-5 * va * va * dtm * pa)
    total = total + (3.28696511e-5 * ta * va * va * dtm * pa)
    total = total + (-7.10542454e-7 * ta * ta * va * va * dtm * pa)
    total = total + (-1.243823e-5 * va * va * va * dtm * pa)
    total = total + (-7.385844e-9 * ta * va * va * va * dtm * pa)
    total = total + (2.20609296e-7 * va * va * va * va * dtm * pa)
    total = total + (-7.3246918e-4 * dtm * dtm * pa)
    total = total + (-1.87381964e-5 * ta * dtm * dtm * pa)
    total = total + (4.80925239e-6 * ta * ta * dtm * dtm * pa)
    total = total + (-8.7549204e-8 * ta * ta * ta * dtm * dtm * pa)
    total = total + (2.7786293e-5 * va * dtm * dtm * pa)
    total = total + (-5.06004592e-6 * ta * va * dtm * dtm * pa)
    total = total + (1.14325367e-7 * ta * ta * va * dtm * dtm * pa)
    total = total + (2.53016723e-6 * va * va * dtm * dtm * pa)
    total = total + (-1.72857035e-8 * ta * va * va * dtm * dtm * pa)
    total = total + (-3.95079398e-8 * va * va * va * dtm * dtm * pa)
    total = total + (-3.59413173e-7 * dtm * dtm * dtm * pa)
    total = total + (7.04388046e-7 * ta * dtm * dtm * dtm * pa)
    total = total + (-1.89309167e-8 * ta * ta * dtm * dtm * dtm * pa)
    total = total + (-4.79768731e-7 * va * dtm * dtm * dtm * pa)
    total = total + (7.96079978e-9 * ta * va * dtm * dtm * dtm * pa)
    total = total + (1.62897058e-9 * va * va * dtm * dtm * dtm * pa)
    total = total + (3.94367674e-8 * dtm * dtm * dtm * dtm * pa)
    total = total + (-1.18566247e-9 * ta * dtm * dtm * dtm * dtm * pa)
    total = total + (3.34678041e-10 * va * dtm * dtm * dtm * dtm * pa)
    total = total + (-1.15606447e-10 * dtm * dtm * dtm * dtm * dtm * pa)
    total = total + (-2.80626406 * pa * pa)
    total = total + (5.48712484e-1 * ta * pa * pa)
    total = total + (-3.9942841e-3 * ta * ta * pa * pa)
    total = total + (-9.54009191e-4 * ta * ta * ta * pa * pa)
    total = total + (1.93090978e-5 * ta * ta * ta * ta * pa * pa)
    total = total + (-3.08806365e-1 * va * pa * pa)
    total = total + (1.16952364e-2 * ta * va * pa * pa)
    total = total + (4.95271903e-4 * ta * ta * va * pa * pa)
    total = total + (-1.90710882e-5 * ta * ta * ta * va * pa * pa)
    total = total + (2.10787756e-3 * va * va * pa * pa)
    total = total + (-6.98445738e-4 * ta * va * va * pa * pa)
    total = total + (2.30109073e-5 * ta * ta * va * va * pa * pa)
    total = total + (4.1785659e-4 * va * va * va * pa * pa)
    total = total + (-1.27043871e-5 * ta * va * va * va * pa * pa)
    total = total + (-3.04620472e-6 * va * va * va * va * pa * pa)
    total = total + (5.14507424e-2 * dtm * pa * pa)
    total = total + (-4.32510997e-3 * ta * dtm * pa * pa)
    total = total + (8.99281156e-5 * ta * ta * dtm * pa * pa)
    total = total + (-7.14663943e-7 * ta * ta * ta * dtm * pa * pa)
    total = total + (-2.66016305e-4 * va * dtm * pa * pa)
    total = total + (2.63789586e-4 * ta * va * dtm * pa * pa)
    total = total + (-7.01199003e-6 * ta * ta * va * dtm * pa * pa)
    total = total + (-1.06823306e-4 * va * va * dtm * pa * pa)
    total = total + (3.61341136e-6 * ta * va * va * dtm * pa * pa)
    total = total + (2.29748967e-7 * va * va * va * dtm * pa * pa)
    total = total + (3.04788893e-4 * dtm * dtm * pa * pa)
    total = total + (-6.42070836e-5 * ta * dtm * dtm * pa * pa)
    total = total + (1.16257971e-6 * ta * ta * dtm * dtm * pa * pa)
    total = total + (7.68023384e-6 * va * dtm * dtm * pa * pa)
    total = total + (-5.47446896e-7 * ta * va * dtm * dtm * pa * pa)
    total = total + (-3.5993791e-8 * va * va * dtm * dtm * pa * pa)
    total = total + (-4.36497725e-6 * dtm * dtm * dtm * pa * pa)
    total = total + (1.68737969e-7 * ta * dtm * dtm * dtm * pa * pa)
    total = total + (2.67489271e-8 * va * dtm * dtm * dtm * pa * pa)
    total = total + (3.23926897e-9 * dtm * dtm * dtm * dtm * pa * pa)
    total = total + (-3.53874123e-2 * pa * pa * pa)
    total = total + (-2.2120119e-1 * ta * pa * pa * pa)
    total = total + (1.55126038e-2 * ta * ta * pa * pa * pa)
    total = total + (-2.63917279e-4 * ta * ta * ta * pa * pa * pa)
    total = total + (4.53433455e-2 * va * pa * pa * pa)
    total = total + (-4.32943862e-3 * ta * va * pa * pa * pa)
    total = total + (1.45389826e-4 * ta * ta * va * pa * pa * pa)
    total = total + (2.1750861e-4 * va * va * pa * pa * pa)
    total = total + (-6.66724702e-5 * ta * va * va * pa * pa * pa)
    total = total + (3.3321714e-5 * va * va * va * pa * pa * pa)
    total = total + (-2.26921615e-3 * dtm * pa * pa * pa)
    total = total + (3.80261982e-4 * ta * dtm * pa * pa * pa)
    total = total + (-5.45314314e-9 * ta * ta * dtm * pa * pa * pa)
    total = total + (-7.96355448e-4 * va * dtm * pa * pa * pa)
    total = total + (2.53458034e-5 * ta * va * dtm * pa * pa * pa)
    total = total + (-6.31223658e-6 * va * va * dtm * pa * pa * pa)
    total = total + (3.02122035e-4 * dtm * dtm * pa * pa * pa)
    total = total + (-4.77403547e-6 * ta * dtm * dtm * pa * pa * pa)
    total = total + (1.73825715e-6 * va * dtm * dtm * pa * pa * pa)
    total = total + (-4.09087898e-7 * dtm * dtm * dtm * pa * pa * pa)
    total = total + (6.14155345e-1 * pa * pa * pa * pa)
    total = total + (-6.16755931e-2 * ta * pa * pa * pa * pa)
    total = total + (1.33374846e-3 * ta * ta * pa * pa * pa * pa)
    total = total + (3.55375387e-3 * va * pa * pa * pa * pa)
    total = total + (-5.13027851e-4 * ta * va * pa * pa * pa * pa)
    total = total + (1.02449757e-4 * va * va * pa * pa * pa * pa)
    total = total + (-1.48526421e-3 * dtm * pa * pa * pa * pa)
    total = total + (-4.11469183e-5 * ta * dtm * pa * pa * pa * pa)
    total = total + (-6.80434415e-6 * va * dtm * pa * pa * pa * pa)
    total = total + (-9.77675906e-6 * dtm * dtm * pa * pa * pa * pa)
    total = total + (8.82773108e-2 * pa * pa * pa * pa * pa)
    total = total + (-3.01859306e-3 * ta * pa * pa * pa * pa * pa)
    total = total + (1.04452989e-3 * va * pa * pa * pa * pa * pa)
    total = total + (2.47090539e-4 * dtm * pa * pa * pa * pa * pa)
    total = total + (1.48348065e-3 * pa * pa * pa * pa * pa * pa)
    return total
# ---------------------------------------------------------------------------
# UTCI
# ---------------------------------------------------------------------------

def utci(t, rh, wind, tmrt):
    """
    Universal Thermal Climate Index (degC), 6th-order polynomial of Broede et al. (2012); UTCI() of
    biometeo.js. Wind at 10 m is limited to 0.5-17 m/s. NaN where t < -50 or > 60 degC,
    tmrt < t - 30 or > t + 70, or rh outside 0-100 (9999 in biometeo.js).

    References: Broede P. et al. (2012). Deriving the operational procedure for the Universal Thermal
    Climate Index (UTCI). Int. J. Biometeorol. 56: 481-494. doi:10.1007/s00484-011-0454-1
    """
    ta, rh, va, tmrt = np.broadcast_arrays(_f(t), _f(rh), _f(wind), _f(tmrt))
    pa = (es(ta) / 10.0) * rh / 100.0
    va = np.where(va < 0.51, 0.5, va)
    va = np.where(va > 17, 17.0, va)
    dtm = tmrt - ta
    total = _utci_polynomial(ta, va, dtm, pa)
    bad = (ta < -50.0) | (ta > 60.0) | (tmrt < ta - 30.0) | (tmrt > ta + 70.0) | (rh < 0.0) | (rh > 100.0)
    return two_dec(np.where(bad, np.nan, total))


def utci_class(utci_value):
    """UTCI assessment class 1-10 (Blazejczyk et al., 2013) for UTCI values; NaN for NaN."""
    u = _f(utci_value)
    c = np.digitize(u, [-40, -27, -13, 0, 9, 26, 32, 38, 46], right=True) + 1.0
    return np.where(np.isnan(u), np.nan, c)


# ---------------------------------------------------------------------------
# Heat index, humidex, NET, THI, wind chill
# ---------------------------------------------------------------------------

def heatindex(t, rh):
    """
    Heat index (degC), NWS Rothfusz regression with its low/high humidity adjustments and the simple
    formula below 27 degC; heatindex() of biometeo.js.

    References: Rothfusz L.P. (1990). NWS Technical Attachment SR 90-23; Steadman R.G. (1979).
    J. Appl. Meteor. 18: 861-873.
    """
    t, rh = np.broadcast_arrays(_f(t), _f(rh))
    tf = t * 9 / 5 + 32
    hif = (-42.379 + 2.04901523 * tf + 10.1433127 * rh - 0.22475541 * tf * rh - 6.83783 * 0.001 * tf ** 2
           - 5.481717 * 0.01 * rh ** 2 + 1.22874 * 0.001 * tf ** 2 * rh + 8.5282 * 0.0001 * tf * rh ** 2
           - 1.99 * 0.000001 * tf ** 2 * rh ** 2)
    with np.errstate(invalid="ignore"):
        low = hif - ((13 - rh) / 4) * np.sqrt((17 - np.abs(tf - 95)) / 17)
    high = hif - ((rh - 85) / 10) * ((87 - tf) / 5)
    simple = 0.5 * (tf + 61.0 + (tf - 68.0) * 1.2 + rh * 0.094)
    resf = np.where((t < 44) & (t > 27) & (rh < 13), low,
           np.where((t < 31) & (t > 27) & (rh > 85), high,
           np.where(t > 27, hif, simple)))
    return two_dec((5 / 9) * (resf - 32))


def heatindex_class(hi):
    """Heat index class 1-5 (US NWS) for heat index values in degC: limits 80, 90, 103, 125 F."""
    hif = _f(hi) * 9 / 5 + 32
    c = np.digitize(hif, [80, 90, 103, 125]) + 1.0
    return np.where(np.isnan(hif), np.nan, c)


def humidex(t, rh):
    """
    Humidex (Masterton & Richardson, 1979); humidex() of biometeo.js. NaN outside valid ranges.
    """
    t, rh = np.broadcast_arrays(_f(t), _f(rh))
    e = (rh / 100.0) * (6.105 * np.exp((t * 17.27) / (237.7 + t)))
    h = t + 0.5555 * (e - 10.0)
    bad = (rh > 100.1) | (rh < 0.0) | (t > 100.0) | (t < -100.0)
    return two_dec(np.where(bad, np.nan, h))


def humidex_class(hx):
    """Humidex class 1-4 (Environment Canada): < 30, 30-39, 40-45, > 45."""
    h = _f(hx)
    c = np.where(h < 30, 1.0, np.where(h < 40, 2.0, np.where(h <= 45, 3.0, 4.0)))
    return np.where(np.isnan(h), np.nan, c)


def net_index(t, rh, wind):
    """
    Normal Effective Temperature (degC), Missenard / Gregorczuk & Cena / Li & Chan (2000);
    net_index() of biometeo.js. NaN outside valid ranges.
    """
    t, rh, wind = np.broadcast_arrays(_f(t), _f(rh), _f(wind))
    with np.errstate(invalid="ignore"):
        net = 37 - (37 - t) / (0.68 - 0.0014 * rh + 1 / (1.76 + 1.4 * wind ** 0.75)) - 0.29 * t * (1.0 - 0.01 * rh)
    bad = (rh > 100.1) | (rh < 0.0) | (wind > 130.0) | (wind < 0.0) | (t > 100.0) | (t < -100.0)
    return two_dec(np.where(bad, np.nan, net))


def net_class(net):
    """NET class 1-7 for Central Europe (Blazejczyk et al., 2012): limits 1, 9, 17, 21, 23, 27 degC."""
    n = _f(net)
    c = np.digitize(n, [1, 9, 17, 21, 23, 27]) + 1.0
    return np.where(np.isnan(n), np.nan, c)


def thi(t, rh):
    """
    Thom's discomfort index in the Giles et al. (1990) form (degC): t - 0.55 (1 - 0.01 rh)(t - 14.5);
    THI() of biometeo.js (0.4).
    """
    t, rh = np.broadcast_arrays(_f(t), _f(rh))
    return t - 0.55 * (1 - 0.01 * rh) * (t - 14.5)


def thi_class(di):
    """Class 1-6 of Thom's discomfort index (Giles et al., 1990): limits 21, 24, 27, 29, 32 degC."""
    d = _f(di)
    c = np.digitize(d, [21, 24, 27, 29, 32]) + 1.0
    return np.where(np.isnan(d), np.nan, c)


def windchill(t, wind):
    """
    Wind chill equivalent temperature (degC), Osczevski & Bluestein (2005); windchill() of
    biometeo.js. Wind in m/s (minimum 1.3 m/s), converted to km/h.
    """
    t, wind = np.broadcast_arrays(_f(t), _f(wind))
    v = np.where(wind < 1.3, 1.3, wind) * 3.6
    return 13.12 + 0.6215 * t - 11.37 * v ** 0.16 + 0.3965 * t * v ** 0.16


def windchill_class(wct):
    """
    Wind chill risk class 1-7 (Environment and Climate Change Canada) on the rounded wind chill:
    1 >= 1, 2 0..-9, 3 -10..-27, 4 -28..-39, 5 -40..-47, 6 -48..-54, 7 <= -55; windchill_cla().
    """
    w = js_round(wct)
    c = np.where(w >= 1, 1.0, np.where(w >= -9, 2.0, np.where(w >= -27, 3.0, np.where(w >= -39, 4.0,
        np.where(w >= -47, 5.0, np.where(w >= -54, 6.0, 7.0))))))
    return np.where(np.isnan(w), np.nan, c)


# ---------------------------------------------------------------------------
# Mean radiant temperature and sun position
# ---------------------------------------------------------------------------

def mrt_globe(t, tg, wind, diam_globe=0.05, emis_globe=0.97):
    """
    Mean radiant temperature (degC) from globe temperature, ISO 7726: convective coefficient as the
    larger of forced (6.3 v^0.6 / D^0.4) and natural (1.4 (|tg - t| / D)^0.25) convection;
    mrt_globe() of biometeo.js. Diameter in metres.
    """
    t, tg, wind = np.broadcast_arrays(_f(t), _f(tg), _f(wind))
    d = _f(diam_globe)
    dt = tg - t
    hc = np.maximum(6.3 * wind ** 0.6 / d ** 0.4, 1.4 * (np.abs(dt) / d) ** 0.25)
    x = (tg + 273.15) ** 4 + (hc / (_f(emis_globe) * 5.67e-8)) * dt
    with np.errstate(invalid="ignore"):
        return np.where(x > 0, x ** 0.25 - 273.15, np.nan)


def mrt_solar_proj(t, rh, solar, sunelev, albedo=0.3, emis_sfc=0.97, fdir=0.8):
    """
    Mean radiant temperature (degC) from global solar radiation (W/m2) and sun elevation (degrees),
    with the projected area factor of Jendritzky et al. (1990); mrt_solar_proj() of biometeo.js (0.4).
    """
    t, rh, solar, sunelev = np.broadcast_arrays(_f(t), _f(rh), _f(solar), _f(sunelev))
    rshort = solar * fdir
    rdiffuse = solar - rshort
    sig = 5.67e-8
    emiair = emis_atm(t + 273.15, rh / 100)
    tsk = t + 273.15
    ratio = 0.0429 * np.sin(sunelev * RAD) + 0.345 * np.cos(sunelev * RAD)
    proj = 0.308 * np.cos(RAD * (sunelev * (0.998 - sunelev ** 2.0 / 50000.0)))
    x = (emiair * tsk ** 4 + ((1 - albedo) * rdiffuse) / (sig * emis_sfc)
         + (1 - albedo) * proj * ratio * ((rshort - rdiffuse) / (sig * emis_sfc)))
    with np.errstate(invalid="ignore"):
        return x ** 0.25 - 273.15


def sun_elevation(time_utc, lat, lon):
    """
    Solar elevation (degrees, without refraction) with the PSA algorithm (Blanco-Muriel et al., 2001),
    as in sun_data() of biometeo.js but always in UTC. time_utc: numpy datetime64 (scalar or array).

    References: Blanco-Muriel M. et al. (2001). Computing the solar vector. Solar Energy 70(5): 431-441.
    """
    tt = np.asarray(time_utc, dtype="datetime64[ms]")
    ms = tt.astype("int64").astype(float)
    hours = (ms / 3.6e6) % 24.0
    jd = ms / 86400000.0 - 0.5 + 2440588
    n = jd - 2451545
    omega = 2.1429 - 0.0010394594 * n
    mean_lon = 4.895063 + 0.017202791698 * n
    mean_anom = 6.24006 + 0.0172019699 * n
    ecl_lon = mean_lon + 0.03341607 * np.sin(mean_anom) + 3.4894e-4 * np.sin(2 * mean_anom) - 1.134e-4 - 2.03e-5 * np.sin(omega)
    ecl_obl = 0.4090928 - 6.214e-9 * n + 3.96e-5 * np.cos(omega)
    ra = np.arctan2(np.cos(ecl_obl) * np.sin(ecl_lon), np.cos(ecl_lon))
    ra = np.where(ra < 0, ra + 2 * np.pi, ra)
    dec = np.arcsin(np.sin(ecl_obl) * np.sin(ecl_lon))
    gmst = 6.6974243242 + 0.0657098283 * n + hours
    lmst = (15 * gmst + _f(lon)) * RAD
    ha = lmst - ra
    latr = _f(lat) * RAD
    zen = np.arccos(np.cos(latr) * np.cos(ha) * np.cos(dec) + np.sin(dec) * np.sin(latr))
    zen = zen + (6371.01 / 149597890) * np.sin(zen)
    return 90.0 - zen / RAD


# ---------------------------------------------------------------------------
# PET classes
# ---------------------------------------------------------------------------

def pet_class(pet):
    """PET class 1-9 (Matzarakis & Mayer, 1996): limits 4, 8, 13, 18, 23, 29, 35, 41 degC."""
    p = _f(pet)
    c = np.digitize(p, [4, 8, 13, 18, 23, 29, 35, 41]) + 1.0
    return np.where(np.isnan(p), np.nan, c)


# ---------------------------------------------------------------------------
# Labels of the classes (same texts as the *_class_text functions of biometeo.js)
# ---------------------------------------------------------------------------

CLASSES = {
    "utci": {
        "it": ["stress da freddo estremo", "stress da freddo molto forte", "stress da freddo forte",
               "stress da freddo moderato", "stress da freddo lieve", "nessuno stress termico",
               "stress da caldo moderato", "stress da caldo forte", "stress da caldo molto forte",
               "stress da caldo estremo"],
        "en": ["extreme cold stress", "very strong cold stress", "strong cold stress", "moderate cold stress",
               "slight cold stress", "no thermal stress", "moderate heat stress", "strong heat stress",
               "very strong heat stress", "extreme heat stress"],
    },
    "heatindex": {
        "it": ["nessun avviso", "attenzione", "estrema attenzione", "pericolo", "pericolo estremo"],
        "en": ["no warning", "caution", "extreme caution", "danger", "extreme danger"],
    },
    "humidex": {
        "it": ["disagio lieve o nullo", "un certo disagio", "forte disagio: evitare sforzi",
               "pericoloso: possibile colpo di calore"],
        "en": ["little or no discomfort", "some discomfort", "great discomfort; avoid exertion",
               "dangerous; heat stroke possible"],
    },
    "net": {
        "it": ["molto freddo", "freddo", "fresco", "gradevole", "confortevole", "caldo moderato", "caldo"],
        "en": ["very cold", "cold", "cool", "fresh", "comfortable", "warm", "hot"],
    },
    "thi": {
        "it": ["nessun disagio", "meno del 50% della popolazione avverte disagio",
               "oltre il 50% della popolazione avverte disagio",
               "la maggior parte della popolazione avverte disagio", "tutti avvertono un forte stress",
               "stato di emergenza sanitaria"],
        "en": ["no discomfort", "less than 50% of the population feels discomfort",
               "more than 50% of the population feels discomfort", "most of the population feels discomfort",
               "everyone feels severe stress", "state of medical emergency"],
    },
    "windchill": {
        "it": ["nessun rischio", "rischio basso", "rischio moderato", "rischio alto: congelamento in 10-30 minuti",
               "rischio molto alto: congelamento in 5-10 minuti", "rischio grave: congelamento in 2-5 minuti",
               "rischio estremo: congelamento in meno di 2 minuti"],
        "en": ["no risk", "low risk", "moderate risk", "high risk: frostbite in 10-30 minutes",
               "very high risk: frostbite in 5-10 minutes", "severe risk: frostbite in 2-5 minutes",
               "extreme risk: frostbite in less than 2 minutes"],
    },
    "pet": {
        "it": ["molto freddo - stress da freddo estremo", "freddo - stress da freddo forte",
               "fresco - stress da freddo moderato", "leggermente fresco - stress da freddo lieve",
               "confortevole - nessuno stress termico", "leggermente caldo - stress da caldo lieve",
               "caldo - stress da caldo moderato", "molto caldo - stress da caldo forte",
               "estremamente caldo - stress da caldo estremo"],
        "en": ["very cold - extreme cold stress", "cold - strong cold stress", "cool - moderate cold stress",
               "slightly cool - slight cold stress", "comfortable - no thermal stress",
               "slightly warm - slight heat stress", "warm - moderate heat stress", "hot - strong heat stress",
               "very hot - extreme heat stress"],
    },
}


def class_labels(index, lang="it"):
    """List of the class labels of an index ("utci", "heatindex", "humidex", "net", "thi", "windchill", "pet")."""
    table = CLASSES[index]
    return list(table.get(lang, table["it"]))


def class_text(classes, index, lang="it"):
    """Labels (object array) for an array of class numbers; '' where the class is NaN."""
    labels = np.array([""] + class_labels(index, lang), dtype=object)
    c = _f(classes)
    idx = np.where(np.isnan(c), 0, c).astype(int)
    return labels[idx]
