"""
Fast WBGT of Liljegren and of Bernard, compiled with numba and run in parallel.

Port of HeatStress::wbgt.Liljegren (with fTg, fTnwb, calZenith and the heat-transfer helpers) and
HeatStress::wbgt.Bernard (R package HeatStress 1.0.8.1, A. Casanueva). The 1D minimisations use
``brent_fmin``, a translation of Brent_fmin of R (src/library/stats/src/optimize.c), the routine
behind ``stats::optimize``: with the same interval and tolerance the results agree with R to
about 1e-6 degC (see tests/test_heatstress.py).

Without numba the same code runs as plain Python (slow). The previous implementation
(salutepy.indices.wbgt.wbgt_liljegren, scipy) is left unchanged.

Differences from R that can be switched on:
* ``hour=True`` uses the time of day of each date (R: hour=FALSE uses 12 UTC for every date, i.e.
  daily data). As in R, only the whole hour is used (minutes are dropped).
* ``solar_time_correction=True`` adds the equation of time and the longitude to the hour angle.
  calZenith of HeatStress sets this offset to 0, so the zenith angle is computed as if every point
  were at longitude 0 (about 45 minutes of solar time off in Italy). Default False = as in R.

References
----------
Liljegren J.C., Carhart R.A., Lawday P., Tschopp S., Sharp R. (2008). Modeling the wet bulb globe
temperature using standard meteorological measurements. J. Occup. Environ. Hyg. 5(10): 645-655.
Bernard T.E., Pourmoghani M. (1999). Prediction of workplace wet bulb global temperature.
Appl. Occup. Environ. Hyg. 14: 126-134. Lemke B., Kjellstrom T. (2012). Calculating workplace WBGT
from meteorological data. Ind. Health 50: 267-278.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit, prange

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

    prange = range

STEFANB = 0.000000056696
CP = 1003.5
M_AIR = 28.97
M_H2O = 18.015
R_GAS = 8314.34
R_AIR = R_GAS / M_AIR
RATIO = CP * M_AIR / M_H2O
PR = CP / (CP + (1.25 * R_AIR))
DBL_EPS = 2.220446049250313e-16


# ---------------------------------------------------------------------------
# helpers (HeatStress)
# ---------------------------------------------------------------------------

@njit(cache=True)
def esat(tk):
    return 1.004 * (6.1121 * math.exp(17.502 * (tk - 273.15) / (tk - 32.18)))


@njit(cache=True)
def emis_atm(tk, rh):
    e = rh * esat(tk)
    return 0.575 * e ** 0.143


@njit(cache=True)
def viscosity(tk):
    omega = (tk / 97 - 2.9) / 0.4 * (-0.034) + 1.048
    return 0.0000026693 * (28.97 * tk) ** 0.5 / (3.617 ** 2 * omega)


@njit(cache=True)
def thermal_cond(tk):
    return (CP + 1.25 * R_AIR) * viscosity(tk)


@njit(cache=True)
def diffusivity(tk, pair):
    pcrit13 = (36.4 * 218) ** (1 / 3)
    tcrit512 = (132 * 647.3) ** (5 / 12)
    tcrit12 = (132 * 647.3) ** 0.5
    mmix = (1 / 28.97 + 1 / 18.015) ** 0.5
    return 0.000364 * (tk / tcrit12) ** 2.334 * pcrit13 * tcrit512 * mmix / (pair / 1013.25) * 0.0001


@njit(cache=True)
def h_sphere_in_air(tk, pair, speed, min_speed, diam):
    density = pair * 100 / (R_AIR * tk)
    if speed < min_speed:
        speed = min_speed
    re = speed * density * diam / viscosity(tk)
    nu = 2 + 0.6 * re ** 0.5 * PR ** 0.3333
    return nu * thermal_cond(tk) / diam


@njit(cache=True)
def h_cylinder_in_air(tk, pair, speed, min_speed, diam):
    density = pair * 100 / (R_AIR * tk)
    if speed < min_speed:
        speed = min_speed
    re = speed * density * diam / viscosity(tk)
    nu = 0.281 * re ** 0.6 * PR ** 0.44
    return nu * thermal_cond(tk) / diam


@njit(cache=True)
def h_evap(tk):
    return (313.15 - tk) / 30 * (-71100) + 2407300


@njit(cache=True)
def dewp2hurs_scalar(tas, dewp):
    if tas < 0:
        h = 100 * math.exp(((17.856 * dewp) / (245.52 + dewp)) - ((17.856 * tas) / (245.52 + tas)))
    else:
        h = 100 * math.exp(((17.368 * dewp) / (238.83 + dewp)) - ((17.368 * tas) / (238.83 + tas)))
    if h > 100:
        h = 100.0
    if h < 0:
        h = 0.0
    return h


# ---------------------------------------------------------------------------
# Brent minimisation (R stats::optimize)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _obj(kind, x, p):
    """Objective functions: 0 = fTg, 1 = fTnwb, 2 = Bernard psychrometric wet bulb."""
    if kind == 0:
        tair, pair, wind, min_speed, radiation, prop_direct, cza, rh, alb_sfc = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]
        tref = 0.5 * (x + tair)
        h = h_sphere_in_air(tref, pair, wind, min_speed, 0.05)
        tg = (0.5 * (emis_atm(tair, rh) * tair ** 4 + 0.999 * tair ** 4) - h / (0.95 * STEFANB) * (x - tair)
              + radiation / (2 * 0.95 * STEFANB) * (1 - 0.05) * (prop_direct * (1 / (2 * cza) - 1) + 1 + alb_sfc))
        return abs(tg ** 0.25 - x)
    if kind == 1:
        tair, pair, wind, min_speed, radiation, prop_direct, zenith, eair, emis_a, density, irad, alb_sfc = (
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11])
        diam_wick = 0.007
        len_wick = 0.0254
        tref = 0.5 * (x + tair)
        fatm = STEFANB * 0.95 * (0.5 * (emis_a * tair ** 4 + 0.999 * tair ** 4) - x ** 4) + (1 - 0.4) * radiation * (
            (1 - prop_direct) * (1 + 0.25 * diam_wick / len_wick)
            + ((math.tan(zenith) / 3.1416) + 0.25 * diam_wick / len_wick) * prop_direct + alb_sfc)
        sc = viscosity(tair) / (density * diffusivity(tref, pair))
        h = h_cylinder_in_air(x, pair, wind, min_speed, diam_wick)
        ewick = esat(x)
        ev = h_evap(x)
        twb = tair - ev / RATIO * (ewick - eair) / (pair - ewick) * (PR / sc) ** 0.56 + fatm / h * irad
        return abs(twb - x)
    tasi, edi = p[0], p[1]
    c1, c2, c3, c4, c5, c6 = 6.106, 17.27, 237.3, 1556.0, 1.484, 1010.0
    ex = math.exp((c2 * x) / (c3 + x))
    return abs(c4 * edi - c5 * edi * x - c4 * c1 * ex + c5 * c1 * ex * x + c6 * (tasi - x))


@njit(cache=True)
def brent_fmin(ax, bx, kind, p, tol):
    """Brent_fmin of R (stats::optimize) on [ax, bx] with absolute tolerance tol."""
    c = (3.0 - math.sqrt(5.0)) * 0.5
    eps = math.sqrt(DBL_EPS)
    a = ax
    b = bx
    v = a + c * (b - a)
    w = v
    x = v
    d = 0.0
    e = 0.0
    fx = _obj(kind, x, p)
    fv = fx
    fw = fx
    tol3 = tol / 3.0
    while True:
        xm = (a + b) * 0.5
        tol1 = eps * abs(x) + tol3
        t2 = tol1 * 2.0
        if abs(x - xm) <= t2 - (b - a) * 0.5:
            break
        p_ = 0.0
        q = 0.0
        r = 0.0
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p_ = (x - v) * q - (x - w) * r
            q = (q - r) * 2.0
            if q > 0.0:
                p_ = -p_
            else:
                q = -q
            r = e
            e = d
        if abs(p_) >= abs(q * 0.5 * r) or p_ <= q * (a - x) or p_ >= q * (b - x):
            if x < xm:
                e = b - x
            else:
                e = a - x
            d = c * e
        else:
            d = p_ / q
            u = x + d
            if u - a < t2 or b - u < t2:
                d = tol1
                if x >= xm:
                    d = -d
        if abs(d) >= tol1:
            u = x + d
        elif d > 0.0:
            u = x + tol1
        else:
            u = x - tol1
        fu = _obj(kind, u, p)
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v = w
            w = x
            x = u
            fv = fw
            fw = fx
            fx = fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v = w
                fv = fw
                w = u
                fw = fu
            elif fu <= fv or v == x or v == w:
                v = u
                fv = fu
    return x


# ---------------------------------------------------------------------------
# Liljegren
# ---------------------------------------------------------------------------

@njit(cache=True)
def cal_zenith(year, doy, hour, lon, lat, use_hour, solar_time_correction):
    """Solar zenith angle (degrees) as HeatStress::calZenith (hour = whole UTC hour)."""
    utc_hour = hour if use_hour else 12.0
    dpy = 366.0 if ((year % 4 == 0 and year % 100 != 0) or year % 400 == 0) else 365.0
    rad_lat = math.pi * lat / 180
    gamma = 2 * math.pi * ((doy - 1) + (utc_hour / 24)) / dpy
    equ_time = 229.18 * (0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
                         - 0.014615 * math.cos(2 * gamma) - 0.040849 * math.sin(2 * gamma))
    decli = (0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma) - 0.006758 * math.cos(2 * gamma)
             + 0.000907 * math.sin(2 * gamma) - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma))
    offset = equ_time + 4 * lon if solar_time_correction else 0.0
    true_solar_time = (utc_hour * 60) + offset
    ha_rad = math.pi * ((true_solar_time / 4) - 180) / 180
    cos_zen = math.sin(rad_lat) * math.sin(decli) + math.cos(rad_lat) * math.cos(decli) * math.cos(ha_rad)
    if cos_zen > 1.0:
        cos_zen = 1.0
    if cos_zen < -1.0:
        cos_zen = -1.0
    return 180.0 * math.acos(cos_zen) / math.pi


@njit(cache=True)
def _clip_zenith(zenith, radiation):
    if zenith <= 0:
        zenith = 0.0000000001
    if radiation > 0 and zenith > 1.57:
        zenith = 1.57
    if radiation > 15 and zenith > 1.54:
        zenith = 1.54
    if radiation > 900 and zenith > 1.52:
        zenith = 1.52
    if radiation < 10 and zenith == 1.57:
        radiation = 0.0
    return zenith, radiation


@njit(cache=True)
def liljegren_point(tas, dewp, wind, radiation, zenith, pair, min_speed, prop_direct, alb_sfc, tol):
    """Returns (Tg, Tnwb) in degC for one point; zenith in radians. Inputs already cleaned as in R."""
    relh = dewp2hurs_scalar(tas, dewp)
    rh = relh * 0.01
    tair = tas + 273.15
    # fTg
    z, rad = _clip_zenith(zenith, radiation)
    p = np.empty(12)
    p[0] = tair
    p[1] = pair
    p[2] = wind
    p[3] = min_speed
    p[4] = rad
    p[5] = prop_direct
    p[6] = math.cos(z)
    p[7] = rh
    p[8] = alb_sfc
    lo = tair - 2
    hi = tair + 10
    tg = brent_fmin(lo, hi, 0, p, tol) - 273.15
    # fTnwb
    tdew = dewp + 273.15
    p[0] = tair
    p[1] = pair
    p[2] = wind
    p[3] = min_speed
    p[4] = rad
    p[5] = prop_direct
    p[6] = z
    p[7] = rh * esat(tair)
    p[8] = emis_atm(tair, rh)
    p[9] = pair * 100 / (tair * R_AIR)
    p[10] = 1.0
    p[11] = alb_sfc
    a = tdew - 1
    b = tair + 1
    if a > b:
        a, b = b, a
    tnwb = brent_fmin(a, b, 1, p, tol) - 273.15
    return tg, tnwb


@njit(parallel=True, cache=True)
def _liljegren_flat(tas, dewp, wind, radiation, year, doy, hour, lon, lat, use_hour, stc, pair, min_speed,
                    prop_direct, alb_sfc, tol, out_wbgt, out_tg, out_tnwb):
    for i in prange(tas.size):
        t = tas[i]
        d = dewp[i]
        w = wind[i]
        r = radiation[i]
        if math.isnan(t) or math.isnan(d) or math.isnan(w) or math.isnan(r):
            out_wbgt[i] = math.nan
            out_tg[i] = math.nan
            out_tnwb[i] = math.nan
            continue
        if r < 0:
            r = 0.0
        if w < 0:
            w = 0.0
        if (d - t) > tol:  # noNAs=TRUE, swap=FALSE
            d = t
        zen_deg = cal_zenith(year[i], doy[i], hour[i], lon[i], lat[i], use_hour, stc)
        tg, tnwb = liljegren_point(t, d, w, r, math.pi * zen_deg / 180, pair, min_speed, prop_direct, alb_sfc, tol)
        out_tg[i] = tg
        out_tnwb[i] = tnwb
        out_wbgt[i] = 0.7 * tnwb + 0.2 * tg + 0.1 * t


def _date_parts(dates, shape):
    d = np.broadcast_to(np.asarray(dates, dtype="datetime64[s]"), shape).ravel()
    years = d.astype("datetime64[Y]").astype(int) + 1970
    doy = (d.astype("datetime64[D]") - d.astype("datetime64[Y]").astype("datetime64[D]")).astype(int) + 1
    hour = ((d - d.astype("datetime64[D]")).astype("timedelta64[h]").astype(int)).astype(float)
    return years.astype(np.int64), doy.astype(float), hour


def wbgt_liljegren(tas, dewp, wind, radiation, dates, lon, lat, tolerance=1e-4, hour=False,
                   solar_time_correction=False, pair=1010.0, min_speed=0.1, prop_direct=0.8, surf_albedo=0.4):
    """
    Outdoor WBGT of Liljegren et al. (2008), as HeatStress::wbgt.Liljegren (noNAs=TRUE, swap=FALSE).

    Parameters: air temperature and dew point (degC), wind speed (m/s), global radiation (W/m2),
    dates (numpy datetime64, UTC), longitude and latitude (degrees); all broadcast together.
    ``hour`` and ``solar_time_correction``: see the module docstring (defaults as in R).

    Returns a dict with ``data`` (WBGT), ``Tnwb`` (natural wet bulb) and ``Tg`` (globe temperature), degC.
    """
    tas, dewp, wind, radiation, lon, lat = np.broadcast_arrays(*(np.asarray(x, dtype=float)
                                                                for x in (tas, dewp, wind, radiation, lon, lat)))
    shape = tas.shape
    year, doy, hr = _date_parts(dates, shape)
    flat = [np.ascontiguousarray(a, dtype=float).ravel() for a in (tas, dewp, wind, radiation)]
    lonf = np.ascontiguousarray(lon, dtype=float).ravel()
    latf = np.ascontiguousarray(lat, dtype=float).ravel()
    n = flat[0].size
    w, g, nw = np.empty(n), np.empty(n), np.empty(n)
    _liljegren_flat(flat[0], flat[1], flat[2], flat[3], year, doy, hr, lonf, latf, bool(hour),
                    bool(solar_time_correction), float(pair), float(min_speed), float(prop_direct),
                    float(surf_albedo), float(tolerance), w, g, nw)
    return {"data": w.reshape(shape), "Tnwb": nw.reshape(shape), "Tg": g.reshape(shape)}


@njit(parallel=True, cache=True)
def _liljegren_elev_flat(tas, dewp, wind, radiation, sunelev, pair, min_speed, prop_direct, alb_sfc, tol,
                         out_wbgt, out_tg, out_tnwb):
    for i in prange(tas.size):
        t = tas[i]
        d = dewp[i]
        w = wind[i]
        r = radiation[i]
        e = sunelev[i]
        if math.isnan(t) or math.isnan(d) or math.isnan(w) or math.isnan(r) or math.isnan(e):
            out_wbgt[i] = math.nan
            out_tg[i] = math.nan
            out_tnwb[i] = math.nan
            continue
        if r < 0:
            r = 0.0
        if w < 0:
            w = 0.0
        if (d - t) > tol:
            d = t
        zen_deg = 90.0 - e
        tg, tnwb = liljegren_point(t, d, w, r, math.pi * zen_deg / 180, pair, min_speed, prop_direct, alb_sfc, tol)
        out_tg[i] = tg
        out_tnwb[i] = tnwb
        out_wbgt[i] = 0.7 * tnwb + 0.2 * tg + 0.1 * t


def wbgt_liljegren_elev(tas, dewp, wind, radiation, sunelev, tolerance=1e-4, pair=1010.0, min_speed=0.1,
                        prop_direct=0.8, surf_albedo=0.4):
    """
    WBGT di Liljegren come ``wbgt_liljegren`` ma con l'altezza del sole (gradi) data direttamente al posto
    di data/ora e coordinate (zenit = 90 - elevazione). Restituisce data/Tnwb/Tg come ``wbgt_liljegren``.
    """
    arrs = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (tas, dewp, wind, radiation, sunelev)))
    shape = arrs[0].shape
    flat = [np.ascontiguousarray(a, dtype=float).ravel() for a in arrs]
    n = flat[0].size
    w, g, nw = np.empty(n), np.empty(n), np.empty(n)
    _liljegren_elev_flat(*flat, float(pair), float(min_speed), float(prop_direct), float(surf_albedo),
                         float(tolerance), w, g, nw)
    return {"data": w.reshape(shape), "Tnwb": nw.reshape(shape), "Tg": g.reshape(shape)}


# ---------------------------------------------------------------------------
# Bernard
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _bernard_flat(tas, dewp, tol, out_wbgt, out_tpwb):
    for i in prange(tas.size):
        t = tas[i]
        d = dewp[i]
        if math.isnan(t) or math.isnan(d):
            out_tpwb[i] = math.nan
            out_wbgt[i] = math.nan
            continue
        if abs(t - d) < tol or (d - t) > tol:  # trivial (RH=100) or dew point above air temperature
            tp = t
        else:
            ed = 6.106 * math.exp((17.27 * d) / (237.3 + d))
            p = np.empty(2)
            p[0] = t
            p[1] = ed
            a = d - 1
            b = t + 1
            if a > b:
                a, b = b, a
            tp = brent_fmin(a, b, 2, p, tol)
        out_tpwb[i] = tp
        out_wbgt[i] = 0.67 * tp + 0.33 * t


def wbgt_bernard(tas, dewp, tolerance=1e-4):
    """
    Indoor/shade WBGT of Bernard (degC), as HeatStress::wbgt.Bernard (noNAs=TRUE, swap=FALSE):
    psychrometric wet bulb from temperature and dew point, WBGT = 0.67 Tpwb + 0.33 tas.
    Returns a dict with ``data`` (WBGT) and ``Tpwb``.
    """
    tas, dewp = np.broadcast_arrays(np.asarray(tas, dtype=float), np.asarray(dewp, dtype=float))
    shape = tas.shape
    t = np.ascontiguousarray(tas).ravel()
    d = np.ascontiguousarray(dewp).ravel()
    w, tp = np.empty(t.size), np.empty(t.size)
    _bernard_flat(t, d, float(tolerance), w, tp)
    return {"data": w.reshape(shape), "Tpwb": tp.reshape(shape)}
