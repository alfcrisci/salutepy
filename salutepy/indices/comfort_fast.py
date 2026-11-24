"""
PMV (ISO 7730), clomin_7730 e clomax_7730 compilati con numba, fedeli a biometeo.js:

* PMV_ISO7730 restituisce il PMV arrotondato a 2 decimali (TwoDec, Math.round) e NaN se la
  temperatura dell'abito non converge (in JS: "Max iterations exceeded");
* clomin/clomax confrontano il PMV arrotondato e restituiscono il clo *dopo* l'ultimo passo
  del ciclo do...while (come il JS), arrotondato a 1 decimale.

Senza numba le stesse funzioni girano in Python puro (lente).
"""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit, prange

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

    prange = range


@njit(cache=True)
def _js_round(x):
    return math.floor(x + 0.5)


@njit(cache=True)
def _pmv_raw(t, rh, wind, trad, M, W, clo):
    pa = rh * 10.0 * math.exp(16.6536 - 4030.183 / (t + 235.0))
    icl = 0.155 * clo
    mw = M - W
    if icl <= 0.078:
        fcl = 1.0 + 1.29 * icl
    else:
        fcl = 1.05 + 0.645 * icl
    hcf = 12.1 * math.sqrt(wind)
    taa = t + 273.0
    tra = trad + 273.0
    tcla = taa + (35.5 - t) / (3.5 * icl + 0.1)
    p1 = icl * fcl
    p2 = p1 * 3.96
    p3 = p1 * 100.0
    p4 = p1 * taa
    p5 = 308.7 - 0.028 * mw + p2 * (tra / 100.0) ** 4
    xn = tcla / 100.0
    xf = tcla / 50.0
    eps = 0.00015
    n = 0
    hc = hcf
    while abs(xn - xf) > eps:
        xf = (xf + xn) / 2.0
        hcn = 2.38 * abs(100.0 * xf - taa) ** 0.25
        hc = hcf if hcf > hcn else hcn
        xn = (p5 + p4 * hc - p2 * xf ** 4) / (100.0 + p3 * hc)
        n += 1
        if n > 150:
            return np.nan
    tcl = 100.0 * xn - 273.0
    hl1 = 3.05 * 0.001 * (5733.0 - 6.99 * mw - pa)
    hl2 = 0.42 * (mw - 58.15) if mw > 58.15 else 0.0
    hl3 = 1.7 * 0.00001 * M * (5867.0 - pa)
    hl4 = 0.0014 * M * (34.0 - t)
    hl5 = 3.96 * fcl * (xn ** 4 - (tra / 100.0) ** 4)
    hl6 = fcl * hc * (tcl - t)
    ts = 0.303 * math.exp(-0.036 * M) + 0.028
    return ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)


@njit(cache=True)
def _pmv(t, rh, wind, trad, M, W, clo):
    p = _pmv_raw(t, rh, wind, trad, M, W, clo)
    if math.isnan(p):
        return p
    return _js_round(100.0 * p) / 100.0


@njit(cache=True)
def _clomin(t, rh, wind, trad, M, W):
    clo = 0.1
    while True:
        p = _pmv(t, rh, wind, trad, M, W, clo)
        clo = clo + 0.1
        if not (p < -0.5) or clo > 1.0e5:
            break
    return _js_round(10.0 * clo) / 10.0


@njit(cache=True)
def _clomax(t, rh, wind, trad, M, W):
    clo = 5.0
    while True:
        p = _pmv(t, rh, wind, trad, M, W, clo)
        clo = clo - 0.1
        if not (p > 0.5) or clo < -1.0e5:
            break
    return _js_round(10.0 * clo) / 10.0


@njit(parallel=True, cache=True)
def _pmv_vec(t, rh, wind, trad, M, W, clo, out):
    for i in prange(t.size):
        if math.isnan(t[i]) or math.isnan(rh[i]) or math.isnan(wind[i]) or math.isnan(trad[i]):
            out[i] = np.nan
        else:
            out[i] = _pmv(t[i], rh[i], wind[i], trad[i], M[i], W[i], clo[i])


@njit(parallel=True, cache=True)
def _clo_vec(t, rh, wind, trad, M, W, which, out):
    for i in prange(t.size):
        if math.isnan(t[i]) or math.isnan(rh[i]) or math.isnan(wind[i]) or math.isnan(trad[i]):
            out[i] = np.nan
        elif which == 0:
            out[i] = _clomin(t[i], rh[i], wind[i], trad[i], M[i], W[i])
        else:
            out[i] = _clomax(t[i], rh[i], wind[i], trad[i], M[i], W[i])


def _prep(*args):
    arrs = np.broadcast_arrays(*[np.asarray(a, dtype=np.float64) for a in args])
    shape = arrs[0].shape
    return shape, [np.ascontiguousarray(a).ravel() for a in arrs]


def pmv_iso7730(t, rh, wind, trad, M=58.15, W=0.0, clo=0.5):
    """PMV ISO 7730 come biometeo.js PMV_ISO7730 (2 decimali; NaN se non converge)."""
    shape, a = _prep(t, rh, wind, trad, M, W, clo)
    out = np.empty(a[0].size)
    _pmv_vec(*a, out)
    return out.reshape(shape)


def clomin_7730(t, rh, wind, trad, M=58.15, W=0.0):
    """clomin_7730 di biometeo.js: clo minimo per PMV >= -0.5."""
    shape, a = _prep(t, rh, wind, trad, M, W)
    out = np.empty(a[0].size)
    _clo_vec(*a, 0, out)
    return out.reshape(shape)


def clomax_7730(t, rh, wind, trad, M=58.15, W=0.0):
    """clomax_7730 di biometeo.js: clo massimo per PMV <= 0.5."""
    shape, a = _prep(t, rh, wind, trad, M, W)
    out = np.empty(a[0].size)
    _clo_vec(*a, 1, out)
    return out.reshape(shape)
