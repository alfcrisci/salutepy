"""
Physiological Equivalent Temperature (PET), original Hoeppe / VDI 3787 Part 2 algorithm.

Python port of PET_Hoppe() of biometeo.js (rBiometeo 0.4), which translates the PETBER routine of
P. Hoeppe as distributed in VDI 3787 Part 2 and in PALM-4U (Copyright 2018 Deutscher Wetterdienst,
GNU GPL v3 or later). This module is therefore distributed under the GNU GPL v3 or later.

Standard person: male, 35 years, 75 kg, 1.75 m, 80 W of activity, 0.9 clo, standing.
The computation is iterative and done pixel by pixel; with numba installed it is compiled and run
in parallel over the array (``pet_hoppe``), otherwise a pure Python loop is used (slow).

References
----------
Hoeppe P. (1999). The physiological equivalent temperature - a universal index for the
biometeorological assessment of the thermal environment. Int. J. Biometeorol. 43: 71-75.
VDI 3787 Part 2 (2008, rev. 2022). Environmental meteorology - Methods for the human-biometeorological
evaluation of the thermal component of the climate. Beuth, Berlin.
"""

from __future__ import annotations

import math

import numpy as np

try:  # numba is optional
    from numba import njit, prange

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

    prange = range

_G = (-2.8365744e3, -6.028076559e3, 1.954263612e1, -2.737830188e-2, 1.6261698e-5, 7.0229056e-10, -1.8680009e-13)


@njit(cache=True)
def _es(t):
    tk = t + 273.15
    e = 2.7150305 * math.log(tk)
    e += _G[0] * tk ** -2 + _G[1] * tk ** -1 + _G[2] + _G[3] * tk + _G[4] * tk ** 2 + _G[5] * tk ** 3 + _G[6] * tk ** 4
    return math.exp(e) * 0.01


@njit(cache=True)
def pet_hoppe_scalar(t, rh, wind, tmrt, pair):
    """PET (degC) for one point; NaN for missing inputs. Same algorithm as PET_Hoppe() in biometeo.js."""
    if math.isnan(t) or math.isnan(rh) or math.isnan(wind) or math.isnan(tmrt) or math.isnan(pair):
        return math.nan
    po = 1013.25
    rob = 1.06
    cb = 3640.0
    emsk = 0.99
    emcl = 0.95
    evap = 2.42e6
    sigm = 5.67e-8
    cair = 1010.0
    age = 35.0
    mbody = 75.0
    ht = 1.75
    work = 80.0
    eta = 0.0
    icl = 0.9
    fcl = 1.15
    vpa = rh * _es(t) / 100.0
    met = 3.45 * mbody ** 0.75 * (1 + 0.004 * (30 - age) + 0.01 * ((ht * 100) / mbody ** (1 / 3) - 43.4))
    met = work + met
    h = met * (1 - eta)
    tex = 0.47 * t + 21.0
    rtv = 1.44e-6 * met
    eres = cair * (t - tex) * rtv
    vpex = 6.11 * 10 ** ((7.45 * tex) / (235 + tex))
    erel = ((0.623 * evap) / pair) * (vpa - vpex) * rtv
    ere = eres + erel
    adu = 0.203 * mbody ** 0.425 * ht ** 0.725
    hc = (2.67 + 6.5 * wind ** 0.67) * (pair / po) ** 0.55
    feff = 0.725
    facl = (-2.36 + 173.51 * icl - 100.76 * icl * icl + 19.28 * icl ** 3) / 100
    if facl > 1:
        facl = 1.0
    rcl = icl / 6.45 / facl
    y = 0.1
    if icl >= 2:
        y = 1.0
    if 0.6 < icl < 2:
        y = (ht - 0.2) / ht
    if 0.3 < icl <= 0.6:
        y = 0.5
    r2 = (adu * (fcl - 1 + facl)) / (2 * 3.14 * ht * y)
    r1 = (facl * adu) / (2 * 3.14 * ht * y)
    di = r2 - r1
    c = np.zeros(11)
    tcore = np.zeros(8)
    rdsk = 0.79e7
    rdcl = 0.0
    tsk = 34.0
    tcl = t
    acl = adu * facl + adu * (fcl - 1)
    esw = 0.0
    wetsk = 0.0
    vpts = 0.0
    for j in range(1, 8):
        tsk = 34.0
        count1 = 0
        tcl = (t + tmrt + tsk) / 3
        count3 = 1
        enbal2 = 0.0
        wetsk = 0.0
        vb = 0.0
        while True:
            acl = adu * facl + adu * (fcl - 1)
            rclo2 = emcl * sigm * ((tcl + 273.2) ** 4 - (tmrt + 273.2) ** 4) * feff
            htcl = (6.28 * ht * y * di) / (rcl * math.log(r2 / r1) * acl)
            tsk = (1 / htcl) * (hc * (tcl - t) + rclo2) + tcl
            aeff = adu * feff
            rbare = aeff * (1 - facl) * emsk * sigm * ((tmrt + 273.2) ** 4 - (tsk + 273.2) ** 4)
            rclo = feff * acl * emcl * sigm * ((tmrt + 273.2) ** 4 - (tcl + 273.2) ** 4)
            rsum = rbare + rclo
            cbare = hc * (t - tsk) * adu * (1 - facl)
            cclo = hc * (t - tcl) * acl
            csum = cbare + cclo
            c[0] = h + ere
            c[1] = adu * rob * cb
            c[2] = 18 - 0.5 * tsk
            c[3] = 5.28 * adu * c[2]
            c[4] = 0.0208 * c[1]
            c[5] = 0.76075 * c[1]
            c[6] = c[3] - c[5] - tsk * c[4]
            c[7] = -c[0] * c[2] - tsk * c[3] + tsk * c[5]
            c[8] = c[6] * c[6] - 4 * c[4] * c[7]
            c[9] = 5.28 * adu - c[5] - c[4] * tsk
            c[10] = c[9] * c[9] - 4 * c[4] * (c[5] * tsk - c[0] - 5.28 * adu * tsk)
            if tsk == 36:
                tsk = 36.01
            tcore[7] = c[0] / (5.28 * adu + (c[1] * 6.3) / 3600) + tsk
            tcore[3] = c[0] / (5.28 * adu + (c[1] * 6.3) / 3600 / (1 + 0.5 * (34 - tsk))) + tsk
            if c[10] >= 0:
                tcore[6] = (-c[9] - math.sqrt(c[10])) / (2 * c[4])
                tcore[1] = (-c[9] + math.sqrt(c[10])) / (2 * c[4])
            if c[8] >= 0:
                tcore[2] = (-c[6] + math.sqrt(abs(c[8]))) / (2 * c[4])
                tcore[5] = (-c[6] - math.sqrt(abs(c[8]))) / (2 * c[4])
                tcore[4] = c[0] / (5.28 * adu + (c[1] * 1) / 40) + tsk
            tbody = 0.1 * tsk + 0.9 * tcore[j]
            swm = (304.94 * (tbody - 36.6) * adu) / 3600000
            vpts = 6.11 * 10 ** ((7.45 * tsk) / (235 + tsk))
            if tbody <= 36.6:
                swm = 0.0
            sw = swm
            eswphy = -sw * evap
            he = (0.633 * hc) / (pair * cair)
            fec = 1 / (1 + 0.92 * hc * rcl)
            eswpot = he * (vpa - vpts) * adu * evap * fec
            wetsk = eswphy / eswpot
            if wetsk > 1:
                wetsk = 1.0
            eswdif = eswphy - eswpot
            if eswdif <= 0:
                esw = eswpot
            if eswdif > 0:
                esw = eswphy
            if esw > 0:
                esw = 0.0
            ed = (evap / (rdsk + rdcl)) * adu * (1 - wetsk) * (vpa - vpts)
            vb1 = 34 - tsk
            vb2 = tcore[j] - 36.6
            if vb2 < 0:
                vb2 = 0.0
            if vb1 < 0:
                vb1 = 0.0
            vb = (6.3 + 75 * vb2) / (1 + 0.5 * vb1)
            enbal = h + ed + ere + esw + csum + rsum
            xx = 0.001
            if count1 == 0:
                xx = 1.0
            elif count1 == 1:
                xx = 0.1
            elif count1 == 2:
                xx = 0.01
            if enbal > 0:
                tcl = tcl + xx
            if enbal < 0:
                tcl = tcl - xx
            skip = False
            if (enbal <= 0 and enbal2 > 0) or (enbal >= 0 and enbal2 < 0):
                skip = True
            else:
                enbal2 = enbal
                count3 = count3 + 1
            if count3 > 200 or skip:
                if count1 < 3:
                    count1 = count1 + 1
                    enbal2 = 0.0
                else:
                    break
        if count1 == 3:
            if j == 2 or j == 5:
                if not (tcore[j] >= 36.6 and tsk <= 34.05):
                    continue
            elif j == 6 or j == 1:
                if c[10] < 0:
                    continue
                if not (tcore[j] >= 36.6 and tsk > 33.85):
                    continue
            elif j == 3:
                if not (tcore[j] < 36.6 and tsk <= 34.0):
                    continue
            elif j == 7:
                if not (tcore[j] < 36.6 and tsk > 34.0):
                    continue
        if j != 4 and vb >= 91:
            continue
        if j == 4 and vb < 89:
            continue
        break
    # reference room
    tx = t
    enbal2b = 0.0
    hcr = (2.67 + 6.5 * 0.1 ** 0.67) * (pair / po) ** 0.55
    aeffr = adu * feff
    for k in range(4):
        step = 1.0
        if k == 1:
            step = 0.1
        elif k == 2:
            step = 0.01
        elif k == 3:
            step = 0.001
        for _ in range(100000):
            rbare2 = aeffr * (1 - facl) * emsk * sigm * ((tx + 273.2) ** 4 - (tsk + 273.2) ** 4)
            rclo3 = feff * acl * emcl * sigm * ((tx + 273.2) ** 4 - (tcl + 273.2) ** 4)
            cbare2 = hcr * (tx - tsk) * adu * (1 - facl)
            cclo2 = hcr * (tx - tcl) * acl
            ed2 = (evap / (rdsk + rdcl)) * adu * (1 - wetsk) * (12 - vpts)
            tex2 = 0.47 * tx + 21
            eres2 = cair * (tx - tex2) * rtv
            vpex2 = 6.11 * 10 ** ((7.45 * tex2) / (235 + tex2))
            erel2 = ((0.623 * evap) / pair) * (12 - vpex2) * rtv
            enbalr = h + ed2 + (eres2 + erel2) + esw + (cbare2 + cclo2) + (rbare2 + rclo3)
            if enbalr > 0:
                tx = tx - step
            if enbalr < 0:
                tx = tx + step
            if enbalr <= 0 and enbal2b > 0:
                break
            if enbalr >= 0 and enbal2b < 0:
                break
            enbal2b = enbalr
    return math.floor(tx * 100 + 0.5) / 100


@njit(parallel=True, cache=True)
def _pet_hoppe_flat(t, rh, wind, tmrt, pair):
    out = np.empty(t.size)
    for i in prange(t.size):
        out[i] = pet_hoppe_scalar(t[i], rh[i], wind[i], tmrt[i], pair[i])
    return out


def pet_hoppe(t, rh, wind, tmrt, pair=1013.25):
    """
    PET (degC) with the Hoeppe / VDI 3787 algorithm for arrays (broadcast together).

    Parameters: air temperature (degC), relative humidity (%), wind speed (m/s), mean radiant
    temperature (degC), air pressure (hPa, default 1013.25). Classes: biometeo_np.pet_class.
    """
    arrs = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (t, rh, wind, tmrt, pair)))
    shape = arrs[0].shape
    flat = [np.ascontiguousarray(a).ravel() for a in arrs]
    return _pet_hoppe_flat(*flat).reshape(shape)
