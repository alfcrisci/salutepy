"""
Porting di rBiometeo::PMV_ISO7730, clomin_7730, clomax_7730
(biometeo.js), modello di Fanger per il Predicted Mean Vote (ISO 7730).
"""

from __future__ import annotations

import numpy as np


def _pmv_iso7730_scalar(t: float, rh: float, wind: float, trad: float, M: float, W: float, clo: float) -> float:
    """Versione scalare, porting diretto della funzione JS PMV_ISO7730."""
    pa = rh * 10 * np.exp(16.6536 - 4030.183 / (t + 235))

    icl = 0.155 * clo
    mw = M - W

    fcl = 1 + (1.29 * icl) if icl <= 0.078 else 1.05 + (0.645 * icl)

    hcf = 12.1 * np.sqrt(wind)
    taa = t + 273
    tra = trad + 273
    tcla = taa + (35.5 - t) / (3.5 * icl + 0.1)

    p1 = icl * fcl
    p2 = p1 * 3.96
    p3 = p1 * 100
    p4 = p1 * taa
    p5 = 308.7 - 0.028 * mw + p2 * (tra / 100) ** 4

    xn = tcla / 100
    xf = tcla / 50
    eps = 0.00015

    n = 0
    while abs(xn - xf) > eps:
        xf = (xf + xn) / 2
        hcn = 2.38 * abs(100.0 * xf - taa) ** 0.25
        hc = hcf if hcf > hcn else hcn
        with np.errstate(over="ignore", invalid="ignore"):
            xn = (p5 + p4 * hc - p2 * xf ** 4) / (100 + p3 * hc)
        n += 1
        if n > 150 or not np.isfinite(xn):
            return np.nan

    tcl = 100 * xn - 273

    hl1 = 3.05 * 0.001 * (5733 - (6.99 * mw) - pa)
    hl2 = 0.42 * (mw - 58.15) if mw > 58.15 else 0
    hl3 = 1.7 * 0.00001 * M * (5867 - pa)
    hl4 = 0.0014 * M * (34 - t)
    hl5 = 3.96 * fcl * (xn ** 4 - (tra / 100) ** 4)
    hl6 = fcl * hc * (tcl - t)

    ts = 0.303 * np.exp(-0.036 * M) + 0.028

    pmv = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)

    return round(pmv, 2)


def pmv_iso7730(t, rh, wind, trad, M, W, clo):
    """
    Porting di rBiometeo::PMV_ISO7730 (biometeo.js PMV_ISO7730).
    Predicted Mean Vote (ISO 7730), calcolato per-elemento (iterazione
    non vettorizzabile in numpy puro, come nel sorgente R/JS originale).

    Parameters
    ----------
    t : temperatura aria (degC)
    rh : umidita' relativa (%)
    wind : velocita' vento (m/s)
    trad : temperatura media radiante (degC)
    M : metabolismo (W/m2)
    W : lavoro meccanico esterno (W/m2, tipicamente 0)
    clo : isolamento termico dell'abbigliamento (clo)

    Note
    ----
    Nel codice originale (meteosalute_op_dir_mol01.r) M e clo variano per
    pixel/timestep, mentre W e' costante (0) — qui tutti gli argomenti
    accettano sia scalari che array della stessa forma.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    rh = np.broadcast_to(np.asarray(rh, dtype=float), t.shape)
    wind = np.broadcast_to(np.asarray(wind, dtype=float), t.shape)
    trad = np.broadcast_to(np.asarray(trad, dtype=float), t.shape)
    M = np.broadcast_to(np.asarray(M, dtype=float), t.shape)
    W = np.broadcast_to(np.asarray(W, dtype=float), t.shape)
    clo = np.broadcast_to(np.asarray(clo, dtype=float), t.shape)

    out = np.empty_like(t)
    for i in range(t.size):
        out.flat[i] = _pmv_iso7730_scalar(
            t.flat[i], rh.flat[i], wind.flat[i], trad.flat[i], M.flat[i], W.flat[i], clo.flat[i]
        )
    return out


def clomax_7730(t, rh, wind, trad, M=58.15, W=0):
    """
    Porting di rBiometeo::clomax_7730 (biometeo.js clomax_7730).
    Massimo livello di abbigliamento (clo) tale che PMV <= 0.5, per
    ricerca decrescente da clo=5 con passo -0.1 (fedele al ciclo
    do...while originale, non un'ottimizzazione).
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    rh = np.broadcast_to(np.asarray(rh, dtype=float), t.shape)
    wind = np.broadcast_to(np.asarray(wind, dtype=float), t.shape)
    trad = np.broadcast_to(np.asarray(trad, dtype=float), t.shape)
    M = np.broadcast_to(np.asarray(M, dtype=float), t.shape)
    W = np.broadcast_to(np.asarray(W, dtype=float), t.shape)

    out = np.empty_like(t)
    for i in range(t.size):
        clo = 5.0
        pmv = _pmv_iso7730_scalar(t.flat[i], rh.flat[i], wind.flat[i], trad.flat[i], M.flat[i], W.flat[i], clo)
        while pmv > 0.5:
            clo -= 0.1
            pmv = _pmv_iso7730_scalar(t.flat[i], rh.flat[i], wind.flat[i], trad.flat[i], M.flat[i], W.flat[i], clo)
        out.flat[i] = round(clo, 1)
    return out


def clomin_7730(t, rh, wind, trad, M=58.15, W=0):
    """
    Porting di rBiometeo::clomin_7730 (biometeo.js clomin_7730).
    Minimo livello di abbigliamento (clo) tale che PMV >= -0.5, per
    ricerca crescente da clo=0.1 con passo +0.1 (fedele al ciclo
    do...while originale).
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    rh = np.broadcast_to(np.asarray(rh, dtype=float), t.shape)
    wind = np.broadcast_to(np.asarray(wind, dtype=float), t.shape)
    trad = np.broadcast_to(np.asarray(trad, dtype=float), t.shape)
    M = np.broadcast_to(np.asarray(M, dtype=float), t.shape)
    W = np.broadcast_to(np.asarray(W, dtype=float), t.shape)

    out = np.empty_like(t)
    for i in range(t.size):
        clo = 0.1
        pmv = _pmv_iso7730_scalar(t.flat[i], rh.flat[i], wind.flat[i], trad.flat[i], M.flat[i], W.flat[i], clo)
        while pmv < -0.5:
            clo += 0.1
            pmv = _pmv_iso7730_scalar(t.flat[i], rh.flat[i], wind.flat[i], trad.flat[i], M.flat[i], W.flat[i], clo)
        out.flat[i] = round(clo, 1)
    return out
