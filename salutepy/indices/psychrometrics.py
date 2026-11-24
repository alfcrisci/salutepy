"""
Funzioni psicrometriche di base.

ATTENZIONE: il codice R originale (HeatStress + rBiometeo) usa TRE formule
diverse di pressione di vapore saturo per scopi diversi. Sono tenute
separate qui perché non sono intercambiabili: usare quella sbagliata al
posto giusto altera silenziosamente il risultato di deficitsat/pvap/UTCI.

1. esat_buck_hpa(Tk)      -> HeatStress::esat.R (Buck 1981), usata SOLO
                              dentro il modello WBGT di Liljegren
                              (fTnwb, emis_atm).
2. es_goff_hpa(t_celsius)  -> rBiometeo JS es(t) (Goff-Gratch-like),
                              usata da UTCI e da pvap().
3. pvs_iapws_hpa(t_celsius)-> rBiometeo JS PVS(t) (IAPWS-IF97, via
                              pvsIce/pvsWater), usata SOLO da deficitsat().
"""

from __future__ import annotations

import numpy as np


def esat_buck_hpa(Tk):
    """
    Porting di HeatStress::esat.R.
    Pressione di vapore saturo su acqua liquida (hPa), Buck (1981),
    approssimazione di Wexler (1976), con correzione per aria umida.

    Parameters
    ----------
    Tk : temperatura in Kelvin
    """
    Tk = np.asarray(Tk, dtype=float)
    e = 6.1121 * np.exp(17.502 * (Tk - 273.15) / (Tk - 32.18))
    return 1.004 * e


def emis_atm(Tk, RH_fraction):
    """
    Porting di HeatStress::emis_atm.R. Emissivita' atmosferica (Oke, 2nd ed., p.373).

    Parameters
    ----------
    Tk : temperatura in Kelvin
    RH_fraction : umidita' relativa in FRAZIONE (0-1), non percentuale
    """
    e = RH_fraction * esat_buck_hpa(Tk)
    return 0.575 * e ** 0.143


def dewp2hurs(tas, dewp):
    """
    Porting di HeatStress::dewp2hurs.R (Dosseger et al. 1992, formula 99 MCH).
    Umidita' relativa (%) da temperatura dell'aria e punto di rugiada (degC).
    """
    tas = np.asarray(tas, dtype=float)
    dewp = np.asarray(dewp, dtype=float)

    a1, b1 = 17.368, 238.83  # T >= 0
    a2, b2 = 17.856, 245.52  # T < 0

    ice = tas < 0
    hurs = np.where(
        ice,
        100 * np.exp((a2 * dewp) / (b2 + dewp) - (a2 * tas) / (b2 + tas)),
        100 * np.exp((a1 * dewp) / (b1 + dewp) - (a1 * tas) / (b1 + tas)),
    )
    return np.clip(hurs, 0, 100)


def es_goff_hpa(t_celsius):
    """
    Porting di rBiometeo (biometeo.js, funzione es(t)).
    Pressione di saturazione (hPa) da temperatura in gradi Celsius.
    Usata da UTCI e da pvap(). NON e' la stessa formula di esat_buck_hpa
    ne' di pvs_iapws_hpa: i coefficienti g[0..7] sono quelli della
    equazione WMO/Goff basata su log(T).
    """
    t = np.asarray(t_celsius, dtype=float)
    tk = t + 273.15

    g = [
        -2.8365744e3,
        -6.028076559e3,
        1.954263612e1,
        -2.737830188e-2,
        1.6261698e-5,
        7.0229056e-10,
        -1.8680009e-13,
        2.7150305,
    ]

    es_air = g[7] * np.log(tk)
    for i in range(7):
        es_air = es_air + g[i] * tk ** (i - 2)

    return np.exp(es_air) * 0.01


def pvap(t_celsius, rh_percent):
    """Porting di rBiometeo::pvap.R (JS pvap). Pressione di vapore (hPa) = es_goff_hpa(t) * rh/100."""
    return es_goff_hpa(t_celsius) * (np.asarray(rh_percent, dtype=float) / 100.0)


def _pvs_ice_kpa(Tk):
    """Porting della sotto-funzione JS pvsIce(T) (T in Kelvin), risultato in Pa."""
    Tk = np.asarray(Tk, dtype=float)
    k0, k1, k2, k3, k4, k5 = (
        -5.8666426e3,
        2.232870244e1,
        1.39387003e-2,
        -3.4262402e-5,
        2.7040955e-8,
        6.7063522e-1,
    )
    lnP = k0 / Tk + k1 + (k2 + (k3 + (k4 * Tk)) * Tk) * Tk + k5 * np.log(Tk)
    return np.exp(lnP)


def _pvs_water_kpa(Tk):
    """Porting della sotto-funzione JS pvsWater(T) (T in Kelvin), risultato in Pa (IAPWS-IF97 eq. 30)."""
    Tk = np.asarray(Tk, dtype=float)
    n1, n2, n3, n4, n5 = 0.11670521452767e4, -0.72421316703206e6, -0.17073846940092e2, 0.12020824702470e5, -0.32325550322333e7
    n6, n7, n8, n9, n10 = 0.14915108613530e2, -0.48232657361591e4, 0.40511340542057e6, -0.23855557567849, 0.65017534844798e3

    th = Tk + n9 / (Tk - n10)
    A = (th + n1) * th + n2
    B = (n3 * th + n4) * th + n5
    C = (n6 * th + n7) * th + n8

    p = 2 * C / (-B + np.sqrt(B * B - 4 * A * C))
    p = p * p
    p = p * p
    return p * 1e6


def pvs_iapws_hpa(t_celsius):
    """
    Porting di rBiometeo (biometeo.js, funzione PVS(t)).
    Pressione di vapore saturo (hPa) IAPWS-IF97, ramo ghiaccio/acqua a
    seconda del segno della temperatura. Usata SOLO da deficitsat().

    Fuori dal range di validita' (-100..+405 degC circa) ritorna NaN,
    corrispondente al -9999/9999 sentinella del codice R/JS originale.
    """
    t = np.asarray(t_celsius, dtype=float)
    Tk = t + 273.16
    min_t, max_t = 173.0, 678.0

    out = np.full_like(Tk, np.nan, dtype=float)
    valid = (Tk >= min_t) & (Tk <= max_t)

    ice = valid & (Tk < 273.15)
    water = valid & (Tk >= 273.15)

    out[ice] = _pvs_ice_kpa(Tk[ice]) / 100.0
    out[water] = _pvs_water_kpa(Tk[water]) / 100.0

    return np.round(out, 2)


def deficitsat(t_celsius, rh_percent):
    """
    Porting di rBiometeo::deficitsat.R (JS deficitsat). Deficit di
    saturazione in hPa: pressione di saturazione (IAPWS) meno pressione
    di vapore attuale.
    """
    t = np.asarray(t_celsius, dtype=float)
    rh = np.asarray(rh_percent, dtype=float)
    pws = pvs_iapws_hpa(t)
    pae = rh / 100.0 * pws
    return pws - pae
