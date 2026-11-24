"""
Porting di wbgt_aux.r (gia' presente nel codice originale della pipeline,
non da HeatStress/rBiometeo): lost_productivity e le formule di Work Time
Loss di Kjellstrom et al. (2018).

Riferimento: Kjellstrom T., Freyberg C., Lemke B., Otto M., Briggs D.
"Estimating population heat exposure and impacts on working people in
conjunction with climate change." Int J Biometeorol (2018) 62:291-306.
DOI 10.1007/s00484-017-1407-0
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf


def lost_productivity(wbgt, tresh: float = 33.5):
    """
    Porting di wbgt_aux.r::lost_productivity(wbgt, tresh=33.5).
    Perdita di produttivita' (%) in funzione del WBGT, curva a S (error
    function) centrata su tresh con deviazione 3.75.
    """
    wbgt = np.asarray(wbgt, dtype=float)
    return 0.5 * (1 + erf((wbgt - tresh) / (3.75 * np.sqrt(2)))) * 100


def mean_wbgtTI_WTL(wbgtsun):
    """Temperature-Induced Work Time Loss (% tempo), livello medio. Porting di wbgt_aux.r."""
    w = np.asarray(wbgtsun, dtype=float)
    return (29.147402 - 2.584945 * w + 0.072177 * w ** 2 + 0.000434 * w ** 3) - 8.0


def mean_wbgtWTL(wbgtsun):
    """Total Work Time Loss (ore), livello medio. Porting di wbgt_aux.r."""
    w = np.asarray(wbgtsun, dtype=float)
    return 29.147402 - 2.584945 * w + 0.072177 * w ** 2 + 0.000434 * w ** 3


def mean_airTI_WTL(tair):
    """Temperature-Induced Work Time Loss (% tempo) da temperatura aria, livello medio. Porting di wbgt_aux.r."""
    t = np.asarray(tair, dtype=float)
    return (42.385043 - 4.134546 * t + 0.130371 * t ** 2 - 0.000531 * t ** 3) - 7.1


def mean_airTWTL(tair):
    """Total Work Time Loss (ore) da temperatura aria, livello medio. Porting di wbgt_aux.r."""
    t = np.asarray(tair, dtype=float)
    return 42.385043 - 4.134546 * t + 0.130371 * t ** 2 - 0.000531 * t ** 3
