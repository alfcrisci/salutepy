"""
Porting di funzioni ausiliarie rimanenti in salute_aux.r, usate
direttamente in meteosalute_op_dir_mol01.r (non da HeatStress/rBiometeo).
"""

from __future__ import annotations

import numpy as np


def windchill_class_wk(x):
    """
    Porting di salute_aux.r::windchill_class_wk(x).
    Classificazione windchill (scala lavorativa):
    1=ASSENTE (x>10), 2=DEBOLE (5<x<=10), 3=MODERATO (0<x<=5),
    4=INTENSO (-10<x<=0), 5=MOLTO INTENSO (x<=-10).
    """
    x = np.asarray(x, dtype=float)
    return np.where(x > 10, 1, np.where(x > 5, 2, np.where(x > 0, 3, np.where(x > -10.0, 4, 5)))).astype(float)


def metabolism(t):
    """Porting di salute_aux.r::metabolism(t). Metabolismo stimato da temperatura (W/m2 circa, prima della /1.8)."""
    t = np.asarray(t, dtype=float)
    return -3.0909 * t + 203.64


def reducewind(x, ref=10, fin=2):
    """
    Porting di salute_aux.r::reducewind(x, ref=10, fin=2).
    Riduce la velocita' del vento dall'altezza di riferimento (default 10m)
    all'altezza finale (default 2m) secondo il profilo logaritmico.
    """
    x = np.asarray(x, dtype=float)
    return x * 1 / (np.log(ref / 0.01) / np.log(fin / 0.01))


def oralegale_l(d) -> int:
    """
    Porting di salute_aux.r::oralegale_l(x).
    Ritorna 1 se `d` cade nel periodo di ora legale (CEST, 31 marzo -
    27 ottobre dell'anno corrente), altrimenti 0.

    Nota: come nel sorgente R, l'anno di riferimento per il periodo CEST
    e' quello CORRENTE (Sys.Date()), non l'anno di `d` — replicato
    fedelmente anche se puo' dare risultati sorprendenti se `d` e'
    di un anno diverso da oggi.
    """
    import datetime as _dt

    if isinstance(d, _dt.datetime):
        d = d.date()

    anno = _dt.date.today().year
    cest_start = _dt.date(anno, 3, 31)
    cest_end = _dt.date(anno, 10, 27)

    return 1 if cest_start <= d <= cest_end else 0
