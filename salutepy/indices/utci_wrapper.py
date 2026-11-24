"""
UTCI: porting di rBiometeo::utci/utci_class.

La funzione UTCI in biometeo.js e' il polinomio di regressione ufficiale
a 6° ordine di Brode et al. (2012) — lo stesso standard implementato in
pythermalcomfort.models.utci (stessi limiti di validita' 0.5-17 m/s per
il vento, stessa clamp). Per questo motivo, invece di ritrascrivere a
mano circa 210 coefficienti del polinomio (alto rischio di errore di
trascrizione in entrambe le direzioni), questo modulo fa da wrapper verso
pythermalcomfort.

*** DA VERIFICARE PRIMA DELL'USO OPERATIVO ***
Prima di sostituire l'output R in produzione, eseguire un confronto
numerico diretto: calcolare UTCI in R (rBiometeo::utci) e in Python
(questo modulo) sugli stessi input di test e verificare che la differenza
sia entro un'epsilon trascurabile (es. <0.05 degC). Non e' stato ancora
fatto in questa sessione per mancanza di un ambiente R con rBiometeo
installato per il confronto diretto.

utci_class() invece usa le soglie di classificazione CUSTOM di rBiometeo
(scala 1-10), diverse dalle "stress category" di default di
pythermalcomfort — quelle sono riportate qui esplicitamente.
"""

from __future__ import annotations

import numpy as np

try:
    from pythermalcomfort.models import utci as _pythermalcomfort_utci
except ImportError:  # pragma: no cover
    _pythermalcomfort_utci = None


def utci(tdb, tr, v, rh):
    """
    Porting/wrapper di rBiometeo::utci (biometeo.js UTCI), via
    pythermalcomfort.models.utci. Vedi nota di verifica nel docstring
    del modulo prima dell'uso in produzione.

    Parameters
    ----------
    tdb : temperatura aria (degC)
    tr : temperatura media radiante (degC)
    v : velocita' vento a 10m (m/s) — clampata a [0.5,17] come nel JS originale
    rh : umidita' relativa (%)
    """
    if _pythermalcomfort_utci is None:
        raise ImportError("pythermalcomfort non installato: pip install pythermalcomfort")

    result = _pythermalcomfort_utci(tdb=tdb, tr=tr, v=v, rh=rh, limit_inputs=False)
    return np.asarray(result.utci, dtype=float)


def utci_class(tdb, tr, v, rh):
    """
    Porting di rBiometeo::utci_class (biometeo.js utci_class).
    Classe di stress termico UTCI su scala CUSTOM 1-10 (non le stress
    category standard di pythermalcomfort):

        classe 10 : utci >  46
        classe  9 : 38  < utci <= 46
        classe  8 : 32  < utci <= 38
        classe  7 : 26  < utci <= 32
        classe  6 : 9   < utci <= 26
        classe  5 : 0   < utci <= 9
        classe  4 : -13 < utci <= 0
        classe  3 : -27 < utci <= -13
        classe  2 : -40 < utci <= -27
        classe  1 : utci <= -40
    """
    u = utci(tdb, tr, v, rh)

    bins = [-np.inf, -40, -27, -13, 0, 9, 26, 32, 38, 46, np.inf]
    # digitize assegna in ordine crescente: classe = indice del bin + 1 (poiche' i bin sono in ordine 1..10)
    classes = np.digitize(u, bins[1:-1], right=True) + 1
    classes = classes.astype(float)
    classes[np.isnan(u)] = np.nan
    return classes
