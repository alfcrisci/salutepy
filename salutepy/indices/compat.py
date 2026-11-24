"""
Layer di compatibilita' con i nomi R esatti usati in
meteosalute_op_dir_mol01.r. Ogni funzione qui e' un semplice alias verso
l'implementazione reale nei moduli specifici (wbgt.py, thermal_indices.py,
comfort.py, productivity.py, utci_wrapper.py, psychrometrics.py), cosi'
che il porting dello script R sia un mapping diretto nome-per-nome.

Esempio d'uso — dal codice R:

    wbgt.outdoors <- wbgt.Liljegren(T_fhg, TD_fhg, wind2m, dswrf_sfc, dates, lon, lat)
    thi_index <- THI(T_fhg, RH)
    tglob <- wbgt.outdoors$Tg
    ...

diventa in Python:

    from salutepy.indices import compat as R

    wbgt_outdoors = R.wbgt_Liljegren(T_fhg, TD_fhg, wind2m, dswrf_sfc, dates, lon, lat)
    thi_index = R.THI(T_fhg, RH)
    tglob = wbgt_outdoors.Tg
    ...
"""

from __future__ import annotations

from .comfort import clomax_7730 as clomax_7730
from .comfort import clomin_7730 as clomin_7730
from .comfort import pmv_iso7730 as PMV_ISO7730
from .productivity import lost_productivity as lost_productivity
from .productivity import mean_airTI_WTL as mean_airTI_WTL
from .productivity import mean_airTWTL as mean_airTWTL
from .productivity import mean_wbgtTI_WTL as mean_wbgtTI_WTL
from .productivity import mean_wbgtWTL as mean_wbgtWTL
from .psychrometrics import deficitsat as deficitsat
from .psychrometrics import dewp2hurs as dewp2hurs
from .psychrometrics import pvap as pvap
from .thermal_indices import frostime as frostime
from .thermal_indices import heat_risk_level as heat_risk_level
from .thermal_indices import mrt_globe as mrt_globe
from .thermal_indices import steadman_class as steadman_class
from .thermal_indices import steadman_indoor as steadman_indoor
from .thermal_indices import steadman_outdoor_shade as steadman_outdoor_shade
from .thermal_indices import steadman_outdoor_sun as steadman_outdoor_sun
from .thermal_indices import thi as THI
from .thermal_indices import windchill as windchill
from .utci_wrapper import utci as utci
from .utci_wrapper import utci_class as utci_class
from .wbgt import wbgt_liljegren as wbgt_Liljegren  # nome R: wbgt.Liljegren
