"""
Porting di list_bio_wkclimate.r, list_bio_daymax.r, list_bio_t_dayice.r,
list_bio_day.r, list_bio_only.r: liste delle variabili da plottare,
con nome variabile, flag classe/float, nome file di output e testo
descrittivo — stesse stringhe esatte del sorgente R (usate come titoli
sulle mappe finali, non vanno modificate senza motivo).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VarSpec:
    var: str
    floatflag: str  # "cla" o "float"
    fileout: str
    textvar: str


def list_bio_wkclimate() -> list[VarSpec]:
    """Porting di list_bio_wkclimate.r."""
    names = ["rkcla_int", "rkcla_nint", "rkshacla_int", "rkshacla_nint"]
    outs = ["rkcla_int_ita", "rkcla_nint_ita", "rkshacla_int_ita", "rkshacla_nint_ita"]
    texts = [
        "Rischio caldo al sole con attivita' fisica intensa",
        "Rischio caldo al sole con attivita' fisica moderata",
        "Rischio caldo in ombra con attivita' fisica intensa",
        "Rischio caldo in ombra con attivita' fisica moderata",
    ]
    return [VarSpec(n, "cla", o, t) for n, o, t in zip(names, outs, texts)]


def list_bio_daymax() -> VarSpec:
    """Porting di list_bio_daymax.r (una sola variabile, 5 file di output per i 5 giorni)."""
    return VarSpec("t_gt34", "cla", "TGG35_ita", "Aree con possibili temperature massime superiori a 35 C")


def list_bio_t_dayice() -> VarSpec:
    """Porting di list_bio_t_dayice.r (una sola variabile, 5 file di output per i 5 giorni)."""
    return VarSpec("t_dayice", "cla", "t_dayice_ita", "Aree con possibili con temperature minime inferiori a 0 C")


def list_bio_day() -> list[VarSpec]:
    """Porting di list_bio_day.r (max/min/escursione termica giornaliera)."""
    names = ["ta2m", "ta2m", "ta2m"]
    outs = ["max_t2m_ita", "min_t2m_ita", "exc_t2m_ita"]
    texts = ["Temperatura massima", "Temperatura minima ", "Escursione termica"]
    return [VarSpec(n, "float", o, t) for n, o, t in zip(names, outs, texts)]


def list_bio_only() -> list[VarSpec]:
    """Porting di list_bio_only.r (19 variabili orarie, esclude il rischio caldo)."""
    names = [
        "utcicla", "utciclasha", "clomin", "lprod", "lprodsha", "wbgttiwtlsha",
        "wbgttwtlsha", "wbgttiwtlsun", "wbgttwtlsun", "airtiwtl", "airtwtl",
        "utci", "windchill", "windcla", "wbgtsha", "wbgtsun", "tglob", "thi", "frostime",
    ]
    outs = [
        "utcicla_ita", "utciclasha_ita", "clomin_ita", "lprod_ita", "lprodsha_ita",
        "wbgttiwtlsha_ita", "wbgttwtlsha_ita", "wbgttiwtlsun_ita", "wbgttwtlsun_ita",
        "airtiwtl_ita", "airtwtl_ita", "utci_ita", "windchill_ita", "windchill_cla_ita",
        "wbgtsha_ita", "wbgtsun_ita", "tglob_ita", "thi_ita", "frostime_ita",
    ]
    flags = [
        "cla", "cla", "float", "float", "float", "float", "float", "float", "float",
        "float", "float", "float", "float", "cla", "float", "float", "float", "float", "float",
    ]
    texts = [
        "Disagio termico al sole (Classi UTCI)",
        "Disagio termico in ombra (Classi UTCI)",
        "Resistenza termica minima outdoor (clo)",
        "Potenziale perdita di produttivita' al sole (%)",
        "Potenziale perdita di produttivita' in ombra (%)",
        "Work Time Loss wbgt shade (%)",
        "Total Work Time Loss wbgt shade (hours)",
        "Work Time Loss wbgtsun (%)",
        "Total Work Time Loss wbgt sun (hours)",
        "Work Time Loss Tair (%)",
        "Total Work Time Loss Tair(hours)",
        "UTCI (Universal Thermal Climate Index) ",
        "Windchill  index",
        "Classi di rischio congelamento",
        "WBGT ( Wet Bulb Globe Temperature) in ombra",
        "WBGT ( Wet Bulb Globe Temperature)  al sole",
        "Temperatura Globometrica",
        "THI Temperature humidity index",
        "Tempo di congelamento (min)",
    ]
    return [VarSpec(n, f, o, t) for n, f, o, t in zip(names, flags, outs, texts)]
