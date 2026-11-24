"""
Porting di run_gradsbio() — presente identica in plot_images_wkclimate_
mol01gfs.r, plot_images_daymax.r, plot_images_t_dayice.r, plot_images_day.r,
plot_images_onlybio.r.

*** DIPENDENZA ESTERNA NON PORTABILE ***
GrADS (Grid Analysis and Display System) è un binario esterno per il
rendering delle mappe da NetCDF — non un pacchetto Python, non
sostituibile con una libreria pip. Questo modulo chiama `grads` via
subprocess esattamente come faceva R via `system()`: la disponibilità di
GrADS sul PATH resta un prerequisito operativo identico a prima (il
sorgente originale lo dichiarava già nell'intestazione di
salute_procedura.R: "Need ImageMagick and cdo installed").

*** biometeo_fig_class.gs / biometeo_fig_float.gs: RECUPERATI ***
Questi due script — quelli che fanno il rendering vero e proprio delle
mappe — erano assenti nel primo caricamento ma sono risultati presenti
in codice.zip (mancati nella prima ricerca perché filtrata sui soli
script .sh/.R/.py). Sono copiati in `salutepy/gs_scripts/` e NON
riscritti: restano script GrADS (linguaggio proprietario, nessun
equivalente Python) invocati via subprocess esattamente come nel
sorgente R.

*** ASSET ANCORA MANCANTI (verificati puntualmente) ***
biometeo_fig_class.gs, per le variabili "classificate" (floatflag="cla"),
richiama a sua volta:
  - legend/palette/palette_<var>.gs  — MANCANTI per: t_gt34, utcicla,
    utciclasha, windcla (presenti invece per rkcla_int/nint,
    rkshacla_int/nint, t_dayice)
  - legend/code/cbarn_<var>.gs        — tutti presenti (nessuna mancanza)
  - legend/strings/<var>_string.gs   — l'intera cartella legend/strings/
    è assente: mancano TUTTI gli script string per le 9 variabili "cla"
    usate (rkcla_int/nint, rkshacla_int/nint, t_gt34, t_dayice, utcicla,
    utciclasha, windcla)
Per le variabili "float" (biometeo_fig_float.gs) non serve alcuna
palette (lo script non la richiama): quelle presenti in legend/palette/
per variabili float (lprod, lprodsha, airtiwtl, airtwtl, thi, frostime)
non sono usate da questo flusso e sono probabilmente asset di un'altra
pipeline di visualizzazione.

Senza questi asset mancanti, GrADS molto probabilmente si fermerà con un
errore (script .gs non trovato) quando tenta di renderizzare le
variabili "cla" elencate sopra. Le variabili "float" e le "cla" con
palette disponibile (rkcla_int/nint, rkshacla_int/nint, t_dayice)
potrebbero invece renderizzare la mappa base correttamente, con la sola
mancanza delle etichette testuali di classe (da legend/strings/, sempre
assente per tutte le "cla").
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .shell import run_cmd


class GradsNotAvailable(RuntimeError):
    pass


def find_grads_binary() -> str:
    """Equivalente di `system("which grads", intern=TRUE)`."""
    path = shutil.which("grads")
    if path is None:
        raise GradsNotAvailable(
            "Binario 'grads' non trovato sul PATH. GrADS è una dipendenza esterna "
            "(non installabile via pip) richiesta per il rendering delle mappe, "
            "esattamente come nel sorgente R originale."
        )
    return path


def run_gradsbio(file: str | Path, var: str, id_: int, codegs: str | Path, stop_on_error: bool = False) -> None:
    """
    Porting di run_gradsbio(file, var, id, codegs) — identica in tutti
    e cinque gli script plot_images_*.r originali.

    Parameters
    ----------
    file : percorso del file NetCDF/GRIB2 di input
    var : nome della variabile da plottare
    id_ : indice (giorno o ora) passato allo script .gs
    codegs : percorso dello script GrADS (.gs) da eseguire
    """
    grads_bin = find_grads_binary()
    cmd = f'{grads_bin} -blc "run {codegs} {file} {var} {id_}"'
    run_cmd(cmd, stop_on_error=stop_on_error)


def codegs_for(floatflag: str, gs_dir: str | Path = "") -> str:
    """
    Equivalente di:
        codegrads = ifelse(floatflag=="cla", "biometeo_fig_class.gs", "biometeo_fig_float.gs")
    """
    name = "biometeo_fig_class.gs" if floatflag == "cla" else "biometeo_fig_float.gs"
    return f"{gs_dir}{name}" if gs_dir else name


# Asset .gs verificati puntualmente come MANCANTI nei materiali caricati
# (vedi docstring del modulo). Aggiornare se/quando vengono recuperati.
KNOWN_MISSING_PALETTES = {"t_gt34", "utcicla", "utciclasha", "windcla"}
KNOWN_MISSING_STRINGS_DIR = True  # l'intera cartella legend/strings/ è assente


def check_grads_assets(var: str, floatflag: str, legend_dir: Path) -> list[str]:
    """
    Verifica la presenza degli asset .gs richiesti da biometeo_fig_class.gs
    per una data variabile "cla", PRIMA di lanciare GrADS — per fallire
    con un messaggio chiaro invece di scoprire l'errore a metà rendering.
    Per le variabili "float" non c'è nulla da verificare (biometeo_fig_
    float.gs non richiama palette/string).

    Returns
    -------
    Lista di messaggi di warning per gli asset mancanti (vuota se tutto ok).
    """
    if floatflag != "cla":
        return []

    problems = []
    palette_path = legend_dir / "palette" / f"palette_{var}.gs"
    if not palette_path.exists():
        problems.append(f"palette mancante: {palette_path}")

    string_path = legend_dir / "strings" / f"{var}_string.gs"
    if not string_path.exists():
        problems.append(f"string mancante: {string_path}")

    return problems
