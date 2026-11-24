"""
Download di ICON-EU (DWD) e conversione nel formato usato dalla pipeline
meteosalute (stesse 9 variabili di grib_reader.NAMES_METEOS), per
alimentare process_timestep_from_grib() senza passare da MOLOCH.

*** NON TESTATO CONTRO IL SERVER DWD REALE ***
Questo sandbox non ha accesso di rete a opendata.dwd.de (whitelist di
rete ristretta a pypi/github/npm ecc.). Il modulo e' scritto sulla base
delle convenzioni DWD verificate via ricerca web (URL, nomi parametro,
semantica "mean from run start" dei campi di radiazione), ma va validato
al primo uso reale: URL esatti, finestra di disponibilita' degli step,
e la formula di deaccumulazione NON sono stati testati su un download
vero.

*** MAPPING VARIABILI (best-effort, da validare) ***
| names_meteos | parametro DWD ICON-EU | note                              |
|--------------|------------------------|-----------------------------------|
| T_fhg        | T_2M                   | Kelvin -> Celsius (-273.15)        |
| TD_fhg       | TD_2M                  | Kelvin -> Celsius (-273.15)        |
| u_fhg        | U_10M                  | m/s, nessuna conversione            |
| v_fhg        | V_10M                  | m/s, nessuna conversione            |
| gust_sfc     | VMAX_10M               | m/s, nessuna conversione            |
| pmsl         | PMSL                   | Pa (coerente con /100 in meteosalute.py) |
| N_atm        | CLCT                   | % (0-100) — MOLOCH potrebbe usare frazione 0-1: VERIFICARE |
| dswrf_sfc    | ASWDIR_S + ASWDIFD_S   | "mean from run start" -> deaccumulazione, poi somma diretta+diffusa |
| lapr         | NESSUN EQUIVALENTE     | non disponibile in ICON-EU single-level; riempito a NaN |

Radiazione: ASWDIR_S/ASWDIFD_S in ICON-EU sono medie cumulate dall'inizio
del run (non accumuli semplici) — verificato via documentazione DWD
(gribstream.com/models/iconeu, dwd-geoportal.de). La deaccumulazione a
media-sull'intervallo richiede la coppia di timestep consecutivi:

    mean_interval(t) = (mean_from_start(t)*t - mean_from_start(t-1)*(t-1)) / (t-(t-1))

implementata in deaccumulate_mean_from_start().
"""

from __future__ import annotations

import bz2
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio

from .grib_reader import NAMES_METEOS, MolochGribData
from .shell import run_cmd

ICON_EU_BASE = "https://opendata.dwd.de/weather/nwp/icon-eu/grib"

# parametro DWD (nome file/URL, maiuscolo) per ciascuna variabile names_meteos
# che ha un equivalente diretto in ICON-EU single-level
DWD_PARAM = {
    "T_fhg": "T_2M",
    "TD_fhg": "TD_2M",
    "u_fhg": "U_10M",
    "v_fhg": "V_10M",
    "gust_sfc": "VMAX_10M",
    "pmsl": "PMSL",
    "N_atm": "CLCT",
}
# variabili di radiazione: gestite a parte (somma diretta+diffusa, deaccumulazione)
DWD_RADIATION_PARAMS = ("ASWDIR_S", "ASWDIFD_S")

# run disponibili (00/06/12/18 UTC) e finestra step nota (verificata via
# documentazione DWD/GribStream): orario fino a 78h, poi trioario fino a 120h
# per i run principali 00/06/12/18. Da confermare sul server reale.
ICON_EU_RUNS = (0, 6, 12, 18)


@dataclass
class IconDownloadConfig:
    work_dir: Path  # download/decompressione/remap intermedi
    target_grid_file: Path  # griglia target per cdo remapbil (es. mol01gfs.grb2.topo_landmask.interp_grid)
    base_url: str = ICON_EU_BASE
    keep_intermediate: bool = False


def build_url(run_datetime: datetime, param: str, step_h: int, base_url: str = ICON_EU_BASE) -> str:
    """
    Costruisce l'URL DWD per un parametro/step, secondo la convenzione
    verificata:
        <base>/<HH>/<param_lower>/icon-eu_europe_regular-lat-lon_single-level_<YYYYMMDDHH>_<fff>_<PARAM>.grib2.bz2
    """
    hh = f"{run_datetime.hour:02d}"
    run_str = run_datetime.strftime("%Y%m%d%H")
    fff = f"{step_h:03d}"
    return f"{base_url}/{hh}/{param.lower()}/icon-eu_europe_regular-lat-lon_single-level_{run_str}_{fff}_{param}.grib2.bz2"


def download_param_step(
    run_datetime: datetime, param: str, step_h: int, work_dir: Path, base_url: str = ICON_EU_BASE
) -> Path:
    """
    Scarica e decomprime un singolo file parametro/step. Ritorna il
    percorso del .grib2 decompresso.
    """
    url = build_url(run_datetime, param, step_h, base_url)
    bz2_path = work_dir / Path(url).name
    grib_path = bz2_path.with_suffix("")  # rimuove ".bz2"

    result = run_cmd(f"curl -sf -o {bz2_path} {url}", stop_on_error=False)
    if result.returncode != 0 or not bz2_path.exists() or bz2_path.stat().st_size == 0:
        raise RuntimeError(f"Download fallito per {param} step {step_h}h: {url}")

    with bz2.open(bz2_path, "rb") as f_in, open(grib_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    bz2_path.unlink()

    return grib_path


def remap_to_grid(grib_path: Path, target_grid_file: Path, out_path: Path) -> Path:
    """Equivalente di: cdo remapbil,<target_grid_file> <in> <out>."""
    run_cmd(f"cdo remapbil,{target_grid_file} {grib_path} {out_path}")
    return out_path


def read_single_band(grib_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Legge un file GRIB2 a singola banda (dopo remap) e ritorna
    (values_1d, lon_unique, lat_unique, dummy) nello stesso ordine
    riga-maggiore usato in tutto il resto del package (vedi grib_reader.py).
    """
    with rasterio.open(grib_path) as ds:
        band = ds.read(1).astype(float)  # shape (height, width)
        height, width = ds.height, ds.width
        col_ids = np.arange(width)
        row_ids = np.arange(height)
        lon = np.asarray(rasterio.transform.xy(ds.transform, [0] * width, col_ids)[0], dtype=float)
        lat = np.asarray(rasterio.transform.xy(ds.transform, row_ids, [0] * height)[1], dtype=float)
        values = band.ravel()  # row-major: lat lento, lon veloce — coerente con grib_reader.py
    return values, lon, lat, None


def deaccumulate_mean_from_start(mean_now: np.ndarray, mean_prev: np.ndarray, t_now_h: float, t_prev_h: float) -> np.ndarray:
    """
    Deaccumula un campo "mean from run start" (es. ASWDIR_S/ASWDIFD_S in
    ICON-EU) nella media sull'intervallo [t_prev, t_now].

        mean_interval = (mean_now*t_now - mean_prev*t_prev) / (t_now - t_prev)

    Per il primo step disponibile (t_prev=0, mean_prev assente/zero),
    passare mean_prev=zeros_like(mean_now) e t_prev_h=0: la media
    sull'intervallo coincide con la media dall'inizio.
    """
    if t_now_h == t_prev_h:
        raise ValueError("t_now_h e t_prev_h non possono essere uguali (divisione per zero).")
    return (mean_now * t_now_h - mean_prev * t_prev_h) / (t_now_h - t_prev_h)


def fetch_icon_eu_timestep(
    run_datetime: datetime,
    step_h: int,
    cfg: IconDownloadConfig,
    prev_step_h: int | None = None,
) -> MolochGribData:
    """
    Scarica, remappa e assembla un singolo timestep ICON-EU nel formato
    MolochGribData (stesse 9 variabili di NAMES_METEOS), pronto per
    process_timestep_from_grib().

    Parameters
    ----------
    run_datetime : datetime del run ICON-EU (deve avere hour in ICON_EU_RUNS)
    step_h : ora di forecast da scaricare (0, 1, 2, ... secondo la
        disponibilita' del run)
    cfg : IconDownloadConfig
    prev_step_h : ora di forecast precedente, necessaria per deaccumulare
        la radiazione (ASWDIR_S/ASWDIFD_S). Se None e step_h==0, si
        assume che sia il primissimo step del run (media sull'intervallo
        = media dall'inizio, t_prev=0). Se None e step_h>0, viene
        assunto prev_step_h = step_h - 1.
    """
    if run_datetime.hour not in ICON_EU_RUNS:
        raise ValueError(f"Run non valido: {run_datetime.hour}h. Run ICON-EU disponibili: {ICON_EU_RUNS}")

    cfg.work_dir.mkdir(parents=True, exist_ok=True)

    if prev_step_h is None:
        prev_step_h = max(step_h - 1, 0)

    variables: dict[str, np.ndarray] = {}
    lon = lat = None

    # --- variabili con equivalente diretto ---
    for label, param in DWD_PARAM.items():
        raw = download_param_step(run_datetime, param, step_h, cfg.work_dir, cfg.base_url)
        remapped = cfg.work_dir / f"{param}_{step_h:03d}_remap.grib2"
        remap_to_grid(raw, cfg.target_grid_file, remapped)

        values, lon_i, lat_i, _ = read_single_band(remapped)
        lon, lat = lon_i, lat_i

        if label in ("T_fhg", "TD_fhg"):
            values = values - 273.15  # Kelvin -> Celsius

        variables[label] = values

        if not cfg.keep_intermediate:
            raw.unlink(missing_ok=True)
            remapped.unlink(missing_ok=True)

    # --- radiazione: deaccumulazione + somma diretta/diffusa ---
    rad_now = {}
    rad_prev = {}
    for param in DWD_RADIATION_PARAMS:
        raw_now = download_param_step(run_datetime, param, step_h, cfg.work_dir, cfg.base_url)
        remapped_now = cfg.work_dir / f"{param}_{step_h:03d}_remap.grib2"
        remap_to_grid(raw_now, cfg.target_grid_file, remapped_now)
        values_now, _, _, _ = read_single_band(remapped_now)
        rad_now[param] = values_now

        if step_h == 0:
            rad_prev[param] = np.zeros_like(values_now)
        else:
            raw_prev = download_param_step(run_datetime, param, prev_step_h, cfg.work_dir, cfg.base_url)
            remapped_prev = cfg.work_dir / f"{param}_{prev_step_h:03d}_remap.grib2"
            remap_to_grid(raw_prev, cfg.target_grid_file, remapped_prev)
            values_prev, _, _, _ = read_single_band(remapped_prev)
            rad_prev[param] = values_prev
            if not cfg.keep_intermediate:
                raw_prev.unlink(missing_ok=True)
                remapped_prev.unlink(missing_ok=True)

        if not cfg.keep_intermediate:
            raw_now.unlink(missing_ok=True)
            remapped_now.unlink(missing_ok=True)

    t_prev = 0.0 if step_h == 0 else float(prev_step_h)
    direct = deaccumulate_mean_from_start(rad_now["ASWDIR_S"], rad_prev["ASWDIR_S"], float(step_h), t_prev)
    diffuse = deaccumulate_mean_from_start(rad_now["ASWDIFD_S"], rad_prev["ASWDIFD_S"], float(step_h), t_prev)
    variables["dswrf_sfc"] = direct + diffuse

    # --- lapr: nessun equivalente ICON-EU, riempito a NaN ---
    variables["lapr"] = np.full_like(variables["T_fhg"], np.nan)

    n_lon, n_lat = len(lon), len(lat)
    lonvect = np.tile(lon, n_lat)
    latvect = np.repeat(lat, n_lon)

    return MolochGribData(
        variables={k: variables[k] for k in NAMES_METEOS},
        lon=lon,
        lat=lat,
        lonvect=lonvect,
        latvect=latvect,
    )
