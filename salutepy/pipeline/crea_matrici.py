"""
Porting di crea_matrici.r (versione corretta: fixato il bug per cui
windchill_df veniva calcolato ma mai scritto su CSV — vedi commento
inline più sotto).

Mapping rispetto al file R:
- righe 1-45   (setup, date, time_models)         -> _build_time_labels(), _build_date_ranges()
- righe 47-86  (pulizia csv, controllo archivio)  -> _fill_missing_archive_days()
- righe 88-110 (matrice tmax)                     -> _extract_tmax_matrix()
- righe 112-165 (estrazioni acclim/wbgt/utci/...)  -> _extract_forecast_matrices()
- righe 166-258 (scrittura CSV, poste risk, date) -> run()
"""

from __future__ import annotations

import shutil
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..indices import compat as R
from ..indices.misc import oralegale_l
from .extraction import PIXELNUMBER_MOL01, estrai_acclimdays, estrai_param


@dataclass
class CreaMatriciConfig:
    work_dir: Path  # dir_products (contiene mol01_daymax_12_* e i file 12_mol01gfs.*.grb2)
    dir_archive: Path  # archivio "day_*.nc"
    dir_out: Path  # output CSV (contiene anche geo/XYZ_mol02.csv)
    dir_archive_current_forecast: Path  # archiviazione con data nel nome
    poste_risk_dir: Path  # "/home/salute/data/POSTE_RISK/"
    landmask_path: Path  # landmask_data.rds


def _build_time_labels(model_date: date) -> list[str]:
    """Equivalente di time_models: 72 etichette orarie 'YYYY-MM-DD HH:MM' a partire da model_date+1."""
    start = datetime(model_date.year, model_date.month, model_date.day) + timedelta(days=1)
    return [(start + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M") for i in range(72)]


def _daterange(start: date, end: date) -> list[date]:
    n = (end - start).days
    return [start + timedelta(days=i) for i in range(n + 1)]


def _build_date_ranges(model_date: date) -> dict:
    """Equivalente di pp_dates/ff_dates_prev/ff_dates_max_nam ecc."""
    pp = _daterange(model_date - timedelta(days=9), model_date)
    ff_prev = _daterange(model_date + timedelta(days=1), model_date + timedelta(days=3))
    ff_max = _daterange(model_date + timedelta(days=1), model_date + timedelta(days=5))

    return {
        "pp_dates": [f"day_{d.isoformat()}.nc" for d in pp],
        "pp_dates_nam": [f"day_{d.isoformat()}" for d in pp],
        "ff_dates_prev": [f"day_{d.isoformat()}.nc" for d in ff_prev],
        "ff_dates_max_nam": [f"day_{d.isoformat()}" for d in ff_max],
    }


def _fill_missing_archive_days(dir_archive: Path, pp_dates: list[str]) -> None:
    """
    Equivalente delle righe 61-74 R: se mancano file d'archivio per gli
    ultimi 10 giorni, li rimpiazza copiando l'ULTIMO file disponibile
    (non un'interpolazione: duplica sempre lo stesso file più recente
    su tutti gli slot mancanti, esattamente come in R).
    """
    missing = [f for f in pp_dates if not (dir_archive / f).exists()]
    ready = [f for f in pp_dates if (dir_archive / f).exists()]

    if missing:
        warnings.warn(f"File mancanti in archivio: {missing}")
    if not ready:
        warnings.warn("Nessun file disponibile in archivio per il gap-filling.")
        return

    most_recent = ready[-1]
    if missing:
        for f in missing:
            shutil.copy(dir_archive / most_recent, dir_archive / f)


def _extract_tmax_matrix(work_dir: Path, ff_dates_max_nam: list[str]) -> np.ndarray | None:
    """
    Equivalente righe 98-103 R: legge mol01_daymax_12_*.nc (5 layer
    temporali, uno per giorno) e ritorna una matrice (pixelnumber, 5).
    Ritorna None se il file non esiste (equivalente al blocco
    `if (!is.na(tmaxmol))`).
    """
    candidates = sorted(work_dir.glob("mol01_daymax_12_*"))
    if not candidates:
        return None

    from netCDF4 import Dataset

    with Dataset(candidates[0]) as ds:
        varname = [v for v in ds.variables if v not in ("lon", "lat", "time", "longitude", "latitude", "record")][0]
        var = ds.variables[varname]
        n_lon, n_lat, n_time = var.shape
        raw = np.asarray(var[:])
        cols = [raw[:, :, t].T.ravel() for t in range(n_time)]

    return np.round(np.column_stack(cols), 1)


def run(cfg: CreaMatriciConfig) -> dict[str, Path]:
    """
    Orchestrazione completa, equivalente a tutto crea_matrici.r.
    Ritorna un dict {nome_prodotto: percorso_csv_scritto}.
    """
    # --- localizza il file del modello e la data ---
    mol01_candidates = sorted(cfg.work_dir.glob("*12_mol01gfs*.grb2"))
    if not mol01_candidates:
        raise RuntimeError("Nessun file '*12_mol01gfs*.grb2' trovato in work_dir.")
    mol01_name = mol01_candidates[0].name
    model_date = datetime.strptime(mol01_name[:8], "%Y%m%d").date()

    dates = _build_date_ranges(model_date)
    time_models = _build_time_labels(model_date)

    # --- pulizia csv precedenti ---
    for f in cfg.dir_out.glob("*.csv"):
        f.unlink()

    # --- gap-filling archivio ultimi 10 giorni ---
    _fill_missing_archive_days(cfg.dir_archive, dates["pp_dates"])

    # --- coordinate e landmask ---
    geo_xyz = pd.read_csv(cfg.dir_out / "geo" / "XYZ_mol02.csv")
    geo_xyz.columns = ["X", "Y", "Z"][: len(geo_xyz.columns)]  # normalizza nomi, come in R (X,Y,Z)

    import pyreadr

    if cfg.landmask_path.suffix == ".rds":
        landmask_result = pyreadr.read_r(str(cfg.landmask_path))
        landmask = next(iter(landmask_result.values())).iloc[:, 0].to_numpy()
    else:
        landmask = np.load(cfg.landmask_path)

    n_points = len(geo_xyz)
    meta_cols = pd.DataFrame({"ID": np.arange(1, n_points + 1)})
    meta_cols = pd.concat([meta_cols, geo_xyz.reset_index(drop=True)], axis=1)
    meta_cols["P"] = landmask

    outputs: dict[str, Path] = {}

    # --- matrice tmax (5 giorni) ---
    tmax_matrix = _extract_tmax_matrix(cfg.work_dir, dates["ff_dates_max_nam"])
    if tmax_matrix is not None:
        moltmax = pd.concat(
            [meta_cols, pd.DataFrame(tmax_matrix, columns=dates["ff_dates_max_nam"])], axis=1
        )
        out_path = cfg.dir_out / "moloch02_tmax_latest.csv"
        moltmax.to_csv(out_path, index=False)
        outputs["tmax"] = out_path

        archive_name = f"moloch02_tmax_{model_date.isoformat()}.csv"
        moltmax.to_csv(cfg.dir_archive_current_forecast / archive_name, index=False)

    # --- acclimatazione (10 giorni, daymax) ---
    acclim_cols = [estrai_acclimdays(cfg.dir_archive / f, n_points) for f in dates["pp_dates"]]
    acclim_df = pd.concat(
        [meta_cols, pd.DataFrame(np.column_stack(acclim_cols), columns=dates["pp_dates_nam"])], axis=1
    )
    out_path = cfg.dir_out / "acclim_data_latest.csv"
    acclim_df.to_csv(out_path, index=False)
    outputs["acclim"] = out_path

    # --- estrazioni orarie a 3 giorni (72 colonne) ---
    def extract_hourly(par: str) -> pd.DataFrame:
        mats = [estrai_param(cfg.dir_archive / f, par, n_points) for f in dates["ff_dates_prev"]]
        combined = np.column_stack(mats)  # (n_points, 72) atteso
        if combined.shape[1] != len(time_models):
            warnings.warn(
                f"Estrazione '{par}': attese {len(time_models)} colonne orarie, ottenute {combined.shape[1]} "
                "(probabile file mancante nell'archivio per uno dei 3 giorni)."
            )
            cols = [f"{par}_{i}" for i in range(combined.shape[1])]
        else:
            cols = time_models
        return pd.concat([meta_cols, pd.DataFrame(combined, columns=cols)], axis=1)

    wbgtsun_df = extract_hourly("wbgtsun")
    wbgtsha_df = extract_hourly("wbgtsha")
    utci_df = extract_hourly("utci")
    utcisha_df = extract_hourly("utcisha")
    rkcla_int_df = extract_hourly("rkcla_int")
    rkcla_nint_df = extract_hourly("rkcla_nint")
    windchill_df = extract_hourly("windchill")

    for name, df in [
        ("wbgtsun", wbgtsun_df),
        ("wbgtsha", wbgtsha_df),
        ("utci", utci_df),
        ("utcisha", utcisha_df),
    ]:
        out_path = cfg.dir_out / f"{name}_latest.csv"
        df.to_csv(out_path, index=False)
        outputs[name] = out_path

    # --- heat risk level "poste" (colonne orarie, classi -1) ---
    hourly_col_start = meta_cols.shape[1]  # indice della prima colonna oraria

    def poste_risk(df: pd.DataFrame) -> pd.DataFrame:
        hourly = df.iloc[:, hourly_col_start:].to_numpy(dtype=float)
        risk = R.heat_risk_level(hourly, 0, 25.3) - 1
        return pd.concat([df.iloc[:, :hourly_col_start].reset_index(drop=True), pd.DataFrame(risk)], axis=1)

    dataposte_risk_sha_df = poste_risk(wbgtsha_df)
    dataposte_risk_sun_df = poste_risk(wbgtsun_df)

    cfg.poste_risk_dir.mkdir(parents=True, exist_ok=True)
    dataposte_risk_sha_df.to_csv(cfg.poste_risk_dir / "hrisksha_latest.csv", index=False)
    dataposte_risk_sun_df.to_csv(cfg.poste_risk_dir / "hrisksun_latest.csv", index=False)
    outputs["hrisksha"] = cfg.poste_risk_dir / "hrisksha_latest.csv"
    outputs["hrisksun"] = cfg.poste_risk_dir / "hrisksun_latest.csv"

    archive_risk_name = f"hrisksha_{model_date.isoformat()}.csv"
    dataposte_risk_sha_df.to_csv(cfg.dir_archive_current_forecast / archive_risk_name, index=False)

    (cfg.poste_risk_dir / "date_latest.txt").write_text((model_date + timedelta(days=1)).strftime("%Y%m%d"))

    # --- scritture rimanenti: FIX applicato rispetto al sorgente R ---
    # Nel sorgente R originale, windchill_df veniva rinominato ma MAI
    # scritto su CSV: al suo posto rkcla_int_df veniva scritto due volte
    # (la seconda sovrascriveva la prima con nomi colonna corretti, ma
    # il prodotto windchill_latest.csv non veniva mai prodotto nonostante
    # fosse stato calcolato). Corretto qui e nel sorgente R.
    out_path = cfg.dir_out / "windchill_latest.csv"
    windchill_df.to_csv(out_path, index=False)
    outputs["windchill"] = out_path

    out_path = cfg.dir_out / "rkcla_int_latest.csv"
    rkcla_int_df.to_csv(out_path, index=False)
    outputs["rkcla_int"] = out_path

    out_path = cfg.dir_out / "rkcla_nint_latest.csv"
    rkcla_nint_df.to_csv(out_path, index=False)
    outputs["rkcla_nint"] = out_path

    # --- date_latest.txt e mat_date_acclim_latest.csv in dir_out ---
    (cfg.dir_out / "date_latest.txt").write_text((model_date + timedelta(days=1)).strftime("%Y%m%d"))

    _ = oralegale_l(model_date)  # calcolato in R (variabile 'delta') ma non utilizzato oltre; mantenuto per fedelta'

    pp_dates_full = _daterange(model_date - timedelta(days=10), model_date)
    mat_date_path = cfg.dir_out / "mat_date_acclim_latest.csv"
    with open(mat_date_path, "a") as f:
        for d in pp_dates_full:
            f.write(f"{d.isoformat()}\n")
    outputs["mat_date_acclim"] = mat_date_path

    return outputs
