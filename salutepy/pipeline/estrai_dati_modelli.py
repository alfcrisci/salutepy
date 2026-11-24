"""
Porting di estrai_dati_modelli.r: per ogni file GRIB2 disponibile in
model_dir (run 00 e run 12 del GFS), lancia extract_samples() e poi
rimuove i file sorgente processati.

A differenza del sorgente R — che lancia un nuovo processo `Rscript
extract_samples_simple.R ...` per ogni file (costo di avvio interprete
ad ogni iterazione, gia' segnalato come punto di inefficienza nella review
iniziale del codice R) — qui extract_samples() viene chiamata
direttamente in-process, eliminando quell'overhead.

ESTENSIONE rispetto al sorgente R originale: supporto per PIU' set di
punti di estrazione (non solo le stazioni di verifica), ciascuno con
etichetta propria nel nome file di output — utile per estrarre anche sui
116 capoluoghi di provincia italiani (capoluoghi_province_italiani.csv),
oltre alle stazioni di verifica meteorologica.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from .extract_samples import capoluoghi_stations_csv, default_stations_csv, extract_samples


@dataclass
class StationSet:
    label: str  # usata come suffisso nel nome file di output (es. "verify", "capoluoghi")
    stations_csv: Path


@dataclass
class EstraiDatiConfig:
    model_dir: Path
    dir_archive_extract: Path
    stations_csv: Path | None = None  # set principale; default: stazioni di verifica bundled nel package
    extra_station_sets: list[StationSet] = field(default_factory=list)
    # es.: extra_station_sets=[StationSet("capoluoghi", capoluoghi_stations_csv())]

    def __post_init__(self):
        if self.stations_csv is None:
            self.stations_csv = default_stations_csv()


def with_capoluoghi(cfg: EstraiDatiConfig) -> EstraiDatiConfig:
    """
    Ritorna una copia di `cfg` con l'estrazione aggiuntiva sui 116
    capoluoghi di provincia italiani (capoluoghi_province_italiani.csv,
    bundled nel package) abilitata, oltre al set principale gia'
    configurato. Non modifica `cfg` in place.
    """
    already_present = any(s.label == "capoluoghi" for s in cfg.extra_station_sets)
    if already_present:
        return cfg
    return replace(
        cfg,
        extra_station_sets=[*cfg.extra_station_sets, StationSet("capoluoghi", capoluoghi_stations_csv())],
    )


def run(cfg: EstraiDatiConfig) -> list[Path]:
    """
    Orchestrazione completa, equivalente a estrai_dati_modelli.r, esteso
    per lanciare l'estrazione su tutti i set di stazioni configurati
    (quello principale + eventuali extra_station_sets) sugli stessi file
    GRIB2, prima di rimuoverli.
    """
    files_mol12 = sorted(cfg.model_dir.glob("2*12_mol01gfs.grb2"))
    files_mol00 = sorted(cfg.model_dir.glob("2*00_mol01gfs.grb2"))

    station_sets = [StationSet("", cfg.stations_csv), *cfg.extra_station_sets]

    outputs = []

    for f in files_mol00:
        for s in station_sets:
            suffix = f"_{s.label}" if s.label else ""
            out_stem = cfg.dir_archive_extract / f"{f.stem}{suffix}"
            out = extract_samples(f, out_stem, 1, 48, "MOLOCH02_GFS_00", s.stations_csv)
            outputs.append(out)

    for f in files_mol12:
        for s in station_sets:
            suffix = f"_{s.label}" if s.label else ""
            out_stem = cfg.dir_archive_extract / f"{f.stem}{suffix}"
            out = extract_samples(f, out_stem, 13, 61, "MOLOCH02_GFS_12", s.stations_csv)
            outputs.append(out)

    for f in files_mol00 + files_mol12:
        f.unlink(missing_ok=True)

    return outputs
