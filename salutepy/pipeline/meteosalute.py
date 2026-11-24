"""
Porting di meteosalute_op_dir_mol01.r: split GRIB2 via cdo, lettura di
ogni timestep, calcolo di tutti gli indici biometeorologici, scrittura
NetCDF orario.

Mapping delle sezioni rispetto al file R originale:
- righe 1-107 (setup, attesa file, copia)      -> locate_and_stage_input()
- righe 109-137 (cdo -splitsel)                -> split_moloch_grib()
- righe 139-489 (loop per timestep)            -> process_timestep() dentro run()

Note di fedeltà:
- `sunposition()` (pacchetto suntools, sorgente non disponibile) è
  sostituita da `sunposition_norefrac` locale — vedi
  salutepy.indices.sun_position per i dettagli della sostituzione.
- Il parallelismo R (`future_lapply`/`plan(multisession)`) non è
  riprodotto: qui i cicli per-pixel di wbgt_liljegren/PMV sono loop
  Python sequenziali (già vettorizzati dove la formula lo permette,
  es. sunposition_norefrac). Se le prestazioni non bastano su griglie
  grandi, si può parallelizzare con joblib/multiprocessing in un secondo
  momento senza cambiare l'interfaccia delle funzioni.
- La lettura GRIB2 (rasterio, posizionale) non è stata testata su un
  file reale in questa sessione — vedi nota in grib_reader.py.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from ..indices import compat as R
from ..indices.misc import metabolism, reducewind, windchill_class_wk
from ..indices.sun_position import sunposition_norefrac
from .grib_reader import NAMES_METEOS, read_moloch_split
from .netcdf_writer import write_meteosalute_nc
from .shell import run_cmd

MODEL_PATTERN = "*12_mol01gfs.grb2"
MODEL_PATTERN_TMAXMIN = "*12_mol01gfs.grb2.DAILYVARS"

FILLVALUE = -999.9


@dataclass
class MeteosaluteConfig:
    model_dir: Path
    work_dir: Path
    script_dir: Path
    fillrkmoloch_path: Path  # fillrkmoloch02.rds (o equivalente gia' convertito)
    send_mail_nomodel_cmd: str | None = None  # es. "python3 <script_dir>/send_mail_nomodel.py"


def _load_fillrkmoloch(path: Path) -> np.ndarray:
    """
    Carica il vettore di fallback 'grads-safe' (358701 elementi) usato
    per evitare il bug GrADS di classe unica. Il file originale e' un
    .rds: letto qui con pyreadr. In alternativa si puo' pre-convertire
    una volta in .npy/.csv con R (`saveRDS`->`write.csv`) e caricarlo
    direttamente con numpy, evitando la dipendenza da pyreadr.
    """
    if path.suffix == ".rds":
        import pyreadr

        result = pyreadr.read_r(str(path))
        df = next(iter(result.values()))
        return df.iloc[:, 0].to_numpy(dtype=float)
    else:
        return np.loadtxt(path)


def locate_and_stage_input(cfg: MeteosaluteConfig, wait_seconds: int = 30) -> tuple[Path, Path, str]:
    """
    Equivalente delle righe 50-100 di meteosalute_op_dir_mol01.r:
    individua il file GRIB2 del run corrente, attende (Sys.sleep),
    verifica la presenza, lo copia in work_dir.

    Returns
    -------
    (percorso_grib_in_workdir, percorso_tmaxmin_in_workdir, dateboldate)
    """
    candidates = sorted(cfg.model_dir.glob(MODEL_PATTERN))
    if not candidates:
        _send_nomodel_mail(cfg)
        raise RuntimeError("Mancano le previsioni del modello: nessun file GRIB2 trovato in model_dir.")

    moloch_namepath = candidates[0]
    tmaxmin_candidates = sorted(cfg.model_dir.glob(MODEL_PATTERN_TMAXMIN))
    if not tmaxmin_candidates:
        _send_nomodel_mail(cfg)
        raise RuntimeError("Mancano le previsioni del modello: nessun file .DAILYVARS trovato in model_dir.")
    moloch_namepath_tmaxmin = tmaxmin_candidates[0]

    import time

    time.sleep(wait_seconds)

    moloch_name = moloch_namepath.name
    moloch_name_tmaxmin = moloch_namepath_tmaxmin.name
    dateboldate = moloch_name[:8]

    dest = cfg.work_dir / moloch_name
    dest_tmaxmin = cfg.work_dir / moloch_name_tmaxmin

    dest.write_bytes(moloch_namepath.read_bytes())
    dest_tmaxmin.write_bytes(moloch_namepath_tmaxmin.read_bytes())

    return dest, dest_tmaxmin, dateboldate


def _send_nomodel_mail(cfg: MeteosaluteConfig) -> None:
    if cfg.send_mail_nomodel_cmd:
        run_cmd(cfg.send_mail_nomodel_cmd, stop_on_error=False)


def split_moloch_grib(moloch_path: Path, work_dir: Path) -> list[Path]:
    """
    Equivalente di:
        cdo -f grb2 -splitsel,1 -seltimestep,11/83 <moloch_name> moloch_

    Returns
    -------
    Lista ordinata dei file splittati prodotti (esclude eventuali .nc
    residui, come il file.remove(list.files(pattern=".nc")) del sorgente R).
    """
    cmdsplit = f"cdo -f grb2 -splitsel,1 -seltimestep,11/83 {moloch_path.name} moloch_"

    rc = run_cmd(cmdsplit, stop_on_error=False, cwd=str(work_dir))
    if rc.returncode != 0:
        raise RuntimeError(f"cdo splitsel fallito (rc={rc.returncode}) su {moloch_path}")

    for f in work_dir.glob("*.nc"):
        f.unlink()

    ls_files = sorted(work_dir.glob("moloch_*"))

    if len(ls_files) < 2:
        raise RuntimeError(
            f"cdo splitsel ha prodotto solo {len(ls_files)} file, atteso >=2. File sorgente: {moloch_path}"
        )

    return ls_files


def process_timestep(
    fname: Path,
    i: int,
    dateboldate: str,
    fillrkmoloch: np.ndarray,
    work_dir: Path,
) -> Path:
    """
    Equivalente del corpo del ciclo `for (i in 1:(length(ls_files)-1))`
    in meteosalute_op_dir_mol01.r: legge un timestep da file GRIB2, calcola
    tutti gli indici, scrive il NetCDF orario. Ritorna il percorso del
    file scritto.

    Thin wrapper attorno a process_timestep_from_grib(): la lettura del
    file e' l'unica parte specifica della fonte MOLOCH — il calcolo
    indici e' condiviso con qualunque altra fonte dati compatibile (es.
    ICON-EU scaricato da DWD, vedi pipeline.icon_download), che costruisce
    un MolochGribData equivalente senza passare da un file GRIB2 su disco.
    """
    grib = read_moloch_split(fname, names=NAMES_METEOS)
    return process_timestep_from_grib(grib, i, dateboldate, fillrkmoloch, work_dir)


def process_timestep_from_grib(
    grib,
    i: int,
    dateboldate: str,
    fillrkmoloch: np.ndarray,
    work_dir: Path,
) -> Path:
    """
    Corpo del calcolo indici + scrittura NetCDF, indipendente dalla fonte
    dati: accetta direttamente un oggetto MolochGribData (o compatibile:
    stesso set di attributi `variables`, `lon`, `lat`, `lonvect`, `latvect`)
    invece di un percorso file. Vedi process_timestep() per il wrapper
    che legge da un file GRIB2 MOLOCH.
    """
    base_date = datetime.strptime(dateboldate, "%Y%m%d").replace(tzinfo=timezone.utc)

    # run 12 date model +1 (riga 146 R)
    posixtimefile_out = base_date + timedelta(days=1) + timedelta(hours=(i - 1))
    outfilename = work_dir / f"mol01gfs12_{posixtimefile_out.strftime('%Y%m%d%H')}.nc"

    # tempo usato per calcolo posizione solare / wbgt (riga 171 R, nota: +1 secondo)
    posixtimefile = base_date + timedelta(hours=(i - 1), seconds=1)

    # variabile "time" del NetCDF: ore da 1992-01-01 00:00:00 (riga 173 R, usa (i-1) senza +1s)
    epoch_1992 = datetime(1992, 1, 1, tzinfo=timezone.utc)
    time_ref = base_date + timedelta(hours=(i - 1))
    timefile = (time_ref - epoch_1992).total_seconds() / 3600.0

    n_pixels = len(grib.variables["T_fhg"])

    t_fhg = grib.variables["T_fhg"]
    td_fhg = grib.variables["TD_fhg"]
    u_fhg = grib.variables["u_fhg"]
    v_fhg = grib.variables["v_fhg"]
    dswrf_sfc = grib.variables["dswrf_sfc"]
    n_atm = grib.variables["N_atm"]
    gust_sfc = grib.variables["gust_sfc"]
    pmsl = grib.variables["pmsl"]
    lapr = grib.variables["lapr"]

    lonvect, latvect = grib.lonvect, grib.latvect

    # --- posizione solare (sostituisce suntools::sunposition, vedi sun_position.py) ---
    sunpos = sunposition_norefrac(posixtimefile, latvect, lonvect)

    # --- vento e umidita' ---
    wind_speed = np.sqrt(u_fhg ** 2 + v_fhg ** 2)  # 10m
    wind_speed2m = reducewind(wind_speed)  # 2m

    rh_fhg = R.dewp2hurs(t_fhg, td_fhg)

    # --- WBGT outdoor (Liljegren), stesso timestamp per tutti i pixel ---
    dates_per_pixel = [posixtimefile] * n_pixels
    wbgt_outdoors = R.wbgt_Liljegren(t_fhg, td_fhg, wind_speed2m, dswrf_sfc, dates_per_pixel, lonvect, latvect)

    thi_index = R.THI(t_fhg, rh_fhg)
    tglob = wbgt_outdoors.Tg
    wbgtsha_var = wbgt_outdoors.Tnwb * 0.7 + 0.3 * t_fhg
    wbgtsun_var = wbgt_outdoors.data

    mrt = R.mrt_globe(t_fhg, tglob, wind_speed2m, 0.15)  # 150mm = 0.15m (bug di unita' risolto)

    # --- indici di comfort ---
    met_data = metabolism(t_fhg) / 1.8
    met_data = np.where(met_data > 58.1 * 1.1, met_data, 70)

    pmv_sha = R.PMV_ISO7730(t_fhg, rh_fhg, wind_speed, t_fhg, met_data, 0, 1.1)
    clomin_var = R.clomin_7730(t_fhg, rh_fhg, wind_speed, t_fhg, met_data, 0)
    clomax_var = R.clomax_7730(t_fhg, rh_fhg, wind_speed, t_fhg, met_data, 0)

    # NOTA: il wrapper Python utci()/utci_class() usa l'ordine posizionale
    # (tdb, tr, v, rh), DIVERSO da quello di rBiometeo (t, rh, wind, trad) —
    # per questo qui si usano keyword args espliciti, per evitare ambiguita'.
    utci_var = R.utci(tdb=t_fhg, tr=mrt, v=wind_speed, rh=rh_fhg)
    utci_shade_var = R.utci(tdb=t_fhg, tr=t_fhg, v=wind_speed, rh=rh_fhg)
    utci_cla_var = R.utci_class(tdb=t_fhg, tr=mrt, v=wind_speed, rh=rh_fhg)
    utci_shade_cla_var = R.utci_class(tdb=t_fhg, tr=t_fhg, v=wind_speed, rh=rh_fhg)

    stosun_var = R.steadman_outdoor_sun(t_fhg, rh_fhg, wind_speed, dswrf_sfc, sunpos["elevation"])
    stosha_var = R.steadman_outdoor_shade(t_fhg, rh_fhg, wind_speed)
    stoind_var = R.steadman_indoor(t_fhg, rh_fhg)
    stosuncla_var = R.steadman_class(stosun_var)
    stoshacla_var = R.steadman_class(stosha_var)
    stosun_var = np.where(np.isnan(stosun_var), stosha_var, stosun_var)

    deficitsat_var = R.deficitsat(t_fhg, rh_fhg)

    heat_risk_int_class = R.heat_risk_level(wbgtsun_var, 0, 23.5)
    heat_risk_nint_class = R.heat_risk_level(wbgtsun_var, 0, 25.3)
    heat_risksha_int_class = R.heat_risk_level(wbgtsha_var, 0, 23.5)
    heat_risksha_nint_class = R.heat_risk_level(wbgtsha_var, 0, 25.3)

    # "prevent grads bugs of unique class": forza almeno un pixel di classe 2
    mask_fill2 = fillrkmoloch == 2
    heat_risk_int_class = np.where(mask_fill2, 2, heat_risk_int_class)
    heat_risk_nint_class = np.where(mask_fill2, 2, heat_risk_nint_class)
    heat_risksha_int_class = np.where(mask_fill2, 2, heat_risksha_int_class)
    heat_risksha_nint_class = np.where(mask_fill2, 2, heat_risksha_nint_class)

    lostproductivity = R.lost_productivity(wbgtsun_var)
    lostproductivityshade = R.lost_productivity(wbgtsha_var)

    wbgt_ti_wtl_sha = R.mean_wbgtTI_WTL(wbgtsha_var)
    wbgt_twtl_sha = R.mean_wbgtWTL(wbgtsha_var)
    wbgt_ti_wtl_sun = R.mean_wbgtTI_WTL(wbgtsun_var)
    wbgt_twtl_sun = R.mean_wbgtWTL(wbgtsun_var)
    air_ti_wtl = R.mean_airTI_WTL(t_fhg)
    air_twtl = R.mean_airTWTL(t_fhg)

    windchill_var = R.windchill(t_fhg, wind_speed2m)
    windchill_var = np.where(windchill_var > 10, FILLVALUE, windchill_var)
    windchill_cla_var = windchill_class_wk(windchill_var)
    windchill_cla_var = np.where(windchill_cla_var > 5.1, FILLVALUE, windchill_cla_var)

    frostime_var = R.frostime(t_fhg, wind_speed2m)

    # "prevent grads bugs of unique class": se quasi tutti i pixel sono classe 1,
    # sostituisce l'intero array col fallback grads-safe
    if np.sum(windchill_cla_var > 1) < 2:
        windchill_cla_var = fillrkmoloch

    # --- scrittura NetCDF ---
    data = {
        "ta2m": t_fhg,
        "urel": rh_fhg,
        "v10m": wind_speed,
        "gust": gust_sfc,
        "pmsl": pmsl / 100,
        "pvap": R.pvap(t_fhg, rh_fhg),
        "defsat": deficitsat_var,
        "csky": n_atm,
        "pmvsha": pmv_sha,
        "clomin": clomin_var,
        "clomax": clomax_var,
        "utci": utci_var,
        "utcisha": utci_shade_var,
        "utciclasha": utci_shade_cla_var,
        "stosun": stosun_var,
        "stosha": stosha_var,
        "stoind": stoind_var,
        "temprad": mrt,
        "rcorte": dswrf_sfc,
        "stosuncla": stosuncla_var,
        "stoshacla": stoshacla_var,
        "utcicla": utci_cla_var,
        "wbgtsha": wbgtsha_var,
        "wbgtsun": wbgtsun_var,
        "tglob": tglob,
        "rkcla_int": heat_risk_int_class,
        "rkcla_nint": heat_risk_nint_class,
        "rkshacla_int": heat_risksha_int_class,
        "rkshacla_nint": heat_risksha_nint_class,
        "lprodsha": lostproductivityshade,
        "lprod": lostproductivity,
        "wbgttiwtlsha": wbgt_ti_wtl_sha,
        "wbgttwtlsha": wbgt_twtl_sha,
        "wbgttiwtlsun": wbgt_ti_wtl_sun,
        "wbgttwtlsun": wbgt_twtl_sun,
        "airtiwtl": air_ti_wtl,
        "airtwtl": air_twtl,
        "windchill": windchill_var,
        "windcla": windchill_cla_var,
        "frostime": frostime_var,
        "thi": thi_index,
        "lapr": lapr,
    }

    write_meteosalute_nc(
        outfilename,
        lon=grib.lon,
        lat=grib.lat,
        time_hours_since_1992=timefile,
        n_lon=len(grib.lon),
        n_lat=len(grib.lat),
        data=data,
    )

    return outfilename


def run(cfg: MeteosaluteConfig) -> list[Path]:
    """
    Orchestrazione completa, equivalente a tutto meteosalute_op_dir_mol01.r.
    Ritorna la lista dei file NetCDF orari scritti.
    """
    fillrkmoloch = _load_fillrkmoloch(cfg.fillrkmoloch_path)

    moloch_path, moloch_tmaxmin_path, dateboldate = locate_and_stage_input(cfg)

    ls_files = split_moloch_grib(moloch_path, cfg.work_dir)

    outputs = []
    for i in range(1, len(ls_files)):  # 1:(length(ls_files)-1) in R (1-indexed, ultimo escluso)
        fname = ls_files[i - 1]
        outpath = process_timestep(fname, i, dateboldate, fillrkmoloch, cfg.work_dir)
        outputs.append(outpath)

    for f in ls_files:
        f.unlink(missing_ok=True)

    (cfg.work_dir / "okfileprocedura.file").write_text("ok")

    return outputs
