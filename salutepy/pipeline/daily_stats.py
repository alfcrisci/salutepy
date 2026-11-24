"""
Porta Python della parte "orchestrazione cdo" di salute_elab.r
(giorni 1-3: cdo cat/timmax/timmin/setdate/sub; soglie gt34/gt35/gelo).

Le formule biometeorologiche (WBGT, Steadman, UTCI, ecc.) NON sono qui:
vivono in salutepy.indices e vengono portate dai sorgenti R di
HeatStress/rBiometeo una volta disponibili.
"""

from __future__ import annotations

from pathlib import Path

from .shell import run_cmd


def daily_max_min_excursion(
    day_nc: str | Path,
    date_str: str,
    varname: str = "ta2m",
    cleanup_tmp: bool = True,
) -> tuple[Path, Path, Path]:
    """
    Equivalente Python del blocco "first 3 days" in salute_elab.r:

        cdo -selvar,ta2m -timmax day_YYYY-MM-DD.nc dmax_YYYY-MM-DD.nc
        cdo -selvar,ta2m -timmin day_YYYY-MM-DD.nc dmin_YYYY-MM-DD.nc
        cdo -setdate,YYYY-MM-DD dmax_YYYY-MM-DD.nc max_YYYY-MM-DD.nc
        cdo -setdate,YYYY-MM-DD dmin_YYYY-MM-DD.nc min_YYYY-MM-DD.nc
        cdo sub max_YYYY-MM-DD.nc min_YYYY-MM-DD.nc excursion_YYYY-MM-DD.nc

    Parameters
    ----------
    day_nc : path del file giornaliero orario di input (day_<date>.nc)
    date_str : data in formato YYYY-MM-DD (usata sia per i nomi file che
        per cdo -setdate)
    varname : variabile su cui calcolare max/min (default ta2m, come in R)
    cleanup_tmp : se True rimuove i file intermedi dmax_/dmin_ (come in R)

    Returns
    -------
    (max_path, min_path, excursion_path)
    """
    day_nc = Path(day_nc)
    workdir = day_nc.parent

    dmax = workdir / f"dmax_{date_str}.nc"
    dmin = workdir / f"dmin_{date_str}.nc"
    fmax = workdir / f"max_{date_str}.nc"
    fmin = workdir / f"min_{date_str}.nc"
    fexc = workdir / f"excursion_{date_str}.nc"

    run_cmd(f"cdo -selvar,{varname} -timmax {day_nc} {dmax}")
    run_cmd(f"cdo -selvar,{varname} -timmin {day_nc} {dmin}")
    run_cmd(f"cdo -setdate,{date_str} {dmax} {fmax}")
    run_cmd(f"cdo -setdate,{date_str} {dmin} {fmin}")

    if cleanup_tmp:
        dmax.unlink(missing_ok=True)
        dmin.unlink(missing_ok=True)

    run_cmd(f"cdo sub {fmax} {fmin} {fexc}")

    return fmax, fmin, fexc


def concat_days(pattern_dir: str | Path, pattern: str, out_nc: str | Path) -> Path:
    """
    Equivalente di: cdo cat max_*.nc tempmol01daymax12.nc
    (e analogo per min_*.nc). 'pattern' e' passato cosi' com'e' a cdo,
    che gestisce i wildcard tramite la shell (shell=True in run_cmd).
    """
    out_nc = Path(out_nc)
    run_cmd(f"cdo cat {Path(pattern_dir) / pattern} {out_nc}")
    return out_nc


def threshold_map(
    in_nc: str | Path,
    out_nc: str | Path,
    varname: str,
    new_varname: str,
    op: str,
    threshold: float,
) -> Path:
    """
    Equivalente di:
        cdo -expr,"ta2m=(ta2m>=34)?1:0;" in.nc tmp.nc
        cdo chname,ta2m,t_gt34 tmp.nc out.nc

    op deve essere uno tra ">=", "<=", ">", "<", "==".
    Esempio (giorni con Tmax>=34):
        threshold_map(in_nc, out_nc, "ta2m", "t_gt34", ">=", 34)
    Esempio (giorni di gelo, Tmin<=0):
        threshold_map(in_nc, out_nc, "ta2m", "t_dayice", "<=", 0)
    """
    if op not in {">=", "<=", ">", "<", "=="}:
        raise ValueError(f"Operatore non supportato: {op}")

    in_nc = Path(in_nc)
    out_nc = Path(out_nc)
    tmp_nc = out_nc.with_name(f"_tmp_{out_nc.name}")

    expr = f'{varname}=({varname}{op}{threshold})?1:0;'
    run_cmd(f'cdo -expr,"{expr}" {in_nc} {tmp_nc}')
    run_cmd(f"cdo chname,{varname},{new_varname} {tmp_nc} {out_nc}")
    tmp_nc.unlink(missing_ok=True)

    return out_nc


def split_by_day(in_nc: str | Path, out_prefix: str | Path) -> None:
    """Equivalente di: cdo splitday in.nc out_prefix"""
    run_cmd(f"cdo splitday {in_nc} {out_prefix}")
