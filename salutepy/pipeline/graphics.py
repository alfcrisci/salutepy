"""
Porting di graph_daily_wkclimate_mol01gfs.r, graph_daily_max.r,
graph_daily_t_dayice.r, graph_daily_onlybio.r (+ i rispettivi
plot_images_*.r) e plot_logo_wkclimate.r / plot_logo_wkclimate_onlybio.r.

Ogni funzione qui accetta un parametro `render_fn` (default
`grads.run_gradsbio`) che genera il PNG grezzo <var>_<id>.png a partire
dal file NetCDF — iniettabile con un renderer fittizio per testare la
logica di orchestrazione/naming/post-processing senza GrADS installato
(vedi grads.py per i limiti di questa dipendenza esterna).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from . import grads as grads_mod
from . import image_postprocess as imgpp
from .list_bio import VarSpec, list_bio_day, list_bio_daymax, list_bio_only, list_bio_t_dayice, list_bio_wkclimate

RenderFn = Callable[[str, str, int, str], None]  # (file, var, id_, codegs) -> crea <var>_<id>.png in scriptdir


@dataclass
class GraphicsConfig:
    scriptdir: Path  # dove GrADS scrive i .png grezzi e dove stanno gli script .gs
    datadir: Path  # file NetCDF orari (products23)
    dir_products: Path  # per individuare inidate dal nome del file *.grb2
    graphdir: Path  # output finale delle mappe
    logoimage: Path  # logo CNR-IBE/Lamma per compose_biometeo_image
    logoimagewk: Path  # logo watermark per add_watermarks
    discimagewk: Path  # disclaimer watermark per add_watermarks
    pattern_model: str = "mol01gfs12_"
    pattern_dayice: str = "mol01_t_dayice_daymin12_"
    pattern_tgmax: str = "mol01_f34_daymax12_"


def _inidate(cfg: GraphicsConfig):
    """Equivalente di: as.Date(substr(list.files(pattern="grb2$"),1,8), format="%Y%m%d")."""
    candidates = sorted(cfg.dir_products.glob("*grb2"))
    if not candidates:
        raise FileNotFoundError(f"Nessun file '*grb2' trovato in {cfg.dir_products} per determinare inidate.")
    return datetime.strptime(candidates[0].name[:8], "%Y%m%d").date()


def _render(cfg: GraphicsConfig, render_fn: RenderFn, file: Path, var: str, id_: int, codegs: str) -> Path:
    render_fn(str(file), var, id_, codegs)
    return cfg.scriptdir / f"{var}_{id_}.png"


def _default_render_fn(cfg: GraphicsConfig) -> RenderFn:
    def _fn(file: str, var: str, id_: int, codegs_name: str) -> None:
        codegs = cfg.scriptdir / codegs_name
        grads_mod.run_gradsbio(file, var, id_, codegs)

    return _fn


# ---------------------------------------------------------------------------
# graph_daily_wkclimate_mol01gfs.r
# ---------------------------------------------------------------------------

def graph_daily_wkclimate(cfg: GraphicsConfig, render_fn: RenderFn | None = None) -> list[Path]:
    """
    Porting di graph_daily_wkclimate_mol01gfs.r + plot_images_wkclimate_mol01gfs.r.
    Per ogni file orario x per ogni variabile di list_bio_wkclimate(),
    genera la mappa con GrADS e la post-processa.
    """
    render_fn = render_fn or _default_render_fn(cfg)

    for f in cfg.scriptdir.glob("*.png"):
        f.unlink()

    hourly_files = sorted(cfg.datadir.glob(f"{cfg.pattern_model}*.nc"))
    inidate = _inidate(cfg)
    variables = list_bio_wkclimate()

    outputs = []
    for z, hourly_file in enumerate(hourly_files, start=1):
        # timefromfilebolam: rimuove pattern_model e ".nc", parsa come %Y%m%d%H
        stem = hourly_file.name.replace(cfg.pattern_model, "").replace(".nc", "")
        file_time = datetime.strptime(stem, "%Y%m%d%H")
        id_ = file_time.hour
        idhour = z - 1

        for spec in variables:
            codegs = grads_mod.codegs_for(spec.floatflag)
            raw_png = _render(cfg, render_fn, hourly_file, spec.var, id_, codegs)

            run_annotation = (
                f" Init.: {inidate} +12 UTC   Valid. CEST: "
                f"{file_time.strftime('%a, %d-%m-%Y %H')} t :+ {idhour} h"
            )
            out_path = cfg.graphdir / f"{spec.fileout}_{idhour}.png"
            imgpp.compose_biometeo_image(
                raw_png, cfg.logoimage, spec.textvar, run_annotation, out_path, title_size=21, subtitle_size=13
            )
            outputs.append(out_path)

    return outputs


# ---------------------------------------------------------------------------
# graph_daily_max.r (mappe soglia >35) e graph_daily_t_dayice.r (giorni di gelo)
# ---------------------------------------------------------------------------

def _graph_daily_single_var_5days(
    cfg: GraphicsConfig,
    spec: VarSpec,
    pattern: str,
    render_fn: RenderFn | None,
    suffix_files: bool,
) -> list[Path]:
    """
    Corpo comune a graph_daily_max.r e graph_daily_t_dayice.r: 5 giorni
    di forecast, un file per giorno, un'unica variabile.
    """
    render_fn = render_fn or _default_render_fn(cfg)

    for f in cfg.scriptdir.glob("*.png"):
        f.unlink()

    inidate = _inidate(cfg)
    today = inidate + timedelta(days=1)
    days = [today + timedelta(days=i) for i in range(5)]
    d_files = [cfg.datadir / f"{pattern}{d.strftime('%d')}.nc" for d in days]

    outputs = []
    for idx, (d_file, day) in enumerate(zip(d_files, days), start=1):
        codegs = grads_mod.codegs_for(spec.floatflag)
        raw_png = _render(cfg, render_fn, d_file, spec.var, idx, codegs)

        run_annotation = f"Init.: {inidate} Valid.: {day}"
        out_name = f"{spec.fileout}_{idx}.png" if suffix_files else f"{spec.fileout}.png"
        out_path = cfg.graphdir / out_name
        imgpp.compose_biometeo_image(
            raw_png, cfg.logoimage, spec.textvar, run_annotation, out_path, title_size=18, subtitle_size=13
        )
        outputs.append(out_path)

    return outputs


def graph_daily_max(cfg: GraphicsConfig, render_fn: RenderFn | None = None) -> list[Path]:
    """Porting di graph_daily_max.r + plot_images_daymax.r (mappe soglia Tmax>35)."""
    for f in cfg.scriptdir.glob("tmax_over*.png"):
        f.unlink()
    return _graph_daily_single_var_5days(cfg, list_bio_daymax(), cfg.pattern_tgmax, render_fn, suffix_files=True)


def graph_daily_t_dayice(cfg: GraphicsConfig, render_fn: RenderFn | None = None) -> list[Path]:
    """Porting di graph_daily_t_dayice.r + plot_images_t_dayice.r (giorni di gelo)."""
    for f in cfg.scriptdir.glob("t_dayice_*.png"):
        f.unlink()
    return _graph_daily_single_var_5days(cfg, list_bio_t_dayice(), cfg.pattern_dayice, render_fn, suffix_files=True)


# ---------------------------------------------------------------------------
# graph_daily_onlybio.r (+ plot_images_onlybio.r, plot_images_day.r)
# ---------------------------------------------------------------------------

def _trimdayfile(name: str) -> str:
    """Porting di trimdayfile(x, words=c("escursion_","max_","min_",".nc"))."""
    for w in ["escursion_", "max_", "min_", ".nc"]:
        name = name.replace(w, "")
    return name


def graph_daily_onlybio(cfg: GraphicsConfig, render_fn: RenderFn | None = None) -> list[Path]:
    """
    Porting di graph_daily_onlybio.r: blocco orario (list_bio_only, come
    graph_daily_wkclimate ma con 19 variabili "solo bio", nessun rischio
    caldo) + triplo blocco giornaliero max/min/escursione (list_bio_day,
    replicato per ciascuno dei tre gruppi di file, come nel sorgente R —
    che itera separatamente su d_files_max, d_files_min, d_files_esc con
    la STESSA lista di 3 variabili ad ogni iterazione, producendo output
    ridondanti tra i tre blocchi tranne per il nome del giorno usato in
    trimdayfile — comportamento preservato fedelmente).
    """
    render_fn = render_fn or _default_render_fn(cfg)

    for f in cfg.scriptdir.glob("*.png"):
        f.unlink()

    inidate = _inidate(cfg)
    outputs = []

    # --- blocco orario (list_bio_only) ---
    hourly_files = sorted(cfg.datadir.glob(f"{cfg.pattern_model}*.nc"))
    variables_h = list_bio_only()

    for z, hourly_file in enumerate(hourly_files, start=1):
        stem = hourly_file.name.replace(cfg.pattern_model, "").replace(".nc", "")
        file_time = datetime.strptime(stem, "%Y%m%d%H")
        id_ = file_time.hour
        idhour = z - 1

        for spec in variables_h:
            codegs = grads_mod.codegs_for(spec.floatflag)
            raw_png = _render(cfg, render_fn, hourly_file, spec.var, id_, codegs)

            run_annotation = (
                f" Init. : {inidate} +12 UTC   Valid. CEST : "
                f"{file_time.strftime('%a, %d-%m-%Y %H')} t :+ {idhour} h"
            )
            out_path = cfg.graphdir / f"{spec.fileout}_{idhour}.png"
            imgpp.compose_biometeo_image(
                raw_png, cfg.logoimage, spec.textvar, run_annotation, out_path, title_size=21, subtitle_size=13
            )
            outputs.append(out_path)

    # --- blocco giornaliero (list_bio_day): max/min/escursione ---
    d_files_max = sorted(cfg.datadir.glob("max_*"))
    d_files_min = sorted(cfg.datadir.glob("min_*"))
    d_files_esc = sorted(cfg.datadir.glob("excursion_*"))
    variables_d = list_bio_day()  # [ta2m/max_t2m_ita, ta2m/min_t2m_ita, ta2m/exc_t2m_ita]

    for group in (d_files_max, d_files_min, d_files_esc):
        for i, ref_file in enumerate(group, start=1):
            day_label = _trimdayfile(ref_file.name)
            # come nel sorgente R: per ogni "ref_file" del gruppo, plotta
            # comunque tutti e tre i file (max[i], min[i], esc[i]) con lo
            # stesso day_label derivato da ref_file
            for spec, src_file in zip(
                variables_d,
                [d_files_max[i - 1], d_files_min[i - 1], d_files_esc[i - 1]],
            ):
                codegs = grads_mod.codegs_for(spec.floatflag)
                raw_png = _render(cfg, render_fn, src_file, spec.var, i, codegs)

                run_annotation = f"Init. : {inidate} Valid. :   {day_label}"
                out_path = cfg.graphdir / f"{spec.fileout}_{i}.png"
                imgpp.compose_biometeo_image(
                    raw_png, cfg.logoimage, spec.textvar, run_annotation, out_path, title_size=21, subtitle_size=13
                )
                outputs.append(out_path)

    return outputs


# ---------------------------------------------------------------------------
# plot_logo_wkclimate.r / plot_logo_wkclimate_onlybio.r
# ---------------------------------------------------------------------------

def _cleanup_na_files(graphdir: Path) -> None:
    for f in graphdir.glob("*NA*"):
        f.unlink(missing_ok=True)


def apply_watermarks_wkclimate(cfg: GraphicsConfig) -> None:
    """
    Porting di plot_logo_wkclimate.r: applica i due watermark
    (logoimagewk + discimagewk) a tutti i PNG che matchano rk*/wind*/
    t_dayice*/tmax_over*/TGG35*, esattamente i gruppi di pattern del
    sorgente R.
    """
    _cleanup_na_files(cfg.graphdir)

    patterns = ["rk*.png", "wind*.png", "t_dayice*.png", "tmax_over*.png", "TGG35*.png"]
    for pattern in patterns:
        for f in cfg.graphdir.glob(pattern):
            imgpp.add_watermarks(f, cfg.logoimagewk, cfg.discimagewk)

    _cleanup_na_files(cfg.graphdir)


def apply_watermarks_onlybio(cfg: GraphicsConfig) -> None:
    """
    Porting di plot_logo_wkclimate_onlybio.r: applica i due watermark a
    TUTTI i .png in graphdir (pattern piu' ampio del wkclimate).
    """
    _cleanup_na_files(cfg.graphdir)

    for f in cfg.graphdir.glob("*.png"):
        imgpp.add_watermarks(f, cfg.logoimagewk, cfg.discimagewk)

    _cleanup_na_files(cfg.graphdir)
