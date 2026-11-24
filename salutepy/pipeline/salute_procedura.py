"""
Porting di salute_procedura.R: l'orchestratore MASTER dell'intera
pipeline Meteosalute (quello lanciato da cron), non salute_elab.r — che
gestisce solo il primo blocco (archiviazione cdo + trigger di
meteosalute_op_dir_mol01.r).

Sequenza esatta del file R originale:
    1.  salute_elab.r            -> check_file4gfs.R + meteosalute_op_dir_mol01.r
    2.  controllo okfileprocedura.file (altrimenti mail + stop)
    3.  rimozione okfileprocedura.file
    4.  graph_daily_wkclimate_mol01gfs.r
    5.  graph_daily_max.r          (mappe soglia >35)
    6.  graph_daily_t_dayice.r     (giorni di gelo)
    7.  plot_logo_wkclimate.r
    8.  crea_matrici.r
    9.  sendfile.R
    10. graph_daily_onlybio.r
    11. plot_logo_wkclimate_onlybio.r
    12. estrai_dati_modelli.r
    13. send_mail_okmodel.py
    14. pulizia model_dir (rimuove file "00_*", "*gfs*", "*ecm*")

STATO DEL PORTING per singolo step — vedi STEP_STATUS più sotto:
- TUTTI e 12 gli step della sequenza originale sono ora portati
  nativamente in Python (salutepy.pipeline.meteosalute, .crea_matrici,
  .graphics, .sendfile, .estrai_dati_modelli). Non restano più fallback
  a script R legacy per default, ma il meccanismo fallback_to_r resta
  disponibile per compatibilità/debug se in futuro si aggiungono step.
- Gli step di grafica (4,5,6,7,10,11) dipendono comunque da GrADS come
  binario esterno — vedi salutepy.pipeline.grads per i dettagli. Gli
  script biometeo_fig_class.gs/biometeo_fig_float.gs SONO stati
  recuperati (erano in codice.zip) e inclusi in salutepy/gs_scripts/,
  ma alcuni asset da essi richiamati restano mancanti: 4 palette (per
  t_gt34, utcicla, utciclasha, windcla) e l'intera cartella
  legend/strings/ (9 script _string.gs per le variabili "cla") — vedi
  grads.py per l'elenco esatto verificato.
- Lo step "estrazioni" (estrai_dati_modelli.r) dipende da un file
  stazioni (wk_staz_obs_61.rds, un SpatVector terra serializzato) MAI
  caricato in questa sessione, che richiede pre-conversione in CSV via R
  prima dell'uso — vedi salutepy.pipeline.extract_samples. La lettura
  GRIB2 per nome-variabile (necessaria qui, diversa dalla lettura
  posizionale di meteosalute.py) non è stata validata su un file reale.
- Il passo "mail_ok" (send_mail_okmodel.py) è già uno script Python
  nella codebase originale: richiamato via subprocess come nel sorgente R.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .crea_matrici import CreaMatriciConfig
from .crea_matrici import run as run_crea_matrici
from .estrai_dati_modelli import EstraiDatiConfig
from .estrai_dati_modelli import run as run_estrai_dati_modelli
from .graphics import GraphicsConfig
from .graphics import apply_watermarks_onlybio, apply_watermarks_wkclimate
from .graphics import graph_daily_max, graph_daily_onlybio, graph_daily_t_dayice, graph_daily_wkclimate
from .meteosalute import MeteosaluteConfig
from .meteosalute import run as run_meteosalute
from .sendfile import SendFileConfig
from .sendfile import run as run_sendfile
from .shell import CommandError, run_cmd

logger = logging.getLogger("salutepy.salute_procedura")


class StepKind(Enum):
    PYTHON = "python"  # pipeline gia' portata, chiamata nativamente
    R_SCRIPT = "r_script"  # script R legacy, richiamato via Rscript
    PY_SCRIPT = "py_script"  # script Python legacy gia' presente nella codebase (es. send_mail_*.py)


@dataclass
class Step:
    name: str
    kind: StepKind
    label: str  # messaggio equivalente al message() del sorgente R
    script_name: str | None = None  # nome file per kind R_SCRIPT/PY_SCRIPT
    python_callable: object = None  # callable per kind PYTHON


@dataclass
class SaluteProceduraConfig:
    meteosalute_cfg: MeteosaluteConfig
    crea_matrici_cfg: CreaMatriciConfig
    graphics_cfg: GraphicsConfig
    sendfile_cfg: SendFileConfig
    estrai_dati_cfg: EstraiDatiConfig
    script_dir: Path  # directory degli script R/py legacy non ancora portati
    model_dir: Path  # per la pulizia finale (righe 93-99 del sorgente R)
    okfile_path: Path  # products23/okfileprocedura.file
    fallback_to_r: bool = True  # se True chiama gli step non portati via Rscript; se False li salta con un warning
    stop_on_missing_model: bool = True  # replica righe 26-27: stop se okfileprocedura.file manca


# Elenco degli step non ancora portati in Python nativo, con relativo
# fallback allo script R legacy — aggiornare questa lista via via che si
# completa il porting (spostando lo step corrispondente da R_SCRIPT/
# PY_SCRIPT a PYTHON con un python_callable reale).
STEP_STATUS = {
    "graph_daily_wkclimate_mol01gfs.r": "portato (pipeline.graphics.graph_daily_wkclimate)",
    "graph_daily_max.r": "portato (pipeline.graphics.graph_daily_max)",
    "graph_daily_t_dayice.r": "portato (pipeline.graphics.graph_daily_t_dayice)",
    "plot_logo_wkclimate.r": "portato (pipeline.graphics.apply_watermarks_wkclimate)",
    "sendfile.R": "portato (pipeline.sendfile.run)",
    "graph_daily_onlybio.r": "portato (pipeline.graphics.graph_daily_onlybio)",
    "plot_logo_wkclimate_onlybio.r": "portato (pipeline.graphics.apply_watermarks_onlybio)",
    "estrai_dati_modelli.r": "portato (pipeline.estrai_dati_modelli.run) - vedi note in extract_samples.py",
    "send_mail_okmodel.py": "script Python gia' nativo nella codebase originale (non riscritto, solo richiamato)",
}


def _build_steps(cfg: SaluteProceduraConfig) -> list[Step]:
    return [
        Step(
            "produzione_netcdf",
            StepKind.PYTHON,
            "Producing Moloch netcdf files",
            python_callable=lambda: run_meteosalute(cfg.meteosalute_cfg),
        ),
        Step(
            "immagini_workclimate",
            StepKind.PYTHON,
            "Producing images for WORKCLIMATE (1/3: wkclimate)",
            python_callable=lambda: graph_daily_wkclimate(cfg.graphics_cfg),
        ),
        Step(
            "immagini_max35",
            StepKind.PYTHON,
            "Producing images for WORKCLIMATE (2/3: mappe >35)",
            python_callable=lambda: graph_daily_max(cfg.graphics_cfg),
        ),
        Step(
            "immagini_gelo",
            StepKind.PYTHON,
            "Producing images for WORKCLIMATE (3/3: giorni di gelo)",
            python_callable=lambda: graph_daily_t_dayice(cfg.graphics_cfg),
        ),
        Step(
            "logo_watermark",
            StepKind.PYTHON,
            "Producing logo watermarked images",
            python_callable=lambda: apply_watermarks_wkclimate(cfg.graphics_cfg),
        ),
        Step(
            "crea_matrici",
            StepKind.PYTHON,
            "Create matrix!",
            python_callable=lambda: run_crea_matrici(cfg.crea_matrici_cfg),
        ),
        Step(
            "sendfile",
            StepKind.PYTHON,
            "Send files",
            python_callable=lambda: run_sendfile(cfg.sendfile_cfg),
        ),
        Step(
            "lamma_images",
            StepKind.PYTHON,
            "Lamma Images (1/2: grafici onlybio)",
            python_callable=lambda: graph_daily_onlybio(cfg.graphics_cfg),
        ),
        Step(
            "lamma_images_logo",
            StepKind.PYTHON,
            "Lamma Images (2/2: logo onlybio)",
            python_callable=lambda: apply_watermarks_onlybio(cfg.graphics_cfg),
        ),
        Step(
            "estrazioni",
            StepKind.PYTHON,
            "Estrazioni",
            python_callable=lambda: run_estrai_dati_modelli(cfg.estrai_dati_cfg),
        ),
        Step(
            "mail_ok",
            StepKind.PY_SCRIPT,
            "Invio mail di conferma modello",
            script_name="send_mail_okmodel.py",
        ),
    ]


def _run_step(step: Step, cfg: SaluteProceduraConfig) -> None:
    logger.info("#" * 110)
    logger.info(step.label)

    if step.kind == StepKind.PYTHON:
        step.python_callable()
        return

    script_path = cfg.script_dir / step.script_name

    if step.kind == StepKind.PY_SCRIPT:
        run_cmd(f"python3 {script_path}", stop_on_error=False)
        return

    # StepKind.R_SCRIPT: step non ancora portato in Python nativo
    if not cfg.fallback_to_r:
        logger.warning(
            "Step '%s' (%s) saltato: non ancora portato in Python e fallback_to_r=False. "
            "Stato: %s",
            step.name,
            step.script_name,
            STEP_STATUS.get(step.script_name, "sconosciuto"),
        )
        return

    if not script_path.exists():
        raise FileNotFoundError(
            f"Step '{step.name}': script legacy R non trovato in {script_path}. "
            f"Questo step ({step.script_name}) non e' ancora stato portato in Python "
            "in questa sessione — serve il sorgente originale disponibile in script_dir "
            "per eseguirlo via fallback, oppure va portato nativamente."
        )

    run_cmd(f"Rscript {script_path}", stop_on_error=False)


def run(cfg: SaluteProceduraConfig) -> None:
    """
    Esegue l'intera procedura, equivalente a salute_procedura.R.

    Solleva CommandError/RuntimeError se un passo bloccante fallisce
    (stesso comportamento del sorgente R su system() con esito diverso
    da 0 seguito da stop(), dove presente esplicitamente).
    """
    steps = _build_steps(cfg)

    # step 1: salute_elab.r (produzione netcdf) + meteosalute_op_dir_mol01.r
    _run_step(steps[0], cfg)

    # righe 26-27 R: controllo presenza okfileprocedura.file
    if not cfg.okfile_path.exists():
        run_cmd("python3 " + str(cfg.script_dir / "send_mail_nomodel.py"), stop_on_error=False)
        if cfg.stop_on_missing_model:
            raise RuntimeError("Mancano le previsioni del modello (okfileprocedura.file non trovato).")

    cfg.okfile_path.unlink(missing_ok=True)

    # step successivi (immagini, logo, crea_matrici, sendfile, lamma, estrazioni, mail)
    for step in steps[1:]:
        _run_step(step, cfg)

    # righe 93-99 R: pulizia model_dir
    logger.info("#" * 110)
    logger.info("Rimuovi tutti i files da lamma_models")
    for pattern in ["00_*", "*gfs*", "*ecm*"]:
        for f in cfg.model_dir.glob(pattern):
            f.unlink(missing_ok=True)

    logger.info("End")
