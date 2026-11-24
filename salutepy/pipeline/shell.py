"""
Equivalente Python di run_cmd() in salute_aux.r.

Esegue un comando di shell (tipicamente cdo, ma anche mv/rsync/ecc.) e
controlla il codice di uscita. Comportamento identico alla versione R:
per default interrompe l'esecuzione con un'eccezione se il comando fallisce
(fail fast), a meno che non si passi stop_on_error=False, nel qual caso
viene solo emesso un warning e l'esecuzione prosegue.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import warnings
from dataclasses import dataclass

logger = logging.getLogger("salutepy")


class CommandError(RuntimeError):
    """Sollevata quando un comando esterno fallisce e stop_on_error=True."""

    def __init__(self, cmd: str, returncode: int, stderr: str = ""):
        self.cmd = cmd
        self.returncode = returncode
        self.stderr = stderr
        msg = f"Comando fallito (rc={returncode}): {cmd}"
        if stderr:
            msg += f"\nstderr:\n{stderr}"
        super().__init__(msg)


@dataclass
class CmdResult:
    cmd: str
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def run_cmd(
    cmd: str | list[str],
    stop_on_error: bool = True,
    capture_output: bool = True,
    cwd: str | None = None,
) -> CmdResult:
    """
    Esegue un comando di shell e ne controlla il codice di uscita.

    Parameters
    ----------
    cmd : str | list[str]
        Comando da eseguire. Se str, viene interpretato con shell=True
        (necessario per pipe/wildcard cdo tipo "day_*.nc"); se list, viene
        eseguito senza shell (più sicuro quando non servono wildcard/pipe).
    stop_on_error : bool, default True
        Se True (default, come in salute_elab.r/meteosalute_op_dir_mol01.r),
        solleva CommandError in caso di fallimento. Se False (come nelle
        funzioni estrai_* usate dentro un loop su tanti file/giorni),
        emette un warning e ritorna comunque il CmdResult.
    capture_output : bool
        Se True cattura stdout/stderr invece di lasciarli ereditare al
        processo padre.
    cwd : str | None
        Working directory in cui eseguire il comando.

    Returns
    -------
    CmdResult
    """
    cmd_str = cmd if isinstance(cmd, str) else " ".join(shlex.quote(c) for c in cmd)
    logger.info(cmd_str)

    proc = subprocess.run(
        cmd,
        shell=isinstance(cmd, str),
        cwd=cwd,
        capture_output=capture_output,
        text=True,
    )

    result = CmdResult(
        cmd=cmd_str,
        returncode=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )

    if not result.ok:
        if stop_on_error:
            raise CommandError(cmd_str, result.returncode, result.stderr)
        else:
            warnings.warn(f"Comando fallito (rc={result.returncode}): {cmd_str}")

    return result
