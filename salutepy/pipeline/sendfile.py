"""
Porting di sendfile.R: copia in archivio, upload FTP (curl -T) delle
mappe prodotte e invio via SCP a un secondo server.

*** CREDENZIALI: NON HARDCODED (a differenza del sorgente R) ***
Il sorgente R originale contiene credenziali FTP/SCP IN CHIARO nel
codice (`userpwd <- 'databiometeo:AG23sd/sa'`, `sshpass -p 'MeteoSalute'`).
Qui sono parametri obbligatori di `SendFileConfig`, da valorizzare a
runtime (idealmente da variabili d'ambiente/secret manager, MAI committate
nel codice sorgente). L'uso di `sshpass` per passare la password SCP in
chiaro sulla command line resta comunque un anti-pattern di sicurezza
(la password è visibile a chiunque legga la process list, es. `ps aux`)
già presente nell'originale: idealmente andrebbe sostituito con
autenticazione a chiave SSH, ma qui è riprodotto fedelmente perché
cambiarlo richiederebbe modificare la configurazione del server remoto,
fuori dallo scope di un porting.

Upload via `curl` (subprocess) invece di riscrivere con `requests`/
`ftplib`, per coerenza con il resto della codebase (wrapping di tool
esterni via run_cmd) e per non rischiare differenze sottili di
comportamento del protocollo FTP (EPSV, modalità passiva) rispetto al
server già in produzione.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from natsort import natsorted

from .shell import run_cmd


@dataclass
class SendFileConfig:
    dir_products: Path
    graphdir: Path
    dir_out: Path
    dir_archive_current_images: Path

    ftp_url: str  # es. "ftp://<host>/httpdocs/Worklimate2/"
    ftp_url_utci: str  # es. "ftp://<host>/httpdocs/UTCI/"
    ftp_userpwd: str  # "user:password" — da env/secret manager, MAI hardcoded

    scp_host: str  # es. "salute@<ip>"
    scp_password: str  # da env/secret manager
    scp_remote_dir: str  # es. "/home/salute/wrf/graph/"

    var_risk: tuple[str, ...] = ("rkcla_int", "rkcla_nint", "rkshacla_int", "rkshacla_nint")


def _model_date(cfg: SendFileConfig) -> tuple[date, str]:
    """Equivalente di model_date/model_daterun (righe 11-13 R)."""
    candidates = sorted(cfg.dir_products.glob("*12_mol01gfs*.grb2"))
    if not candidates:
        raise FileNotFoundError(f"Nessun file '*12_mol01gfs*.grb2' trovato in {cfg.dir_products}.")
    name = candidates[0].name
    model_date = datetime.strptime(name[:8], "%Y%m%d").date()
    model_daterun = name[:10]
    return model_date, model_daterun


def _curl_upload(local_path: Path, remote_url: str, userpwd: str, stop_on_error: bool = False) -> None:
    """Equivalente di: system(paste("curl -T ", file, "-u", userpwd, url))."""
    run_cmd(f"curl -T {local_path} -u {userpwd} {remote_url}", stop_on_error=stop_on_error)


def _scp_send(local_path: Path, cfg: SendFileConfig, stop_on_error: bool = False) -> None:
    """Equivalente di: system(paste0("sshpass -p '<pwd>' scp <file> <host>:<remote_dir><file>"))."""
    remote = f"{cfg.scp_host}:{cfg.scp_remote_dir}{local_path.name}"
    run_cmd(f"sshpass -p '{cfg.scp_password}' scp {local_path} {remote}", stop_on_error=stop_on_error)


def run(cfg: SendFileConfig) -> None:
    """Orchestrazione completa, equivalente a tutto sendfile.R."""
    model_date, model_daterun = _model_date(cfg)

    # --- DATERUN.txt in graphdir e dir_out ---
    (cfg.graphdir / "DATERUN.txt").write_text(model_daterun)
    (cfg.dir_out / "DATERUN.txt").write_text(model_daterun)

    # --- liste file per pattern ---
    file_utci = sorted(cfg.graphdir.glob("*utci*"))
    file_wind = sorted(cfg.graphdir.glob("*wind*"))
    file_wbgt = sorted(cfg.graphdir.glob("*wbgt*"))
    file_dayice = sorted(cfg.graphdir.glob("*t_dayice*"))
    file_tmaxsup = sorted(cfg.graphdir.glob("*TGG35*"))

    # --- archiviazione con data nel nome (righe 46-50 R) ---
    cfg.dir_archive_current_images.mkdir(parents=True, exist_ok=True)
    for f in file_tmaxsup:
        archived = cfg.dir_archive_current_images / f.name.replace(".png", f"_{model_date}.png")
        archived.write_bytes(f.read_bytes())
    for f in file_dayice:
        archived = cfg.dir_archive_current_images / f.name.replace(".png", f"_{model_date}.png")
        archived.write_bytes(f.read_bytes())

    # NOTA: le righe 52-57 del sorgente R (file.copy(x, x, overwrite=TRUE))
    # copiano ogni file su se stesso nella stessa directory — un'operazione
    # di fatto senza effetto (probabile residuo di refactoring). Omessa
    # qui perché non ha alcun effetto osservabile, a differenza di ogni
    # altro passo di questo script.

    # --- upload FTP (righe 63-73 R) ---
    for f in file_tmaxsup:
        _curl_upload(f, cfg.ftp_url, cfg.ftp_userpwd)
    for f in file_dayice:
        _curl_upload(f, cfg.ftp_url, cfg.ftp_userpwd)
    for f in file_utci:
        _curl_upload(f, cfg.ftp_url_utci, cfg.ftp_userpwd)

    # --- upload file "latest" (righe 81-85 R) ---
    for f in sorted(cfg.dir_out.glob("*latest*")):
        _curl_upload(f, cfg.ftp_url, cfg.ftp_userpwd)

    # --- immagini del rischio: selezione con naturalsort + indici fissi (righe 92-111 R) ---
    nfiles_base = ["O8", "O12", "O16", "O20", "D8", "D12", "D16", "D20", "DD8", "DD12", "DD16", "DD20"]
    # NOTA: nel sorgente R il suffisso usa `model_date+1` (il giorno di
    # validita', non la data del run) — model_date qui e' la data del run,
    # quindi il giorno di validita' e' model_date+1
    from datetime import timedelta

    date_suffix = (model_date + timedelta(days=1)).strftime("%Y%m%d")
    nfiles = [f"{n}_{date_suffix}" for n in nfiles_base]

    idfig = [i + 2 for i in (7, 11, 15, 19)] + [i + 26 for i in (7, 11, 15, 19)] + [i + 50 for i in (7, 11, 15, 19)]

    def pick(pattern: str) -> list[Path]:
        matched = natsorted(cfg.graphdir.glob(f"*{pattern}*"), key=lambda p: p.name)
        return [matched[i] for i in idfig if i < len(matched)]

    filerksun_int = pick(cfg.var_risk[0])
    filerksun_nint = pick(cfg.var_risk[1])
    filerksha_int = pick(cfg.var_risk[2])
    filerksha_nint = pick(cfg.var_risk[3])

    filemapzonazero = filerksun_int + filerksun_nint + filerksha_int + filerksha_nint

    # equivalente di expand.grid(nfiles, var_risk): per ogni variabile, tutti gli nfiles
    name_files = [f"{n}_{v}.png" for v in cfg.var_risk for n in nfiles]

    for src, name in zip(filemapzonazero, name_files):
        dest = cfg.graphdir / name
        dest.write_bytes(src.read_bytes())

    for name in name_files[: len(filemapzonazero)]:
        _curl_upload(cfg.graphdir / name, cfg.ftp_url, cfg.ftp_userpwd)

    # --- data ultimo invio (righe 124-129 R) ---
    date_last_send = cfg.dir_out / "date_last_send.txt"
    date_last_send.write_text(date.today().strftime("%Y%m%d"))
    _curl_upload(date_last_send, cfg.ftp_url, cfg.ftp_userpwd)

    date_latest = cfg.dir_out / "date_latest.txt"
    if date_latest.exists():
        _curl_upload(date_latest, cfg.ftp_url, cfg.ftp_userpwd)

    # --- invio SCP al secondo server (righe 135-181 R) ---
    for group in (file_utci, file_wind, file_tmaxsup, file_dayice, file_wbgt, filemapzonazero):
        for f in group:
            _scp_send(f, cfg)

    # --- verifica contenuto server remoto (righe 188-196 R) ---
    result = run_cmd(f"curl -s -u {cfg.ftp_userpwd} {cfg.ftp_url}", stop_on_error=False)
    if result.ok:
        remote_files = [f for f in result.stdout.splitlines() if f.strip()]
        print("Nel server di zonazero ci sono...")
        for f in remote_files:
            print(f)
    print(datetime.now())
