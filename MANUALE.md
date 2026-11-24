# Manuale d'uso — salutepy

Porting Python della pipeline biometeorologica Meteosalute (CNR-IBE).
Questo manuale copre installazione, configurazione e utilizzo di ogni
componente del package. Per lo stato dettagliato del porting (cosa è
stato verificato, cosa resta da validare) vedi `README.md` nel package.

---

## Indice

1. [Installazione](#1-installazione)
2. [Dipendenze esterne non-Python](#2-dipendenze-esterne-non-python)
3. [Struttura del package](#3-struttura-del-package)
4. [Uso degli indici biometeorologici (standalone)](#4-uso-degli-indici-biometeorologici-standalone)
5. [Pipeline: produzione NetCDF orario (meteosalute)](#5-pipeline-produzione-netcdf-orario-meteosalute)
6. [Pipeline: download ICON-EU (alternativa a MOLOCH)](#6-pipeline-download-icon-eu-alternativa-a-moloch)
7. [Pipeline: estrazione matrici CSV (crea_matrici)](#7-pipeline-estrazione-matrici-csv-crea_matrici)
8. [Pipeline: grafica (mappe GrADS + post-processing)](#8-pipeline-grafica-mappe-grads--post-processing)
9. [Pipeline: invio file (sendfile)](#9-pipeline-invio-file-sendfile)
10. [Pipeline: estrazioni su stazioni/capoluoghi](#10-pipeline-estrazioni-su-stazionicapoluoghi)
11. [Orchestratore master (salute_procedura)](#11-orchestratore-master-salute_procedura)
12. [Configurazione tramite variabili d'ambiente](#12-configurazione-tramite-variabili-dambiente)
13. [Limiti noti da validare prima dell'uso in produzione](#13-limiti-noti-da-validare-prima-delluso-in-produzione)
14. [Risoluzione problemi comuni](#14-risoluzione-problemi-comuni)
15. [Come aggiungere una nuova variabile](#15-come-aggiungere-una-nuova-variabile)

---

## 1. Installazione

```bash
cd salutepy
pip install -e .
```

Questo installa tutte le dipendenze Python dichiarate in `pyproject.toml`:

```
numpy, scipy, pandas, xarray, netCDF4, cfgrib, rasterio, pyreadr,
pythermalcomfort, Pillow, natsort
```

Il flag `-e` (editable install) è consigliato durante lo sviluppo: le
modifiche ai file `.py` sono immediatamente attive senza reinstallare.

### Verifica dell'installazione

```python
from salutepy.indices import compat as R
from salutepy.pipeline import meteosalute, crea_matrici, graphics, sendfile
from salutepy.pipeline import estrai_dati_modelli, icon_download, salute_procedura
print("OK: salutepy installato correttamente")
```

---

## 2. Dipendenze esterne non-Python

Alcune parti della pipeline dipendono da binari esterni che **non sono
installabili via pip** e vanno presenti sul sistema:

| Binario | Usato da | Come verificare |
|---|---|---|
| `cdo` (Climate Data Operators) | `pipeline.shell`, `pipeline.daily_stats`, `pipeline.extraction`, `pipeline.icon_download` | `which cdo` |
| `grads` (GrADS) | `pipeline.grads`, `pipeline.graphics` | `which grads` |
| `curl` | `pipeline.icon_download`, `pipeline.sendfile` | `which curl` |
| `sshpass` + `scp` | `pipeline.sendfile` (invio al secondo server) | `which sshpass scp` |

Se uno di questi manca, le funzioni che ne dipendono sollevano un errore
esplicito (es. `GradsNotAvailable` per GrADS) invece di fallire in modo
silenzioso o criptico.

---

## 3. Struttura del package

```
salutepy/
  indices/                    # formule biometeorologiche (portate da HeatStress/rBiometeo)
    wbgt.py                     # wbgt.Liljegren completo
    thermal_indices.py          # Steadman, THI, heat_risk_level, mrt_globe, windchill, frostime
    comfort.py                  # PMV_ISO7730, clomin/clomax_7730
    productivity.py             # lost_productivity, WTL (Kjellstrom)
    psychrometrics.py           # es/PVS/deficitsat/pvap/dewp2hurs
    utci_wrapper.py             # utci/utci_class (via pythermalcomfort)
    sun_position.py             # posizione solare (sostituto di suntools::sunposition)
    misc.py                     # windchill_class_wk, metabolism, reducewind, oralegale_l
    compat.py                   # alias con i NOMI R esatti (uso consigliato)
  pipeline/
    shell.py                    # run_cmd(): esecuzione comandi esterni con controllo errori
    daily_stats.py               # operazioni cdo: max/min/escursione, soglie, split
    grib_reader.py                # lettura GRIB2 posizionale (rasterio)
    netcdf_writer.py               # scrittura NetCDF orario meteosalute
    extraction.py                   # estrai_acclimdays/estrai_param (cdo)
    meteosalute.py                   # produzione NetCDF orario da GRIB2
    crea_matrici.py                   # estrazione matrici punto-griglia -> CSV
    grads.py                            # invocazione GrADS esterno
    image_postprocess.py                 # crop/logo/annotazioni/watermark (Pillow)
    list_bio.py                           # liste variabili di plotting
    graphics.py                            # orchestratori grafici
    sendfile.py                              # invio FTP/SCP
    extract_samples.py                        # campionamento GRIB2 su stazioni
    estrai_dati_modelli.py                     # orchestratore estrazioni multi-set
    icon_download.py                            # download ICON-EU da DWD
    salute_procedura.py                          # ORCHESTRATORE MASTER
  gs_scripts/                 # script GrADS di rendering (biometeo_fig_class/float.gs)
  data/                       # file di riferimento bundled (stazioni, capoluoghi)
```

---

## 4. Uso degli indici biometeorologici (standalone)

Ogni indice è utilizzabile indipendentemente dalle pipeline, su array
numpy/scalari. Il modulo `compat` espone le funzioni con **gli stessi
nomi usati nel codice R originale**, comodo per confronto diretto.

```python
import numpy as np
from salutepy.indices import compat as R

# WBGT outdoor completo (modello di Liljegren)
from datetime import datetime
tas = np.array([32.0])       # temperatura aria, degC
dewp = np.array([20.0])      # temperatura di rugiada, degC
wind = np.array([2.0])       # vento, m/s
rad = np.array([700.0])      # radiazione solare, W/m2
dates = [datetime(2026, 7, 15, 13, 0, 0)]
lon = np.array([11.25])      # Firenze
lat = np.array([43.77])

res = R.wbgt_Liljegren(tas, dewp, wind, rad, dates, lon, lat)
print(res.data)   # WBGT
print(res.Tnwb)   # temperatura naturale di bulbo umido
print(res.Tg)     # temperatura del globo

# THI (Temperature Humidity Index)
thi = R.THI(t=30.0, rh=60.0)

# UTCI (via pythermalcomfort)
utci = R.utci(tdb=25.0, tr=25.0, v=1.0, rh=50.0)
utci_classe = R.utci_class(tdb=25.0, tr=25.0, v=1.0, rh=50.0)

# Windchill
wc = R.windchill(t=np.array([-5.0]), wind=np.array([5.0]))

# Perdita di produttività lavorativa da WBGT
lp = R.lost_productivity(wbgt=np.array([33.5]))
```

**Attenzione alle unità di misura**: `mrt_globe()` richiede il diametro
del globo in **metri** (default `0.05` = 50mm); nel porting del sorgente
R originale è stato corretto un bug di unità (150mm passati come `150`
invece di `0.15`) — vedi sezione 13.

---

## 5. Pipeline: produzione NetCDF orario (meteosalute)

Legge il GRIB2 MOLOCH del run, calcola tutti gli indici per ogni
timestep orario, scrive un NetCDF per ora (stesse 42 variabili del
sorgente R: `ta2m`, `wbgtsun`, `utci`, `windchill`, ecc.).

```python
from pathlib import Path
from salutepy.pipeline.meteosalute import MeteosaluteConfig, run

cfg = MeteosaluteConfig(
    model_dir=Path("/home/salute/data/output/lamma_models/workclimate2/"),
    work_dir=Path("/home/salute/data/output/products23/"),
    script_dir=Path("/home/salute/wrf/scripts23/"),
    fillrkmoloch_path=Path("/home/salute/wrf/scripts23/fillrkmoloch02.rds"),
    send_mail_nomodel_cmd="python3 /home/salute/wrf/scripts23/send_mail_nomodel.py",
)

output_files = run(cfg)   # lista dei NetCDF orari scritti
```

### Parametri di `MeteosaluteConfig`

| Campo | Tipo | Descrizione |
|---|---|---|
| `model_dir` | `Path` | Directory dove arriva il GRIB2 del run MOLOCH |
| `work_dir` | `Path` | Directory di lavoro (split orari, NetCDF di output) |
| `script_dir` | `Path` | Directory degli script/asset ausiliari |
| `fillrkmoloch_path` | `Path` | File `.rds` (letto via `pyreadr`) oppure file di **testo** (`.csv`/`.txt`, un valore per riga, letto via `np.loadtxt` — **non `.npy` binario**, a differenza di `landmask_path` in `crea_matrici.py` che invece usa `np.load()` su `.npy` binario) del vettore fallback "grads-safe" |
| `send_mail_nomodel_cmd` | `str \| None` | Comando per notificare l'assenza del modello (opzionale) |

### Uso avanzato: elaborare un singolo timestep

Se serve elaborare dati già in memoria (es. da `icon_download`, vedi
sezione 6) invece che da un file GRIB2, si usa direttamente
`process_timestep_from_grib()`:

```python
from salutepy.pipeline.meteosalute import process_timestep_from_grib

# `grib` è un oggetto MolochGribData (vedi grib_reader.py o icon_download.py)
outpath = process_timestep_from_grib(
    grib, i=1, dateboldate="20260812", fillrkmoloch=fillrkmoloch_array, work_dir=work_dir
)
```

`fillrkmoloch` deve avere lunghezza pari al numero di pixel della griglia
(358701 per la griglia mol02 standard) — se non hai il file `.rds`
originale, un array di `1.0` della lunghezza giusta funziona come
fallback neutro (nessuna correzione "grads-safe" applicata).

---

## 6. Pipeline: download ICON-EU (alternativa a MOLOCH)

**⚠️ Non testato contro il server DWD reale** (vedi sezione 13) — usare
con cautela e validare al primo download.

```python
from datetime import datetime, timezone
from pathlib import Path
from salutepy.pipeline.icon_download import IconDownloadConfig, fetch_icon_eu_timestep
from salutepy.pipeline.meteosalute import process_timestep_from_grib

cfg = IconDownloadConfig(
    work_dir=Path("/tmp/icon_work"),
    target_grid_file=Path("lav_meteosalute/grids/mol01gfs.grb2.topo_landmask.interp_grid"),
)

run_dt = datetime(2026, 8, 12, 12, tzinfo=timezone.utc)  # run delle 12 UTC

# scarica un singolo timestep (ora di forecast +3h) e lo assembla
grib = fetch_icon_eu_timestep(run_dt, step_h=3, cfg=cfg, prev_step_h=2)

# passa direttamente alla pipeline di calcolo indici, IDENTICA a MOLOCH
import numpy as np
fillrkmoloch = np.ones(len(grib.variables["T_fhg"]))  # o il vettore .rds vero
outpath = process_timestep_from_grib(grib, i=1, dateboldate="20260812",
                                       fillrkmoloch=fillrkmoloch, work_dir=cfg.work_dir)
```

### Scaricare un intero run (loop sugli step orari)

```python
outputs = []
prev_step = 0
for step_h in range(1, 49):  # run 00: tipicamente 48h di forecast orario
    grib = fetch_icon_eu_timestep(run_dt, step_h=step_h, cfg=cfg, prev_step_h=prev_step)
    out = process_timestep_from_grib(grib, i=step_h, dateboldate=run_dt.strftime("%Y%m%d"),
                                       fillrkmoloch=fillrkmoloch, work_dir=cfg.work_dir)
    outputs.append(out)
    prev_step = step_h
```

### Mapping variabili (names_meteos → parametro DWD)

| Variabile pipeline | Parametro ICON-EU | Note |
|---|---|---|
| `T_fhg` | `T_2M` | Kelvin → Celsius |
| `TD_fhg` | `TD_2M` | Kelvin → Celsius |
| `u_fhg`, `v_fhg` | `U_10M`, `V_10M` | nessuna conversione |
| `gust_sfc` | `VMAX_10M` | nessuna conversione |
| `pmsl` | `PMSL` | Pa |
| `N_atm` | `CLCT` | % (0-100) — **da verificare** contro convenzione MOLOCH |
| `dswrf_sfc` | `ASWDIR_S` + `ASWDIFD_S` | deaccumulati (media-dall'inizio → media-sull'intervallo) e sommati |
| `lapr` | *nessun equivalente* | riempito a `NaN` |

---

## 7. Pipeline: estrazione matrici CSV (crea_matrici)

Estrae matrici punto-griglia (temperatura massima a 5 giorni,
acclimatazione a 10 giorni, WBGT/UTCI/windchill orari a 3 giorni) e
scrive tutti i CSV di output, incluso il rischio "poste".

```python
from pathlib import Path
from salutepy.pipeline.crea_matrici import CreaMatriciConfig, run

cfg = CreaMatriciConfig(
    work_dir=Path("/home/salute/data/output/products23/"),
    dir_archive=Path("/DISCO4TB/archivi/archivi_moloch_dati/"),
    dir_out=Path("/home/salute/data/output/ZONAWKCLIMATE2/"),
    dir_archive_current_forecast=Path("/DISCO4TB/archivi/current_forecast/"),
    poste_risk_dir=Path("/home/salute/data/POSTE_RISK/"),
    landmask_path=Path("/home/salute/wrf/scripts23/landmask_data.rds"),
)

outputs = run(cfg)  # dict {nome_prodotto: percorso_csv}
print(outputs.keys())
# dict_keys(['tmax', 'acclim', 'wbgtsun', 'wbgtsha', 'utci', 'utcisha',
#            'hrisksha', 'hrisksun', 'windchill', 'rkcla_int', 'rkcla_nint',
#            'mat_date_acclim'])
```

`landmask_path` accetta sia `.rds` (letto via `pyreadr`) sia `.npy`
**binario** (`np.load()`). **Attenzione**: non è lo stesso formato non-`.rds`
accettato da `fillrkmoloch_path` in `meteosalute.py` (che usa `np.loadtxt()`
su file di testo) — i due loader non sono intercambiabili, vedi sezione 5.

---

## 8. Pipeline: grafica (mappe GrADS + post-processing)

Richiede **GrADS installato e sul PATH** (vedi sezione 2) e gli asset
`.gs` completi (vedi sezione 13 per quelli mancanti).

```python
from pathlib import Path
from salutepy.pipeline.graphics import (
    GraphicsConfig, graph_daily_wkclimate, graph_daily_max,
    graph_daily_t_dayice, graph_daily_onlybio,
    apply_watermarks_wkclimate, apply_watermarks_onlybio,
)

cfg = GraphicsConfig(
    scriptdir=Path("/home/salute/wrf/scripts23/"),
    datadir=Path("/home/salute/data/output/products23/"),
    dir_products=Path("/home/salute/data/output/products23/"),
    graphdir=Path("/home/salute/wrf/graph23/"),
    logoimage=Path("/home/salute/wrf/logo/LogoLammaCNR_bis.png"),
    logoimagewk=Path("/home/salute/wrf/logo/Logo_wkclimate_img.png"),
    discimagewk=Path("/home/salute/wrf/logo/disclaimer_img.png"),
)

graph_daily_wkclimate(cfg)     # mappe orarie rischio caldo (4 variabili x N ore)
graph_daily_max(cfg)           # mappe soglia Tmax>35, 5 giorni
graph_daily_t_dayice(cfg)      # mappe giorni di gelo, 5 giorni
apply_watermarks_wkclimate(cfg)  # applica logo watermark + disclaimer

graph_daily_onlybio(cfg)       # 19 variabili orarie "solo bio" + blocco giornaliero max/min/escursione
apply_watermarks_onlybio(cfg)
```

### Testare la logica senza GrADS installato

Ogni funzione accetta un parametro `render_fn` iniettabile, utile per
verificare naming/loop/post-processing senza dover avere GrADS:

```python
from PIL import Image

def fake_render(file, var, id_, codegs):
    # simula l'output GrADS: un PNG 700x800 al posto della vera mappa
    Image.new("RGBA", (700, 800), (200, 200, 200, 255)).save(cfg.scriptdir / f"{var}_{id_}.png")

graph_daily_wkclimate(cfg, render_fn=fake_render)
```

### Verificare gli asset GrADS mancanti prima di lanciare il rendering

```python
from salutepy.pipeline.grads import check_grads_assets
from pathlib import Path

problemi = check_grads_assets(var="utcicla", floatflag="cla", legend_dir=Path("lav_meteosalute/legend"))
if problemi:
    print("Asset mancanti:", problemi)
```

---

## 9. Pipeline: invio file (sendfile)

**Le credenziali NON sono hardcoded** (a differenza del sorgente R
originale): vanno passate esplicitamente, idealmente lette da variabili
d'ambiente (vedi sezione 12).

```python
import os
from pathlib import Path
from salutepy.pipeline.sendfile import SendFileConfig, run

cfg = SendFileConfig(
    dir_products=Path("/home/salute/data/output/products23/"),
    graphdir=Path("/home/salute/wrf/graph23/"),
    dir_out=Path("/home/salute/data/output/ZONAWKCLIMATE2/"),
    dir_archive_current_images=Path("/DISCO4TB/archivi/current_images/"),
    ftp_url="ftp://<host>/httpdocs/Worklimate2/",
    ftp_url_utci="ftp://<host>/httpdocs/UTCI/",
    ftp_userpwd=os.environ["MS_FTP_USERPWD"],       # "utente:password"
    scp_host="salute@<ip>",
    scp_password=os.environ["MS_SCP_PASSWORD"],
    scp_remote_dir="/home/salute/wrf/graph/",
)

run(cfg)
```

---

## 10. Pipeline: estrazioni su stazioni/capoluoghi

Campiona il GRIB2 nei punti-stazione (per validazione) e, opzionalmente,
anche sui 116 capoluoghi di provincia italiani.

```python
from pathlib import Path
from salutepy.pipeline.estrai_dati_modelli import EstraiDatiConfig, with_capoluoghi, run

cfg = EstraiDatiConfig(
    model_dir=Path("/home/salute/data/output/lamma_models/workclimate2/"),
    dir_archive_extract=Path("/DISCO4TB/archivi/archivi_moloch_dati/extract_archive/"),
    # stations_csv non specificato: usa il default bundled (40 stazioni di verifica)
)

cfg = with_capoluoghi(cfg)  # aggiunge l'estrazione sui 116 capoluoghi

outputs = run(cfg)
# per ogni file GRIB2: <file>.csv (stazioni verifica) + <file>_capoluoghi.csv
```

### Usare un file stazioni personalizzato

```python
from salutepy.pipeline.estrai_dati_modelli import EstraiDatiConfig, StationSet

cfg = EstraiDatiConfig(
    model_dir=Path("..."),
    dir_archive_extract=Path("..."),
    stations_csv=Path("mie_stazioni.csv"),  # colonne: qualunque nome riconosciuto per lon/lat/nome(+elev opzionale)
    extra_station_sets=[
        StationSet("regione_toscana", Path("stazioni_toscana.csv")),
    ],
)
```

Il file stazioni può avere colonne in **qualsiasi ordine**: il mapping è
per nome (case-insensitive), con alias riconosciuti:
- longitudine: `lon`, `lng`, `longitude`, `x`
- latitudine: `lat`, `latitude`, `y`
- nome: `nome`, `name`, `location`, `stazione`, `station`, `comune`
- elevazione (opzionale): `elev`, `elevation`, `quota`, `z`, `alt`, `altitude`

---

## 11. Orchestratore master (salute_procedura)

Esegue l'intera sequenza operativa (equivalente a `salute_procedura.R`,
lo script lanciato da cron): produzione NetCDF → immagini → creazione
matrici → invio file → estrazioni → mail di conferma.

```python
from pathlib import Path
from salutepy.pipeline.salute_procedura import SaluteProceduraConfig, run
from salutepy.pipeline.meteosalute import MeteosaluteConfig
from salutepy.pipeline.crea_matrici import CreaMatriciConfig
from salutepy.pipeline.graphics import GraphicsConfig
from salutepy.pipeline.sendfile import SendFileConfig
from salutepy.pipeline.estrai_dati_modelli import EstraiDatiConfig

cfg = SaluteProceduraConfig(
    meteosalute_cfg=MeteosaluteConfig(...),
    crea_matrici_cfg=CreaMatriciConfig(...),
    graphics_cfg=GraphicsConfig(...),
    sendfile_cfg=SendFileConfig(...),
    estrai_dati_cfg=EstraiDatiConfig(...),
    script_dir=Path("/home/salute/wrf/scripts23/"),
    model_dir=Path("/home/salute/data/output/lamma_models/workclimate2/"),
    okfile_path=Path("/home/salute/data/output/products23/okfileprocedura.file"),
)

run(cfg)
```

Tutti e 12 gli step della sequenza originale sono Python nativo. Il
parametro `fallback_to_r=True/False` resta disponibile per compatibilità
futura (se si aggiungono nuovi step non ancora portati): con `False` gli
step non portati vengono saltati con un warning invece di bloccare
l'esecuzione.

### Comportamento in caso di modello mancante

Se `okfile_path` non esiste dopo la produzione NetCDF, l'orchestratore
invia una mail di notifica e solleva `RuntimeError` (comportamento
identico al sorgente R: `stop_on_missing_model=True` di default).

---

## 12. Configurazione tramite variabili d'ambiente

Le credenziali (FTP, SCP) non sono mai hardcoded nel codice. Pattern
consigliato:

```python
import os
from salutepy.pipeline.sendfile import SendFileConfig

cfg = SendFileConfig(
    ...,
    ftp_userpwd=os.environ["MS_FTP_USERPWD"],
    scp_password=os.environ["MS_SCP_PASSWORD"],
)
```

```bash
export MS_FTP_USERPWD="utente:password"
export MS_SCP_PASSWORD="password_scp"
python3 mio_script_lancio.py
```

---

## 13. Limiti noti da validare prima dell'uso in produzione

| Area | Limite | Dove |
|---|---|---|
| **GrADS** | Script `biometeo_fig_class.gs`/`biometeo_fig_float.gs` recuperati e inclusi, ma mancano 4 palette (`t_gt34`, `utcicla`, `utciclasha`, `windcla`) e l'intera cartella `legend/strings/` (9 script) | `pipeline/grads.py` |
| **ICON-EU** | Download mai testato contro il server DWD reale (sandbox senza accesso di rete a `opendata.dwd.de`); convenzione `CLCT` (%/frazione) da verificare | `pipeline/icon_download.py` |
| **Lettura GRIB2 per nome-variabile** | Usata in `extract_samples.py` (diversa dalla lettura posizionale di `grib_reader.py`), non validata su file reale | `pipeline/extract_samples.py` |
| **File stazioni originale** | `wk_staz_obs_61.rds` (SpatVector `terra` serializzato) mai caricato; sostituito da un file bundled con 40 stazioni (probabilmente diverso dall'originale a 61 stazioni) | `pipeline/extract_samples.py` |
| **UTCI** | Non validato numericamente contro l'output R (usa `pythermalcomfort`, stesso standard ufficiale ma non confrontato direttamente) | `indices/utci_wrapper.py` |
| **Font grafici** | "Palatino"/"Sans" del sorgente R sostituiti con DejaVu Serif/Sans | `pipeline/image_postprocess.py` |
| **sendfile.R** | Uso di `sshpass` per SCP (password in chiaro nella process list) ereditato dall'originale, non corretto perché richiederebbe modificare l'autenticazione del server remoto | `pipeline/sendfile.py` |

Due bug reali del sorgente R **sono stati corretti** (sia in R che nel
porting Python): `mrt_globe()` riceveva un diametro in millimetri invece
che in metri, e `wbgt.Liljegren()` non passava mai `hour=TRUE` (angolo
zenitale sempre a mezzogiorno UTC fisso). Un terzo bug corretto:
`windchill_latest.csv` non veniva mai scritto in `crea_matrici.r`.

---

## 14. Risoluzione problemi comuni

**`GradsNotAvailable: Binario 'grads' non trovato sul PATH`**
→ GrADS non è installato o non è sul PATH. Verificare con `which grads`.

**`CommandError: Comando fallito (rc=127)`**
→ Il comando esterno chiamato (cdo, curl, sshpass) non è installato.
Il codice `127` indica "comando non trovato" in bash.

**`ValueError: Colonna 'nome' non trovata nel file stazioni`**
→ Il file CSV stazioni non ha una colonna riconoscibile come nome/lon/lat.
Vedi gli alias accettati nella sezione 10, oppure rinominare le colonne
prima di passare il file.

**`FileNotFoundError: Nessun file '*12_mol01gfs*.grb2' trovato`**
→ Il file GRIB2 del run non è ancora arrivato in `model_dir`, oppure il
pattern di ricerca non corrisponde al nome file reale.

**Warning `Comando fallito (rc=...)` durante `estrai_acclimdays`/`estrai_param`**
→ Comportamento atteso se il file NetCDF di un giorno manca
dall'archivio: la funzione ritorna un vettore di valori mancanti (`-999`)
invece di interrompere l'intera estrazione.

---

## 15. Come aggiungere una nuova variabile

Aggiungere una variabile alla pipeline richiede toccare **fino a 4 punti
diversi**, a seconda di dove deve comparire (NetCDF orario, CSV
estratti, mappe). Non tutti sono sempre necessari: se la variabile serve
solo nel NetCDF, bastano i punti 1-2.

| # | Punto | File | Necessario se... |
|---|---|---|---|
| 1 | Calcolo del valore | `pipeline/meteosalute.py`, dentro `process_timestep_from_grib()` | sempre |
| 2 | Dichiarazione nel NetCDF | `pipeline/netcdf_writer.py`, lista `VARIABLES` | la variabile deve finire nel NetCDF orario |
| 3 | Estrazione CSV (matrice punto-griglia) | `pipeline/crea_matrici.py`, dentro `run()` | serve anche in `crea_matrici` (CSV a 3/10 giorni) |
| 4 | Mappa GrADS | `pipeline/list_bio.py` (lista `VarSpec`) + asset `.gs` corrispondenti | serve anche una mappa |

### Esempio completo: aggiungere l'escursione di rugiada (`t - td`)

Variabile di esempio: `dewdep` (dewpoint depression, °C) = temperatura
aria meno temperatura di rugiada. Formula banale, usata qui solo come
traccia — per un indice più complesso il procedimento è identico, cambia
solo il calcolo al punto 1.

#### 1. Calcolo — `pipeline/meteosalute.py`

Dentro `process_timestep_from_grib()`, dopo che `t_fhg`/`td_fhg` sono
già disponibili (sono già letti per altri calcoli), aggiungere:

```python
dewdep_var = t_fhg - td_fhg
```

e aggiungerla al dizionario `data` passato a `write_meteosalute_nc()`
(cercare il blocco `data = { "ta2m": t_fhg, ... }` verso la fine della
funzione):

```python
data = {
    "ta2m": t_fhg,
    ...
    "dewdep": dewdep_var,   # <-- nuova riga
    "thi": thi_index,
    "lapr": lapr,
}
```

#### 2. Dichiarazione NetCDF — `pipeline/netcdf_writer.py`

Aggiungere una riga alla lista `VARIABLES` (nome, unità — usate per
creare la variabile e il suo attributo `units` nel file NetCDF):

```python
VARIABLES = [
    ("ta2m", "degC"),
    ...
    ("dewdep", "degC"),   # <-- nuova riga
    ("thi", "degC"),
    ("lapr", "dimensionless"),
]
```

Se un nome in `VARIABLES` non ha una chiave corrispondente in `data`
(punto 1), `write_meteosalute_nc()` scrive quella variabile come
fillvalue ovunque e stampa un warning — utile per accorgersi subito di
un nome scritto in modo diverso nei due punti.

#### 3. Estrazione CSV — `pipeline/crea_matrici.py` (opzionale)

Se la variabile deve comparire anche tra le matrici CSV a 3 giorni
(`wbgtsun_latest.csv`, ecc.), aggiungere una chiamata a `extract_hourly()`
vicino alle altre (cercare il blocco `wbgtsun_df = extract_hourly("wbgtsun")`):

```python
dewdep_df = extract_hourly("dewdep")
```

e includerla nel blocco di scrittura CSV subito sotto:

```python
for name, df in [
    ("wbgtsun", wbgtsun_df),
    ...
    ("dewdep", dewdep_df),   # <-- nuova riga
]:
    out_path = cfg.dir_out / f"{name}_latest.csv"
    df.to_csv(out_path, index=False)
    outputs[name] = out_path
```

`extract_hourly()` funziona per estrazione via `cdo -selvar,<par>` dal
NetCDF archiviato: perché funzioni, la variabile deve già esistere nel
NetCDF orario (punto 2) **e** essere stata archiviata (i file
`day_*.nc` prodotti da `salute_elab`/`daily_stats.py`).

#### 4. Mappa — `pipeline/list_bio.py` + asset GrADS (opzionale)

Aggiungere una voce in una delle liste `list_bio_*()` (tipicamente
`list_bio_only()` per variabili orarie "solo bio"):

```python
def list_bio_only() -> list[VarSpec]:
    names = [..., "dewdep"]
    outs = [..., "dewdep_ita"]
    flags = [..., "float"]   # "float" = scala continua, "cla" = classi discrete
    texts = [..., "Escursione di rugiada"]
    return [VarSpec(n, f, o, t) for n, f, o, t in zip(names, flags, outs, texts)]
```

**Se `floatflag="cla"`** (variabile a classi discrete, non a scala
continua), servono anche gli asset GrADS corrispondenti
(`legend/palette/palette_dewdep.gs`, `legend/code/cbarn_dewdep.gs`,
`legend/strings/dewdep_string.gs`) — verificarne la presenza con
`grads.check_grads_assets()` (sezione 8) prima di lanciare il rendering,
altrimenti GrADS si fermerà con un errore su quella variabile.
**Se `floatflag="float"`** (come in questo esempio) non serve nessun
asset aggiuntivo: `biometeo_fig_float.gs` non richiama palette/string.

### Aggiungere una formula del tutto nuova (non derivabile da variabili già lette)

Se il calcolo richiede una formula non ancora presente in `indices/`
(es. un nuovo indice biometeorologico), il primo passo è scriverla in
un modulo di `indices/` appropriato (o crearne uno nuovo), seguendo lo
stile degli altri: input array-like, output array-like, docstring che
cita la fonte della formula. Poi esporla in `indices/compat.py` se serve
un alias con nome R, e infine usarla nel punto 1 sopra
(`process_timestep_from_grib()`) come le altre chiamate a `R.<funzione>`.

**Non inventare formule scientifiche a memoria** — se l'indice viene da
un pacchetto R/letteratura specifica, serve il sorgente esatto (come
fatto per HeatStress/rBiometeo in questa conversazione) per evitare di
introdurre un errore scientifico silenzioso in un prodotto operativo.
