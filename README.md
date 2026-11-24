# salutepy

Porting Python della pipeline biometeorologica Meteosalute (CNR-IBE),
originariamente in R + cdo (`salute_procedura.R`, `salute_elab.r`,
`meteosalute_op_dir_mol01.r`, `crea_matrici.r`, `crea_archive_moloch_grid.R`).

## Calcolo parallelo su griglie (GRIB, NetCDF, GeoTIFF) — `salutepy.raster` (v0.2)

Modulo per calcolare gli indici di `biometeo.js` (rBiometeo 0.4) su dati raster, in parallelo.

```python
from salutepy import raster

ds = raster.open_raster("2026112312_mol01gfs.grb2")   # GRIB/GRIB2, NetCDF o GeoTIFF
std = raster.standardize(ds)                          # t (degC), rh (%), td, wind (m/s), pair (hPa), solar (W/m2), tmrt, tg
out = raster.compute(std, ["utci", "pet", "humidex", "net", "thi"], workers=8)
raster.write_netcdf(out, "indici.nc")                 # oppure raster.write_geotiff(out.utci, "utci.tif")
```

Da riga di comando:

```bash
python -m salutepy.raster run.grb2 -i utci,pet,humidex -o indici.nc -w 8
python -m salutepy.raster era5.nc -i all --deaccumulate -o era5_indici.nc
python -m salutepy.raster tair.tif --var rh=rh.tif --var tmrt=mrt.tif -i utci,pet --time 2026-07-15T12:00 -o out.tif
```

**Indici** (`raster.list_indices()`): `utci`, `pet` (Höppe/VDI 3787), `wbgt` (Liljegren, da
`indices.wbgt`), `heatindex`, `humidex`, `net`, `thi` (Thom/Giles), `windchill`, `mrt_globe`,
`mrt_solar`, `sunelev`. Per UTCI, PET, heat index, humidex, NET, THI e wind chill viene scritta
anche la classe `<indice>_class` (int8, 0 = non calcolabile) con le etichette italiane e inglesi
negli attributi (`labels_it`, `labels_en`).

**Variabili**: `standardize` riconosce i nomi usuali (ecCodes/cfgrib `t2m`, `d2m`, `u10`, `v10`,
`sp`, `msl`, `ssrd`...; CF/CMIP `tas`, `hurs`, `sfcWind`, `rsds`...; ICON/GFS `T_2M`,
`TMP_2maboveground`...), converte le unità (K→°C, Pa→hPa, frazione→%) e ricava le variabili
mancanti (umidità dal punto di rugiada e viceversa, vento da u/v). `mapping={"t": "nome"}` forza
un nome. La radiazione cumulata in J/m2 viene de-cumulata solo con `deaccumulate=True`.
Se manca `tmrt`, UTCI e PET usano la temperatura media radiante da radiazione solare
(`mrt_solar`, con l'elevazione solare calcolata per ogni pixel e ora) o, senza radiazione, la
temperatura dell'aria (condizioni all'ombra), con un avviso.

**Parallelismo**: la griglia (tempi × pixel) è divisa in blocchi elaborati da un pool di processi
(`concurrent.futures`); la PET è compilata con numba (opzionale, `pip install salutepy[fast]`) e
calcolata in thread paralleli. Sotto `min_block` punti gli indici vettoriali sono calcolati in un
solo processo, perché il costo di avvio dei processi supera il guadagno.

**Implementazione e verifica**: gli indici sono porting vettoriali NumPy di `biometeo.js`
(`indices/biometeo_np.py`; il polinomio UTCI è generato automaticamente dal sorgente JS) e la PET
è in `indices/pet.py`. `tests/test_biometeo_np.py` confronta ogni funzione con `biometeo.js`
eseguito con Node.js su migliaia di input casuali: risultati identici (scarto massimo ≤ 1e-7,
classi identiche). `tests/test_raster.py` prova lettura/scrittura GRIB2, NetCDF e GeoTIFF sintetici
e verifica che il calcolo parallelo dia gli stessi valori di quello seriale.

**Funzioni di HeatStress** (pacchetto R di A. Casanueva, v1.0.8.1): oltre al WBGT di Liljegren già
presente, `indices/heatstress_np.py` porta in NumPy `swbgt`, `wbt.Stull`, `apparentTemp`,
`discomInd`, `effectiveTemp`, `hi`, `humidex`, `tashurs2dewp`, `tashurs2vap.pres`, `dewp2hurs`, e
`indices/wbgt_fast.py` i WBGT di Liljegren e di Bernard compilati con numba (circa 200 volte più
veloci della versione scipy, che resta in `indices/wbgt.py`). Le minimizzazioni usano una traduzione
di `Brent_fmin`, la routine di `stats::optimize` di R. `tests/test_heatstress.py` confronta tutto con
il pacchetto R HeatStress (variabile d'ambiente `HEATSTRESS_RLIB` se non è nella libreria R di
default): scarto massimo 1e-9 per le formule dirette e 1e-6 °C per i due WBGT. Nel modulo raster
sono gli indici `wbgt` (Liljegren, con l'ora di ogni passo: `hour=True`), `wbgt_bernard`, `swbgt`,
`wbt_stull`, `apparent_temp`, `discomfort_index`, `effective_temp`, `heatindex_hs` (convertito in
°C), `humidex_hs`.

Due particolarità di HeatStress riprodotte fedelmente: `calZenith` non usa la longitudine
(l'angolo orario è calcolato come se il punto fosse a longitudine 0; in Italia circa 45 minuti di
tempo solare di differenza: `wbgt_liljegren(..., solar_time_correction=True)` la corregge) e
`hi` calcola la correzione per bassa umidità come `sqrt(17 - |T - 95| / 17)` invece di
`sqrt((17 - |T - 95|) / 17)` della NWS.

**Licenza**: `indices/pet.py` deriva dal codice PETBER del DWD (PALM-4U, GPL-3) e
`data/js/biometeo.js` è sotto GPL-3: il pacchetto è distribuito sotto GPL-3.0 o successiva.

**Nota**: i moduli preesistenti `indices/thermal_indices.py` e `indices/comfort.py`, usati dalla
pipeline Meteosalute, sono porting della versione 0.3 di `biometeo.js` e non sono stati modificati:
`thermal_indices.thi` usa ancora la vecchia formula del THI e `mrt_globe` la sola convezione
forzata. Per i valori corretti usare `indices.biometeo_np`.

## Stato del porting

### Orchestrazione (cdo, shell, file management)

- `pipeline.shell.run_cmd()` — equivalente di `run_cmd()` in `salute_aux.r`.
- `pipeline.daily_stats` — operazioni cdo di `salute_elab.r` (max/min/escursione, mappe soglia, split per giorno).

### Indici biometeorologici — portati dai sorgenti R/JS esatti

Sorgenti usati: `HeatStress-master/R/*.R` e `rBiometeo-master/{R,inst/js}/*`
(le funzioni R di rBiometeo sono thin-wrapper verso JavaScript eseguito via
V8: le formule reali sono state portate da `inst/js/biometeo.js`, non dai
wrapper R). Le formule WTL/lost_productivity vengono da `wbgt_aux.r`, già
presente nel codice originale della pipeline (non da un pacchetto esterno).

| Funzione R | Modulo Python |
|---|---|
| `wbgt.Liljegren` | `indices.wbgt.wbgt_liljegren` |
| `mrt_globe` | `indices.thermal_indices.mrt_globe` |
| `steadman_outdoor_sun/shade`, `steadman_indoor`, `steadman_class` | `indices.thermal_indices` |
| `THI`, `heat_risk_level`, `windchill`, `frostime` | `indices.thermal_indices` |
| `PMV_ISO7730`, `clomin_7730`, `clomax_7730` | `indices.comfort` |
| `lost_productivity`, `mean_wbgtTI_WTL`, `mean_wbgtWTL`, `mean_airTI_WTL`, `mean_airTWTL` | `indices.productivity` |
| `deficitsat`, `pvap`, `dewp2hurs` | `indices.psychrometrics` |
| `utci`, `utci_class` | `indices.utci_wrapper` (wrapper su `pythermalcomfort`) |
| `sunposition` (pacchetto `suntools`) | `indices.sun_position.sunposition_norefrac` — **sostituzione documentata**, vedi sotto |
| `windchill_class_wk`, `metabolism`, `reducewind`, `oralegale_l` | `indices.misc` |

`salutepy.indices.compat` espone le funzioni HeatStress/rBiometeo con gli
stessi nomi usati in R (es. `wbgt_Liljegren`, `THI`, `PMV_ISO7730`).

### Grafica (GrADS + post-processing immagini)

- **`pipeline.grads`** — porting di `run_gradsbio()` (identica in tutti
  i `plot_images_*.r`): chiama il binario esterno `grads` via subprocess,
  con controllo esplicito di disponibilità (`GradsNotAvailable`).
  **GrADS resta una dipendenza esterna non portabile** (non è un pacchetto
  Python), esattamente come nel sorgente R originale.

  **`biometeo_fig_class.gs`/`biometeo_fig_float.gs` sono stati recuperati**
  (erano presenti in `codice.zip`, mancati nella prima ricerca) e sono
  inclusi in `salutepy/gs_scripts/`. Non sono stati riscritti — restano
  script GrADS (linguaggio proprietario) invocati via subprocess.

  **Asset ancora mancanti, verificati puntualmente** (funzione
  `check_grads_assets()` in `grads.py`): per le 9 variabili "classificate"
  (`rkcla_int/nint`, `rkshacla_int/nint`, `t_gt34`, `t_dayice`, `utcicla`,
  `utciclasha`, `windcla`) mancano:
  - 4 palette (`legend/palette/palette_{t_gt34,utcicla,utciclasha,windcla}.gs`)
  - l'intera cartella `legend/strings/` (9 script `<var>_string.gs`)

  Tutti gli 9 `cbarn_<var>.gs` sono invece presenti. Senza gli asset
  mancanti, GrADS si fermerà con un errore per le variabili elencate.

- **`pipeline.image_postprocess`** — porting con Pillow delle chiamate
  `magick` (`image_border`, `image_crop`, `image_composite`,
  `image_annotate`) e `composite` di ImageMagick CLI (`add_watermarks`),
  con la stessa geometria esatta del sorgente R (crop 580×700, bordo
  0×70, offset logo +260+10, posizioni testo). I font "Palatino"/"Sans"
  del sorgente R sono sostituiti con DejaVu Serif/Sans (disponibili di
  default): resa visiva del testo potenzialmente leggermente diversa,
  font path configurabili.
- **`pipeline.list_bio`** — porting di tutti i `list_bio_*.r` (liste
  variabili/testi/nomi file, stesse stringhe esatte del sorgente R).
- **`pipeline.graphics`** — porting completo degli orchestratori:
  `graph_daily_wkclimate` (+ `plot_images_wkclimate_mol01gfs.r`),
  `graph_daily_max`, `graph_daily_t_dayice` (+ `plot_images_daymax.r`/
  `plot_images_t_dayice.r`), `graph_daily_onlybio` (+
  `plot_images_onlybio.r`/`plot_images_day.r`), `apply_watermarks_wkclimate`/
  `apply_watermarks_onlybio` (+ `plot_logo_wkclimate*.r`). Ogni funzione
  accetta un `render_fn` iniettabile per testare la logica di
  orchestrazione/naming/post-processing senza GrADS — testato end-to-end
  con un renderer fittizio: naming dei file, crop/logo/testo/watermark
  verificati visivamente su un caso sintetico.

### Invio file e estrazioni per validazione

- **`pipeline.sendfile`** — porting completo di `sendfile.R`: copia in
  archivio con data nel nome, upload FTP (`curl -T`) e invio SCP a un
  secondo server. **Credenziali NON hardcoded** (a differenza del
  sorgente R, che le conteneva in chiaro): sono parametri obbligatori di
  `SendFileConfig`, da valorizzare da variabili d'ambiente/secret
  manager. L'uso di `sshpass` per SCP resta un anti-pattern di sicurezza
  ereditato dall'originale (password visibile nella process list) — non
  corretto in questo porting perché richiederebbe modificare
  l'autenticazione del server remoto, fuori scope. Testato end-to-end
  con `curl`/`scp` mockati.
- **`pipeline.extract_samples`** e **`pipeline.estrai_dati_modelli`** —
  porting di `extract_samples_simple.R`/`estrai_dati_modelli.r`:
  campionamento del GRIB2 nei punti-stazione per validazione/confronto
  con osservazioni.
  1. **File stazioni**: il sorgente R legge `wk_staz_obs_61.rds` (un
     `terra::SpatVector` serializzato, illeggibile da `pyreadr` — va
     pre-convertito in CSV con R, comando nel docstring di
     `extract_samples.py`). Non essendo mai stato caricato in questa
     sessione, `EstraiDatiConfig.stations_csv` usa ora come **default**
     `salutepy/data/staz_verify_italia.csv` — normalizzato da
     `staz_verify_italy.txt` (caricato dall'utente), **40 stazioni**
     contro le 61 implicite nel nome del file originale: probabilmente
     non lo stesso elenco usato in produzione, ma un sostituto
     ragionevole. Contiene anche due stazioni entrambe chiamate
     "Perugia" con coordinate diverse (verificare se intenzionale). La
     lettura colonne è per NOME (case-insensitive, con alias comuni:
     `location`/`nome`/`name`/`comune`, `quota`/`elev`/`elevation`,
     ecc.), non posizionale — altrimenti un file con ordine colonne
     diverso da `lon,lat,nome,elev` verrebbe mappato silenziosamente in
     modo sbagliato (bug reale trovato e corretto durante il porting,
     scoperto proprio caricando un file con ordine colonne diverso).
  2. **Estensione: estrazione anche sui 116 capoluoghi di provincia
     italiani** (`capoluoghi_province_italiani.csv`, caricato
     dall'utente e incluso in `salutepy/data/`, senza colonna
     elevazione — non essenziale, riempita con NaN). Non presente nel
     sorgente R originale: `EstraiDatiConfig` supporta ora
     `extra_station_sets: list[StationSet]`, ciascuno con etichetta
     propria usata come suffisso nel nome file di output (es.
     `<file>_capoluoghi.csv` accanto a `<file>.csv` per il set
     principale). Attivabile con
     `cfg = with_capoluoghi(EstraiDatiConfig(...))`. **Opt-in**, non
     abilitato di default nell'orchestratore master per non cambiare
     silenziosamente il comportamento esistente.
  3. La lettura GRIB2 qui è per NOME variabile (via tag GDAL
     `GRIB_COMMENT`/`GRIB_VALID_TIME`), diversa dalla lettura posizionale
     di `grib_reader.py` — **non validata su un file reale** in questa
     sessione.
  4. L'output è scritto come CSV invece di `.rds` (pyreadr non supporta
     la scrittura di `.rds` senza un bridge R/rpy2).

### Download ICON-EU (DWD) — alternativa/complemento a MOLOCH

- **`pipeline.icon_download`** — scarica ICON-EU dal server open data DWD
  e lo assembla nello stesso formato `MolochGribData` usato da MOLOCH,
  pronto per `process_timestep_from_grib()` (vedi refactoring sotto) —
  quindi l'intera pipeline di calcolo indici (WBGT, UTCI, Steadman, ecc.)
  si riusa senza modifiche, sia che i dati vengano da MOLOCH sia da ICON.

  **⚠️ Non testato contro il server DWD reale**: questo sandbox non ha
  accesso di rete a `opendata.dwd.de` (whitelist ristretta a pypi/github/
  npm). Il modulo è scritto sulla base delle convenzioni DWD verificate
  via ricerca web (URL, nomi parametro, semantica dei campi di
  radiazione), con test automatici solo sulla logica di deaccumulazione
  (verificata analiticamente con casi noti) e sull'assemblaggio dati
  (con download/remap mockati) — **va validato al primo download reale**.

  **Mapping variabili** (`names_meteos` → parametro DWD ICON-EU):
  T_fhg→T_2M, TD_fhg→TD_2M, u_fhg→U_10M, v_fhg→V_10M, gust_sfc→VMAX_10M,
  pmsl→PMSL, N_atm→CLCT, dswrf_sfc→ASWDIR_S+ASWDIFD_S (deaccumulati e
  sommati). **`lapr` non ha equivalente in ICON-EU single-level**:
  riempito a NaN (non usato per calcolare alcun indice, solo riportato
  nell'output NetCDF).

  **Deaccumulazione radiazione**: `ASWDIR_S`/`ASWDIFD_S` in ICON-EU sono
  medie cumulate dall'inizio del run (non accumuli semplici) — verificato
  su documentazione DWD/GribStream. La formula di deaccumulazione a
  media-sull'intervallo (`deaccumulate_mean_from_start()`) richiede la
  coppia di timestep consecutivi ed è testata con casi analitici (campo
  costante, rampa).

  **Riuso della griglia MOLOCH esistente**: il remap (`cdo remapbil`)
  usa come target `mol01gfs.grb2.topo_landmask.interp_grid` (già presente
  in `lav_meteosalute.zip`), garantendo che l'output ICON sia sulla
  stessa griglia italiana 0.02° di MOLOCH.

  **Altre incertezze da verificare al primo uso reale**: `CLCT` è in %
  (0-100) — MOLOCH potrebbe usare una convenzione a frazione (0-1), da
  confrontare; finestra di disponibilità degli step (oraria fino a 78h,
  poi trioraria, secondo la documentazione DWD) non verificata
  empiricamente.

- **Refactoring in `meteosalute.py`**: `process_timestep()` è stato
  scomposto in un thin wrapper (legge il file GRIB2 MOLOCH) +
  `process_timestep_from_grib()` (tutta la logica di calcolo indici,
  indipendente dalla fonte dati). Nessun cambio di comportamento per il
  percorso MOLOCH esistente (verificato con lo stesso test end-to-end
  usato in precedenza); `process_timestep_from_grib()` è il punto
  d'ingresso riusato da `icon_download.py`.

### Pipeline complete portate

- **`pipeline.salute_procedura`** — porting dell'orchestratore MASTER
  `salute_procedura.R` (quello lanciato da cron, non `salute_elab.r` —
  che gestisce solo il primo blocco). **Tutti e 12 gli step della
  sequenza originale sono ora Python nativo** (non restano più fallback
  a script R legacy per default): produzione NetCDF, 3 blocchi di
  immagini, watermark logo, creazione matrici, invio file, immagini
  Lamma, estrazioni, mail di conferma. Il meccanismo `fallback_to_r`
  resta disponibile per compatibilità/debug futuro. Riproduce fedelmente
  il controllo `okfileprocedura.file` (stop + mail se il modello manca)
  e la pulizia finale di `model_dir`. Testato end-to-end con tutte le
  sotto-pipeline mockate/con renderer fittizio: nessuno step saltato.
- **`pipeline.meteosalute`** — porting completo di `meteosalute_op_dir_mol01.r`:
  localizzazione/staging del GRIB2 del run, split cdo, lettura di ogni
  timestep, calcolo di tutti gli indici, scrittura NetCDF orario.
  Testato end-to-end con dati sintetici (griglia mockata, senza un vero
  file GRIB2 — vedi limitazioni sotto).
- **`pipeline.crea_matrici`** — porting completo di `crea_matrici.r`:
  gap-filling archivio, estrazione matrici punto-griglia (tmax,
  acclimatazione, WBGT/UTCI/windchill orari), calcolo "poste risk",
  scrittura di tutti i CSV. Testato end-to-end con un archivio NetCDF
  sintetico (12 pixel).
- **`pipeline.extraction`** — `estrai_acclimdays`/`estrai_param`, versioni
  Python delle funzioni consolidate in `salute_aux.r`, usate da `crea_matrici`.
- **`pipeline.grib_reader`** — lettura GRIB2 posizionale via `rasterio`.
- **`pipeline.netcdf_writer`** — scrittura del NetCDF orario "meteosalute"
  (stesse 42 variabili/unità/dimensioni del file R).

### Bug trovati e risolti (nel sorgente R e nel porting Python)

1. **`mrt_globe` — bug di unità di misura.** La chiamata in
   `meteosalute_op_dir_mol01.r` passava `150` invece di `0.15` come
   diametro del globo (la formula si aspetta metri, non millimetri) — un
   fattore ~27x sul termine di scambio convettivo. **Corretto** in R e
   riflesso nel default Python (`diam_globe=0.05`, invariato — era già
   corretto; il bug era solo nella chiamata).

2. **`wbgt.Liljegren` — angolo zenitale sempre a mezzogiorno UTC.** La
   chiamata non passava mai `hour=`, quindi l'angolo zenitale solare
   veniva calcolato sempre alle 12 UTC fisse anche per il WBGT orario.
   **Corretto** in R (`hour=TRUE` esplicito) e nel default Python
   (`hour=True`).

3. **`crea_matrici.r` — `windchill_latest.csv` non veniva mai scritto.**
   `windchill_df` veniva calcolato e rinominato ma il `write.csv`
   successivo scriveva per errore `rkcla_int_df` (che veniva poi
   scritto una seconda volta, correttamente, sovrascrivendo la prima
   copia mal-nominata). Il prodotto `windchill_latest.csv`, nonostante
   fosse calcolato, non veniva mai prodotto. **Corretto** in R e nel
   porting Python (`pipeline/crea_matrici.py`).

### Sostituzioni documentate (sorgente non disponibile)

- **`sunposition` (pacchetto `suntools`)**: sorgente non disponibile in
  questa sessione. Sostituito con `sunposition_norefrac`, già presente
  in `salute_aux.r` (stessa codebase, non un pacchetto esterno) con
  identica firma e struttura di output. Possibile lieve differenza
  (tipicamente <1°) vicino all'orizzonte se `suntools::sunposition`
  applica una correzione di rifrazione atmosferica che
  `sunposition_norefrac` non applica per costruzione (come indica il nome).

### Limitazioni note / da validare prima dell'uso operativo

1. **Lettura GRIB2 non testata su un file reale.** `pipeline.grib_reader`
   usa `rasterio` (stesso binding GDAL di `terra::rast()` in R, scelto
   deliberatamente al posto di `cfgrib` per preservare l'ordinamento
   posizionale delle bande su cui si basa `names_meteos`). Il codice è
   stato scritto per essere logicamente equivalente a
   `terra::rast()+as.data.frame()`, ma va validato al primo uso su un
   file `.grb2` reale prodotto da
   `cdo -f grb2 -splitsel,1 -seltimestep,11/83`.

2. **`utci`/`utci_class` non ancora validati numericamente contro R.**
   Il polinomio UTCI non è stato ritrascritto a mano (alto rischio di
   errore su ~210 coefficienti): si usa `pythermalcomfort`, che implementa
   lo stesso standard ufficiale. Prima dell'uso operativo, confrontare
   numericamente `rBiometeo::utci()` in R e `utci()` qui su un set di
   input di test.

3. **3 formule di pressione di vapore saturo distinte** in
   `psychrometrics.py` — non intercambiabili: `esat_buck_hpa` (Buck 1981,
   solo per `wbgt_liljegren`), `es_goff_hpa` (per `utci`/`pvap`),
   `pvs_iapws_hpa` (IAPWS-IF97, solo per `deficitsat`).

4. **Parallelismo non riprodotto.** Il codice R usa
   `future_lapply`/`plan(multisession)` per parallelizzare i calcoli
   per-pixel. Qui sono loop Python sequenziali (già vettorizzati dove la
   formula lo permette, es. `sunposition_norefrac`). Se le prestazioni
   non bastano su griglie da 358701 pixel, si può parallelizzare con
   `joblib`/`multiprocessing` senza cambiare l'interfaccia delle funzioni.

5. **Test eseguiti sono smoke test, non validazione numerica contro R.**
   Tutti i moduli sono stati testati con dati sintetici per verificare
   che l'esecuzione end-to-end non produca errori e che i risultati
   siano fisicamente plausibili (range di valori, casi limite noti). Non
   sostituiscono un confronto diretto con l'output R su dati reali di
   produzione.

## Struttura

```
salutepy/
  pipeline/
    shell.py            # run_cmd() ed esecuzione comandi esterni
    daily_stats.py       # operazioni cdo: max/min/escursione, soglie, split
    grib_reader.py        # lettura GRIB2 posizionale (rasterio)
    netcdf_writer.py       # scrittura NetCDF orario meteosalute
    extraction.py            # estrai_acclimdays/estrai_param (cdo)
    meteosalute.py             # porting completo meteosalute_op_dir_mol01.r
    crea_matrici.py              # porting completo crea_matrici.r
    salute_procedura.py            # orchestratore master (salute_procedura.R)
    grads.py                          # invocazione GrADS esterno (run_gradsbio) + check_grads_assets
    image_postprocess.py               # crop/logo/annotazioni/watermark (Pillow, ex-magick)
    list_bio.py                          # liste variabili di plotting (list_bio_*.r)
    graphics.py                            # orchestratori grafici (graph_daily_*.r, plot_logo_*.r)
    sendfile.py                              # invio FTP/SCP (sendfile.R)
    extract_samples.py                         # campionamento GRIB2 su stazioni (extract_samples_simple.R)
    estrai_dati_modelli.py                       # orchestratore estrazioni (estrai_dati_modelli.r)
  gs_scripts/
    biometeo_fig_class.gs                          # recuperati da codice.zip, non riscritti (GrADS)
    biometeo_fig_float.gs
  indices/
    wbgt.py               # wbgt.Liljegren completo (HeatStress)
    thermal_indices.py    # Steadman, THI, heat_risk_level, mrt_globe, windchill, frostime
    comfort.py             # PMV_ISO7730, clomin/clomax_7730
    productivity.py        # lost_productivity, WTL (Kjellstrom)
    psychrometrics.py      # es/PVS/deficitsat/pvap/dewp2hurs
    utci_wrapper.py         # utci/utci_class (via pythermalcomfort)
    sun_position.py          # sunposition_norefrac (sostituto di suntools::sunposition)
    misc.py                    # windchill_class_wk, metabolism, reducewind, oralegale_l
    compat.py                   # alias con i nomi R esatti
tests/
```

## Dipendenze

```
numpy, scipy, pandas, xarray, netCDF4, cfgrib, rasterio, pyreadr, pythermalcomfort
```

`pyreadr` legge i file `.rds` (es. `fillrkmoloch02.rds`, `landmask_data.rds`)
direttamente in Python; in alternativa si possono pre-convertire una volta
con R (`saveRDS`→`write.csv`/`.npy`) per evitare la dipendenza.

## WBGT sole/ombra da GRIB MOLOCH (`pipeline/wbgt_grib.py`)

Sostituisce il blocco wbgt.Liljegren di `meteosalute_op_dir_mol01.r` leggendo direttamente il GRIB2 (senza split cdo):

```python
from salutepy.pipeline.wbgt_grib import wbgt_from_grib
ds = wbgt_from_grib("meteosalute/2026112312_mol01gfs.grb2", timesteps=(11, 83), workers=8)
ds[["wbgtsun", "wbgtsha", "tglob"]].to_netcdf("wbgt.nc")
```

CLI: `python -m salutepy.pipeline.wbgt_grib file.grb2 -o wbgt.nc -w 8 --steps 11,83 --time-mode valid|r`

- `time_mode="valid"` (default): ora valida reale di ogni step, zenit all'ora (hour=True).
- `time_mode="r"`: riproduce esattamente lo script R (data run 00 UTC + (i-1)h, hour=FALSE); scarto da HeatStress < 1e-6 °C.
- Output: `wbgtsun`, `wbgtsha` (0.7·Tnwb + 0.3·T), `tglob`, `tnwb`, `ta2m`, `wind2m`.

### Verifica contro un NetCDF prodotto in R (`pipeline/verifica_wbgt.py`)

```
python -m salutepy.pipeline.verifica_wbgt 2026112312_mol01gfs.grb2 mol01gfs12_2026112412.nc            # come R
python -m salutepy.pipeline.verifica_wbgt 2026112312_mol01gfs.grb2 mol01gfs12_2026112412.nc --time-mode valid
```

Dal nome del NetCDF ricava l'indice i dello script R e il passo GRIB (run 12 UTC + 10 + i - 1 ore), ricalcola
wbgtsun/wbgtsha/tglob e stampa bias, MAE, RMSE, massimo scarto e correlazione; `-o file.nc` salva campi e differenze.
