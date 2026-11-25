# salutepy

**Biometeorological indices and the Meteosalute processing chain in Python**
**Indici biometeorologici e catena di elaborazione Meteosalute in Python**

[English](#english) · [Italiano](#italiano)

---

## English

### What it is

`salutepy` computes human biometeorological and occupational heat-stress indices on gridded
weather data (GRIB, NetCDF, GeoTIFF) and on plain arrays. It is the Python version of the
**Meteosalute** forecasting chain of the Institute of BioEconomy of the Italian National Research
Council (CNR-IBE, Florence), which was written in R on top of the JavaScript library `biometeo.js`
(R package `rBiometeo`) and of the R package `HeatStress`.

The library:

- reads model output directly (e.g. MOLOCH, ICON, ERA5 GRIB2 files) and recognises the usual
  variable names and units (K → °C, Pa → hPa, u/v → wind speed, dew point ↔ relative humidity);
- computes the indices on the whole grid at once, with compiled (numba) kernels for the iterative
  models (WBGT of Liljegren, PET, PMV, clothing insulation) and parallel threads;
- reproduces the hourly NetCDF products of Meteosalute (42 variables, same file names, time axis
  and fill values as the operational R script) and can also work from GeoTIFF rasters with no date
  or time information.

### Indices

| Group | Indices |
|---|---|
| Thermal comfort | UTCI and classes; PET (Höppe / VDI 3787); PMV (ISO 7730) and minimum/maximum clothing insulation; Steadman apparent temperature (sun, shade, indoor) and classes; Normal Effective Temperature (NET); Thom discomfort index (THI); heat index; humidex; wind chill and frostbite time |
| Heat stress at work | WBGT in the sun and in the shade (Liljegren et al. 2008, as in HeatStress); WBGT of Bernard; simplified WBGT; heat-risk classes; lost productivity and work-time loss (TI-WTL, TWTL; Kjellstrom et al. 2018) |
| Radiation | mean radiant temperature from globe temperature (ISO 7726) or from global radiation and sun elevation; globe temperature; sun position |
| Psychrometrics | vapour pressure, saturation deficit, dew point, wet-bulb temperature (Stull) |

Every index carries its reference in the code (`salutepy.raster.list_indices()` prints them).

### Installation

Python ≥ 3.10.

```bash
pip install "salutepy[fast] @ git+https://github.com/alfcrisci/salutepy"
```

The `fast` extra installs numba, needed for good performance with WBGT, PET, PMV and clothing
insulation. Reading GRIB files uses `cfgrib`/`ecCodes`, GeoTIFF uses `rasterio`.

### Quick start

**1. Indices on any grid (GRIB, NetCDF, GeoTIFF)**

```python
from salutepy import raster

ds = raster.open_raster("2026112312_mol01gfs.grb2")
std = raster.standardize(ds)            # t (°C), rh (%), wind (m/s), solar (W/m2), ...
out = raster.compute(std, ["utci", "pet", "wbgt", "humidex"], workers=8)
raster.write_netcdf(out, "indices.nc")
```

```bash
python -m salutepy.raster run.grb2 -i utci,pet,humidex -o indices.nc -w 8
```

**2. Indices from GeoTIFF rasters without date and time**

Temperature, relative humidity, wind and global radiation, plus the sun elevation (a raster or a
single value) instead of the date and time:

```bash
python -m salutepy.raster.indici_tif --t t2m.tif --rh rh.tif --wind ws10.tif --rad glob.tif --elev 55 -o indices
```

It writes one GeoTIFF per index (UTCI, PET, WBGT sun/shade, mean radiant and globe temperature and
the six work-time-loss indices) on the input grid; multi-band inputs (e.g. hours) give multi-band
outputs.

**3. The Meteosalute hourly NetCDF files from a MOLOCH GRIB2 run**

```bash
python -m salutepy.pipeline.meteosalute_nc 2026112312_mol01gfs.grb2 -d out                     # as the operational R script
python -m salutepy.pipeline.meteosalute_nc 2026112312_mol01gfs.grb2 -d out --all --time-mode valid  # all hours, true valid time
```

`--time-mode r` (default) reproduces the operational script exactly, including its time
convention; `--time-mode valid` uses the true UTC valid time for the sun position.

### Validation

Each port was checked against its reference implementation:

- `biometeo.js` (Node.js) and `rBiometeo` for the comfort indices, PMV and clothing insulation;
- the R package `HeatStress` for WBGT (Liljegren and Bernard) and the other HeatStress indices
  (agreement better than 10⁻⁶ °C);
- an operational Meteosalute NetCDF produced in R: all 42 variables are reproduced within the
  2-decimal rounding of the original, apart from a few dozen pixels (out of 358 701) at class or
  clothing-step boundaries.

The test suite (`pytest`) contains these comparisons; the R and Node.js ones run when those
programs are available.

### Notes

- The mean radiant temperature from globe temperature uses a 0.15 m globe by default, as the
  operational Meteosalute products; the estimate from global radiation (`--tmrt solar`) can be
  very different with a low sun. Choose and document the option you use.
- The work-time-loss formulas are the polynomial fits used by Meteosalute (`wbgt_aux.r`).
- Technical notes on the port, known differences and fixed bugs (in Italian):
  [docs/PORTING_it.md](docs/PORTING_it.md); user manual: [MANUALE.md](MANUALE.md).

### Credits

- `biometeo.js` / `rBiometeo`: Alfonso Crisci, Marco Morabito, Alessandro Messeri (CNR-IBE), GPL-3.
- WBGT of Liljegren and the other HeatStress indices: port of the R package `HeatStress`
  (A. Casanueva et al.), GPL-3.
- PET: algorithm of Höppe (1999) / VDI 3787, after the PALM-4U implementation (GPL-3).

### How to cite

See [CITATION.cff](CITATION.cff) or the Zenodo record of the release you use.

### License

GNU General Public License v3.0 or later — see [LICENSE](LICENSE).

---

## Italiano

### Che cos'è

`salutepy` calcola indici biometeorologici e di stress termico lavorativo su dati meteorologici in
griglia (GRIB, NetCDF, GeoTIFF) e su semplici array. È la versione Python della catena di
previsione **Meteosalute** dell'Istituto per la BioEconomia del Consiglio Nazionale delle Ricerche
(CNR-IBE, Firenze), scritta in origine in R sopra la libreria JavaScript `biometeo.js` (pacchetto R
`rBiometeo`) e il pacchetto R `HeatStress`.

La libreria:

- legge direttamente le uscite dei modelli (per esempio GRIB2 di MOLOCH, ICON, ERA5) e riconosce i
  nomi e le unità usuali delle variabili (K → °C, Pa → hPa, u/v → velocità del vento, punto di
  rugiada ↔ umidità relativa);
- calcola gli indici su tutta la griglia insieme, con codice compilato (numba) per i modelli
  iterativi (WBGT di Liljegren, PET, PMV, isolamento del vestiario) e thread in parallelo;
- riproduce i NetCDF orari di Meteosalute (42 variabili, stessi nomi di file, asse dei tempi e
  valori mancanti dello script R operativo) e lavora anche da GeoTIFF senza data né ora.

### Indici

| Gruppo | Indici |
|---|---|
| Comfort termico | UTCI e classi; PET (Höppe / VDI 3787); PMV (ISO 7730) e isolamento minimo/massimo del vestiario; temperatura apparente di Steadman (sole, ombra, interno) e classi; temperatura effettiva normale (NET); indice di disagio di Thom (THI); heat index; humidex; wind chill e tempo di congelamento |
| Stress termico al lavoro | WBGT al sole e all'ombra (Liljegren et al. 2008, come in HeatStress); WBGT di Bernard; WBGT semplificato; classi di rischio; perdita di produttività e di tempo di lavoro (TI-WTL, TWTL; Kjellstrom et al. 2018) |
| Radiazione | temperatura media radiante dalla temperatura del globo (ISO 7726) o da radiazione globale ed elevazione del sole; temperatura del globo; posizione del sole |
| Psicrometria | pressione di vapore, deficit di saturazione, punto di rugiada, temperatura di bulbo umido (Stull) |

Ogni indice ha il suo riferimento bibliografico nel codice (`salutepy.raster.list_indices()` li
elenca).

### Installazione

Python ≥ 3.10.

```bash
pip install "salutepy[fast] @ git+https://github.com/alfcrisci/salutepy"
```

L'opzione `fast` installa numba, necessario per avere buone prestazioni con WBGT, PET, PMV e
isolamento del vestiario. La lettura dei GRIB usa `cfgrib`/`ecCodes`, i GeoTIFF `rasterio`.

### Uso rapido

**1. Indici su qualunque griglia (GRIB, NetCDF, GeoTIFF)**

```python
from salutepy import raster

ds = raster.open_raster("2026112312_mol01gfs.grb2")
std = raster.standardize(ds)            # t (°C), rh (%), wind (m/s), solar (W/m2), ...
out = raster.compute(std, ["utci", "pet", "wbgt", "humidex"], workers=8)
raster.write_netcdf(out, "indici.nc")
```

```bash
python -m salutepy.raster run.grb2 -i utci,pet,humidex -o indici.nc -w 8
```

**2. Indici da GeoTIFF senza data e ora**

Temperatura, umidità relativa, vento e radiazione globale, più l'altezza del sole (raster o valore
unico) al posto di data e ora:

```bash
python -m salutepy.raster.indici_tif --t t2m.tif --rh rh.tif --wind ws10.tif --rad glob.tif --elev 55 -o indici
```

Scrive un GeoTIFF per indice (UTCI, PET, WBGT sole/ombra, temperatura media radiante e del globo e
le sei perdite di tempo di lavoro) sulla griglia degli ingressi; ingressi a più bande (per esempio
le ore) danno uscite a più bande.

**3. I NetCDF orari di Meteosalute da un run MOLOCH in GRIB2**

```bash
python -m salutepy.pipeline.meteosalute_nc 2026112312_mol01gfs.grb2 -d out                     # come lo script R operativo
python -m salutepy.pipeline.meteosalute_nc 2026112312_mol01gfs.grb2 -d out --all --time-mode valid  # tutte le ore, tempo valido reale
```

`--time-mode r` (predefinito) riproduce esattamente lo script operativo, compresa la sua
convenzione sui tempi; `--time-mode valid` usa il vero tempo valido UTC per la posizione del sole.

### Validazione

Ogni parte è stata confrontata con l'implementazione di riferimento:

- `biometeo.js` (Node.js) e `rBiometeo` per gli indici di comfort, PMV e isolamento del vestiario;
- il pacchetto R `HeatStress` per il WBGT (Liljegren e Bernard) e gli altri indici di HeatStress
  (accordo migliore di 10⁻⁶ °C);
- un NetCDF operativo di Meteosalute prodotto in R: tutte le 42 variabili sono riprodotte entro
  l'arrotondamento a 2 decimali dell'originale, salvo qualche decina di pixel (su 358 701) sui
  limiti di classe o sul passo del vestiario.

I test (`pytest`) contengono questi confronti; quelli con R e Node.js girano se i programmi sono
installati.

### Note

- La temperatura media radiante dalla temperatura del globo usa per default un globo da 0,15 m,
  come i prodotti operativi di Meteosalute; la stima da radiazione globale (`--tmrt solar`) può
  essere molto diversa con il sole basso. Scegliete e dichiarate l'opzione usata.
- Le formule di perdita di tempo di lavoro sono i polinomi usati in Meteosalute (`wbgt_aux.r`).
- Note tecniche sul porting, differenze note e bug corretti: [docs/PORTING_it.md](docs/PORTING_it.md);
  manuale d'uso: [MANUALE.md](MANUALE.md).

### Crediti

- `biometeo.js` / `rBiometeo`: Alfonso Crisci, Marco Morabito, Alessandro Messeri (CNR-IBE), GPL-3.
- WBGT di Liljegren e gli altri indici di HeatStress: porting del pacchetto R `HeatStress`
  (A. Casanueva et al.), GPL-3.
- PET: algoritmo di Höppe (1999) / VDI 3787, dall'implementazione di PALM-4U (GPL-3).

### Come citare

Vedere [CITATION.cff](CITATION.cff) o il record Zenodo della versione usata.

### Licenza

GNU General Public License v3.0 o successiva — vedere [LICENSE](LICENSE).
