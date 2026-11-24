"""
Porting di extract_samples_simple.R: estrae dal GRIB2 del modello i
valori nei punti-stazione (per confronto/validazione contro osservazioni),
per un intervallo di timestep richiesto.

*** LIMITI NOTI DI QUESTO PORTING ***

1. STAZIONI: il sorgente R legge `location/wk_staz_obs_61.rds`, un
   oggetto terra::SpatVector serializzato con terra::wrap() (non un
   semplice data.frame) — pyreadr NON può leggerlo correttamente, perché
   contiene puntatori C++ serializzati, non dati tabellari puri. Questo
   file va prima convertito in R con:

       x = terra::unwrap(readRDS("wk_staz_obs_61.rds"))
       df = data.frame(terra::crds(x), as.data.frame(x))
       names(df) = c("lon","lat","nome","elev")
       write.csv(df, "wk_staz_obs_61.csv", row.names=FALSE)

   e il CSV risultante passato a `stations_csv`. Il file "wk_staz_obs_61.rds"
   stesso non è mai stato caricato in questa sessione. Per riferimento,
   `staz_verify_italy.txt` (caricato in questa stessa conversazione) ha
   una struttura compatibile (nome/lon/lat/quota) ma solo 39 stazioni
   contro le 61 implicite nel nome del file R — probabilmente NON lo
   stesso file, ma un punto di partenza ragionevole se non si dispone
   dell'originale.

2. LETTURA BANDE GRIB PER NOME: a differenza di grib_reader.py (lettura
   posizionale, 9 variabili in ordine fisso), qui il file GRIB2 ha
   centinaia di bande (fino a 61 timestep x 6 variabili) e la selezione
   avviene per NOME variabile (grep su "Temperature", "Dew point
   temperature", ecc.) — esattamente come fa R con
   `names(temp_staz_all)`. Qui si usa il tag GDAL `GRIB_COMMENT` di
   ciascuna banda (esposto da rasterio) per lo stesso grep-matching.
   *** NON TESTATO *** in questa sessione per mancanza di un file GRIB2
   reale con più timestep: il nome esatto del tag GDAL e il contenuto
   testuale possono variare a seconda della versione di GDAL/eccodes e
   andrebbero validati al primo uso reale.

3. OUTPUT: il sorgente R salva con `saveRDS()`. pyreadr non supporta la
   scrittura di .rds (solo lettura) senza un bridge R (rpy2). Qui il
   risultato è salvato come CSV (`<fileout>.csv` invece di
   `<fileout>.data.rds`) — se un consumer a valle richiede esplicitamente
   .rds, va aggiunta una conversione con R o rpy2.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from ..indices.psychrometrics import dewp2hurs

def default_stations_csv() -> Path:
    """
    Ritorna il percorso del file stazioni di default incluso nel package
    (`salutepy/data/staz_verify_italia.csv`), normalizzato da
    `staz_verify_italy.txt` caricato dall'utente.

    ATTENZIONE: 40 stazioni, contro le 61 implicite nel nome del file R
    originale (`wk_staz_obs_61.rds`) — probabilmente NON lo stesso
    elenco stazioni usato in produzione, ma un sostituto ragionevole se
    l'originale non è disponibile. Contiene inoltre due stazioni
    entrambe chiamate "Perugia" con coordinate/quota diverse — verificare
    se intenzionale (es. aeroporto vs centro città) prima di un uso che
    aggreghi per nome stazione.
    """
    return Path(__file__).resolve().parent.parent / "data" / "staz_verify_italia.csv"


def capoluoghi_stations_csv() -> Path:
    """
    Ritorna il percorso del file dei 116 capoluoghi di provincia italiani
    incluso nel package (`salutepy/data/capoluoghi_province_italiani.csv`),
    normalizzato da `capoluoghi_province_italiani.csv` caricato
    dall'utente. Non contiene elevazione (colonna 'elev' = NaN per tutte
    le righe): il file sorgente ha solo comune/sigla_provincia/provincia/
    regione/lat/lon, nessuna quota — non usata comunque da
    extract_samples() se non per riferimento nell'output.
    """
    return Path(__file__).resolve().parent.parent / "data" / "capoluoghi_province_italiani.csv"


VARS = ["Temperature", "Dew point temperature", "u-component of wind", "v-component of wind",
        "Total cloud cover", "Downward short-wave radiation"]
VARS_SHORT = ["X2t", "X2d", "X10u", "X10v", "clct", "dswrf"]

# alias comuni per le colonne del file stazioni: mapping per NOME, non per
# posizione (un file con colonne in ordine diverso da lon,lat,nome,elev —
# es. "location,lon,lat,quota" come staz_verify_italy.txt, o
# "comune,sigla_provincia,provincia,regione,lat,lon" come
# capoluoghi_province_italiani.csv — andrebbe altrimenti mappato in modo
# silenziosamente sbagliato con un rename puramente posizionale)
_COLUMN_ALIASES = {
    "lon": {"lon", "lng", "longitude", "x"},
    "lat": {"lat", "latitude", "y"},
    "nome": {"nome", "name", "location", "stazione", "station", "comune"},
}
_ELEV_ALIASES = {"elev", "elevation", "quota", "z", "alt", "altitude"}


def _normalize_station_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rinomina le colonne del file stazioni sulla base del NOME
    (case-insensitive), non della posizione. La colonna elevazione è
    OPZIONALE: se assente (es. capoluoghi_province_italiani.csv, che non
    ha quota) viene riempita con NaN invece di bloccare l'estrazione —
    l'elevazione non è comunque usata da extract_samples(), solo
    riportata nell'output per riferimento.
    """
    lower_map = {c.lower(): c for c in df.columns}
    rename = {}
    for target, aliases in _COLUMN_ALIASES.items():
        found = next((lower_map[a] for a in aliases if a in lower_map), None)
        if found is None:
            raise ValueError(
                f"Colonna '{target}' non trovata nel file stazioni (colonne disponibili: {list(df.columns)}). "
                f"Alias accettati per '{target}': {sorted(aliases)}."
            )
        rename[found] = target

    df = df.rename(columns=rename)

    elev_col = next((lower_map[a] for a in _ELEV_ALIASES if a in lower_map), None)
    if elev_col is not None:
        df = df.rename(columns={elev_col: "elev"})
    else:
        df["elev"] = float("nan")

    return df[["lon", "lat", "nome", "elev"]]


def _band_descriptions(ds: rasterio.DatasetReader) -> list[str]:
    """
    Ritorna una descrizione testuale per ciascuna banda, cercando prima il
    tag GDAL 'GRIB_COMMENT' (dove terra tipicamente pesca il nome
    variabile), poi 'GRIB_ELEMENT', poi la description nativa di rasterio.
    *** Non validato su un file reale, vedi nota di modulo. ***
    """
    descs = []
    for i in range(1, ds.count + 1):
        tags = ds.tags(i)
        desc = tags.get("GRIB_COMMENT") or tags.get("GRIB_ELEMENT") or ds.descriptions[i - 1] or ""
        descs.append(desc)
    return descs


def _band_times(ds: rasterio.DatasetReader) -> list[datetime]:
    """
    Ritorna il timestamp di validità di ciascuna banda, dal tag GDAL
    'GRIB_VALID_TIME' (secondi epoch, formato tipico del driver GRIB di
    GDAL). *** Non validato su un file reale. ***
    """
    times = []
    for i in range(1, ds.count + 1):
        tags = ds.tags(i)
        valid_time = tags.get("GRIB_VALID_TIME")
        if valid_time:
            # formato tipico: "1234567890 sec UTC"
            epoch = int(str(valid_time).split()[0])
            times.append(datetime.utcfromtimestamp(epoch))
        else:
            times.append(None)
    return times


def extract_samples(
    filegrb: str | Path,
    fileout: str | Path,
    interval_ini: int,
    interval_fin: int,
    model: str,
    stations_csv: str | Path,
) -> Path:
    """
    Porting di extract_samples_simple.R.

    Parameters
    ----------
    filegrb : file GRIB2 di input
    fileout : percorso di output SENZA estensione (verrà scritto <fileout>.csv,
        equivalente semantico di <fileout>.data.rds nel sorgente R — vedi
        nota di modulo sul formato di output)
    interval_ini, interval_fin : range di timestep da estrarre (1-indexed,
        stesso significato di intervalini/intervalfin in R)
    model : etichetta modello (es. "MOLOCH02_GFS_00")
    stations_csv : CSV con colonne lon,lat,nome,elev (vedi nota di modulo
        sulla necessità di pre-convertire wk_staz_obs_61.rds)

    Returns
    -------
    Path del file CSV scritto
    """
    fileout = Path(str(fileout) + ".csv")

    stations = pd.read_csv(stations_csv)
    stations = _normalize_station_columns(stations)

    with rasterio.open(filegrb) as ds:
        descs = _band_descriptions(ds)
        times = _band_times(ds)

        # indici di banda (1-based rasterio) per ciascuna variabile, in ordine
        # di apparizione nel file — equivalente di grep(vars[j], names(...))
        var_band_idx: dict[str, list[int]] = {v: [] for v in VARS}
        for band_i, desc in enumerate(descs, start=1):
            for v in VARS:
                if v in desc:
                    var_band_idx[v].append(band_i)

        for v in VARS:
            if len(var_band_idx[v]) < interval_fin:
                raise ValueError(
                    f"Variabile '{v}': trovate solo {len(var_band_idx[v])} bande nel file, "
                    f"richiesto intervallo fino a {interval_fin}. Verificare il tag GDAL usato "
                    "per il match (vedi nota in cima al modulo: match non validato su file reale)."
                )

        rows = []
        for _, st in stations.iterrows():
            row_idx, col_idx = ds.index(st["lon"], st["lat"])

            var_values = {}
            for v, short in zip(VARS, VARS_SHORT):
                band_ids = var_band_idx[v][interval_ini - 1 : interval_fin]
                values = [ds.read(b, window=((row_idx, row_idx + 1), (col_idx, col_idx + 1)))[0, 0] for b in band_ids]
                var_values[short] = values

            selected_times = times[var_band_idx[VARS[0]][interval_ini - 1] - 1 : var_band_idx[VARS[0]][interval_fin - 1]]
            n = interval_fin - interval_ini + 1

            df = pd.DataFrame({
                "stazione": [st["nome"]] * n,
                "lon": [st["lon"]] * n,
                "lat": [st["lat"]] * n,
                "Data": [t.strftime("%Y-%m-%d") if t else None for t in selected_times],
                "hour": [t.strftime("%H") if t else None for t in selected_times],
                "X2t": var_values["X2t"],
                "X2d": var_values["X2d"],
                "X10u": var_values["X10u"],
                "X10v": var_values["X10v"],
                "clct": var_values["clct"],
                "dswrf": var_values["dswrf"],
            })

            # dayindex: come in R, indice progressivo per giorno distinto
            df["dayindex"] = df["Data"].astype("category").cat.codes.astype(str)

            # NOTA: conversione K->degC con -273.16 (non -273.15), stessa
            # convenzione usata altrove nel sorgente R per questo dataset
            # (vedi commento in psychrometrics.py sulla differenza tra le
            # varie formule di conversione presenti nella codebase originale)
            df["RH"] = dewp2hurs(df["X2t"] - 273.16, df["X2d"] - 273.16)
            df["wind_speed"] = np.sqrt(df["X10u"] ** 2 + df["X10v"] ** 2)
            df["model"] = model

            rows.append(df)

    result = pd.concat(rows, ignore_index=True)
    result.to_csv(fileout, index=False)
    return fileout
