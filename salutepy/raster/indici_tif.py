"""
UTCI, PET, WBGT al sole e all'ombra e perdite di tempo di lavoro (WTL) da GeoTIFF di temperatura,
umidità relativa, vento e radiazione globale, senza data e ora: l'altezza del sole si dà come ingresso
(GeoTIFF o valore unico).

Catena di calcolo (come Meteosalute):

* WBGT al sole: Liljegren et al. (2008) / HeatStress, con zenit = 90 - elevazione, vento a 2 m;
* temperatura del globo Tg dello stesso calcolo; WBGT all'ombra = 0.7 Tnwb + 0.3 T (Meteosalute);
* temperatura media radiante: da Tg con ``mrt_globe`` (ISO 7726) con diametro 0.15 m come i NetCDF
  operativi di Meteosalute (``--globe-diam 0.05`` = il globo di Liljegren), oppure ``--tmrt solar`` con
  ``mrt_solar_proj`` (radiazione + elevazione). Le due stime possono differire molto (con sole basso
  mrt_solar_proj resta vicina alla temperatura dell'aria, mrt_globe con il globo piccolo è molto più alta);
* UTCI (Broede et al. 2012) con vento a 10 m; PET (Hoeppe, VDI 3787) con vento a 1.1 m;
* WTL (Kjellstrom et al.): % di tempo perso (TI-WTL) e ore perse (TWTL) da WBGT al sole, all'ombra e
  dalla temperatura dell'aria;
* VPD: deficit di pressione di vapore (deficitsat di biometeo.js, IAPWS), in hPa o kPa (``--vpd-unit``).

Uso::

    python -m salutepy.raster.indici_tif --t t2m.tif --rh rh.tif --wind ws10.tif --rad glob.tif --elev 55 -o indici
    python -m salutepy.raster.indici_tif --t t.tif --rh rh.tif --wind w.tif --rad r.tif --elev elev.tif -o indici

Unità: t in degC, rh in %, vento in m/s (all'altezza ``--wind-height``, default 10 m), radiazione in W/m2,
elevazione in gradi. Se i GeoTIFF hanno più bande (es. ore), ogni banda è calcolata e scritta nello stesso
ordine. Un file per indice in ``-o`` (stessa griglia e proiezione degli ingressi, nodata -9999).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from ..indices import biometeo_np as B
from ..indices.heatstress_np import tashurs2dewp
from ..indices.misc import reducewind
from ..indices.pet import pet_hoppe
from ..indices.productivity import mean_airTI_WTL, mean_airTWTL, mean_wbgtTI_WTL, mean_wbgtWTL
from ..indices.psychrometrics import deficitsat
from ..indices.wbgt_fast import wbgt_liljegren_elev

NODATA = -9999.0

OUTPUTS = {
    "utci": ("degC", "UTCI"),
    "pet": ("degC", "PET (Hoeppe, VDI 3787)"),
    "wbgtsun": ("degC", "WBGT al sole (Liljegren)"),
    "wbgtsha": ("degC", "WBGT all'ombra 0.7 Tnwb + 0.3 T"),
    "tmrt": ("degC", "temperatura media radiante"),
    "tglob": ("degC", "temperatura del globo (Liljegren)"),
    "wbgttiwtlsun": ("perc", "TI-WTL da WBGT al sole"),
    "wbgttwtlsun": ("hours", "TWTL da WBGT al sole"),
    "wbgttiwtlsha": ("perc", "TI-WTL da WBGT all'ombra"),
    "wbgttwtlsha": ("hours", "TWTL da WBGT all'ombra"),
    "airtiwtl": ("perc", "TI-WTL dalla temperatura dell'aria"),
    "airtwtl": ("hours", "TWTL dalla temperatura dell'aria"),
    "vpd": ("hPa", "deficit di pressione di vapore (vapour pressure deficit)"),
}


def read_tif(path):
    """Array (bande, righe, colonne) float con NaN al posto del nodata, e profilo rasterio."""
    import rasterio

    with rasterio.open(path) as src:
        a = src.read().astype(float)
        if src.nodata is not None:
            a[a == src.nodata] = np.nan
        return a, src.profile.copy(), src.transform, src.crs


def _input(value, name, ref=None):
    """GeoTIFF (percorso) o numero; controlla che la griglia sia quella di riferimento."""
    try:
        return np.asarray(float(value)), None
    except (TypeError, ValueError):
        pass
    a, prof, tr, crs = read_tif(value)
    if ref is not None:
        if a.shape[1:] != ref["shape"] or not tr.almost_equals(ref["transform"]) or crs != ref["crs"]:
            raise ValueError(f"{name}: griglia diversa da quella della temperatura ({value})")
        if a.shape[0] not in (1, ref["bands"]):
            raise ValueError(f"{name}: {a.shape[0]} bande, attese 1 o {ref['bands']}")
    return a, prof


def calcola(t, rh, wind, rad, elev, wind_height=10.0, tmrt="globe", pair=1013.25, workers=None, globe_diam=0.15,
            vpd_unit="hPa"):
    """
    Calcola tutti gli indici su array già letti (si allineano per broadcasting).
    Restituisce un dict nome -> array.
    """
    if workers:
        try:
            import numba

            numba.set_num_threads(min(workers, numba.config.NUMBA_NUM_THREADS))
        except ImportError:
            pass
    t, rh, wind, rad, elev = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (t, rh, wind, rad, elev)))
    ws10 = reducewind(wind, ref=wind_height, fin=10.0) if wind_height != 10 else wind
    ws2 = reducewind(wind, ref=wind_height, fin=2.0)
    ws11 = reducewind(wind, ref=wind_height, fin=1.1)
    td = tashurs2dewp(t, rh)
    w = wbgt_liljegren_elev(t, td, ws2, rad, elev, pair=pair)
    wsun, tnwb, tg = w["data"], w["Tnwb"], w["Tg"]
    wsha = 0.7 * tnwb + 0.3 * t
    if tmrt == "globe":
        mrt = B.mrt_globe(t, tg, ws2, globe_diam)
    elif tmrt == "solar":
        mrt = B.mrt_solar_proj(t, rh, rad, elev)
    else:
        raise ValueError("tmrt: 'globe' oppure 'solar'")
    out = {
        "utci": B.utci(t, rh, ws10, mrt),
        "pet": pet_hoppe(t, rh, ws11, mrt, pair),
        "wbgtsun": wsun, "wbgtsha": wsha, "tmrt": mrt, "tglob": tg,
        "wbgttiwtlsun": mean_wbgtTI_WTL(wsun), "wbgttwtlsun": mean_wbgtWTL(wsun),
        "wbgttiwtlsha": mean_wbgtTI_WTL(wsha), "wbgttwtlsha": mean_wbgtWTL(wsha),
        "airtiwtl": mean_airTI_WTL(t), "airtwtl": mean_airTWTL(t),
        "vpd": deficitsat(t, rh) / (10.0 if vpd_unit.lower() == "kpa" else 1.0),
    }
    base = np.isnan(t) | np.isnan(rh)
    full = base | np.isnan(wind) | np.isnan(rad) | np.isnan(elev)
    mask = {k: full for k in out}
    mask["vpd"] = base
    mask["airtiwtl"] = mask["airtwtl"] = np.isnan(t)
    return {k: np.where(mask[k], np.nan, np.asarray(v, dtype=float)) for k, v in out.items()}


# indici che richiedono solo temperatura (e umidità): calcolabili senza vento, radiazione ed elevazione
SOLO_T_RH = {"vpd", "airtiwtl", "airtwtl"}


def indici_da_geotiff(t, rh, wind, rad, elev, outdir="indici", wind_height=10.0, tmrt="globe",
                      pair=1013.25, indices=None, workers=None, verbose=True, globe_diam=0.15,
                      vpd_unit="hPa") -> list[Path]:
    """Legge i GeoTIFF, calcola gli indici e scrive un GeoTIFF per indice in ``outdir``."""
    import rasterio

    t0 = time.perf_counter()
    indices = list(indices or OUTPUTS)
    missing = [n for n, v in (("wind", wind), ("rad", rad), ("elev", elev)) if v is None]
    if missing and set(indices) - SOLO_T_RH:
        raise ValueError(f"mancano {missing}: senza vento, radiazione ed elevazione si possono calcolare solo "
                         f"{sorted(SOLO_T_RH)}")
    wind, rad, elev = (np.nan if v is None else v for v in (wind, rad, elev))
    ta, prof, tr, crs = read_tif(t)
    ref = {"shape": ta.shape[1:], "transform": tr, "crs": crs, "bands": ta.shape[0]}
    arrs = [ta] + [_input(v, n, ref)[0] for v, n in ((rh, "rh"), (wind, "wind"), (rad, "rad"), (elev, "elev"))]
    if vpd_unit.lower() not in ("hpa", "kpa"):
        raise ValueError("vpd_unit: 'hPa' oppure 'kPa'")
    res = calcola(*arrs, wind_height=wind_height, tmrt=tmrt, pair=pair, workers=workers, globe_diam=globe_diam,
                  vpd_unit=vpd_unit)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prof.update(dtype="float32", nodata=NODATA, count=ta.shape[0], compress="deflate", predictor=3, driver="GTiff")
    files = []
    for name in indices:
        a = np.where(np.isnan(res[name]), NODATA, res[name]).astype("float32")
        out = outdir / f"{name}.tif"
        with rasterio.open(out, "w", **prof) as dst:
            dst.write(a)
            units = ("kPa" if vpd_unit.lower() == "kpa" else "hPa") if name == "vpd" else OUTPUTS[name][0]
            dst.update_tags(index=name, units=units, description=OUTPUTS[name][1])
            for b in range(a.shape[0]):
                dst.set_band_description(b + 1, f"{name}_{b + 1}" if a.shape[0] > 1 else name)
        files.append(out)
    if verbose:
        print(f"{len(files)} GeoTIFF ({ta.shape[0]} bande, {ta.shape[1]}x{ta.shape[2]}) in "
              f"{time.perf_counter() - t0:.1f} s -> {outdir}")
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m salutepy.raster.indici_tif",
                                 description="UTCI, PET, WBGT sole/ombra e WTL da GeoTIFF senza data e ora")
    ap.add_argument("--t", required=True, help="temperatura dell'aria (degC), GeoTIFF")
    ap.add_argument("--rh", required=True, help="umidità relativa (%%), GeoTIFF o numero")
    ap.add_argument("--wind", help="velocità del vento (m/s), GeoTIFF o numero")
    ap.add_argument("--rad", help="radiazione globale (W/m2), GeoTIFF o numero")
    ap.add_argument("--elev", help="altezza del sole (gradi), GeoTIFF o numero (non servono vento, radiazione "
                    "ed elevazione se si chiedono solo vpd, airtiwtl, airtwtl)")
    ap.add_argument("-o", "--outdir", default="indici")
    ap.add_argument("--wind-height", type=float, default=10.0, help="quota del vento in ingresso (m)")
    ap.add_argument("--tmrt", default="globe", choices=["globe", "solar"])
    ap.add_argument("--globe-diam", type=float, default=0.15,
                    help="diametro (m) per mrt_globe: 0.15 = come i NetCDF operativi (default); 0.05 = globo di Liljegren")
    ap.add_argument("--pair", type=float, default=1013.25, help="pressione (hPa)")
    ap.add_argument("--vpd-unit", default="hPa", choices=["hPa", "kPa"], help="unità del deficit di pressione di vapore")
    ap.add_argument("--indici", help="sottoinsieme, separato da virgole (default tutti)")
    ap.add_argument("-w", "--workers", type=int)
    a = ap.parse_args(argv)
    indices = a.indici.split(",") if a.indici else None
    if indices and set(indices) - set(OUTPUTS):
        ap.error(f"indici sconosciuti: {sorted(set(indices) - set(OUTPUTS))}; disponibili: {', '.join(OUTPUTS)}")
    if (a.wind is None or a.rad is None or a.elev is None) and (not indices or set(indices) - SOLO_T_RH):
        ap.error("--wind, --rad ed --elev sono necessari, salvo con --indici solo tra vpd, airtiwtl, airtwtl")
    indici_da_geotiff(a.t, a.rh, a.wind, a.rad, a.elev, a.outdir, a.wind_height, a.tmrt, a.pair, indices, a.workers,
                      globe_diam=a.globe_diam, vpd_unit=a.vpd_unit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
