"""
Porting di HeatStress::wbgt.Liljegren (modello outdoor completo, Liljegren
et al. 2008). Codice fortran originale di James C. Liljegren, tradotto in
VBA da Bruno Lemke e in R da Ana Casanueva.

Fedele riga per riga al sorgente R (HeatStress-master/R/{wbgt.Liljegren,
fTg,fTnwb,calZenith,degToRad,radToDeg,h_cylinder_in_air,h_sphere_in_air,
h_evap,thermal_cond,viscosity,diffusivity,is.leapyear}.R).

Diversamente dal resto della codebase (vettorizzata con numpy), qui il
calcolo e' intrinsecamente scalare per via della minimizzazione 1D
(scipy.optimize al posto di stats::optimize di R) dentro il ciclo su
ogni timestep/pixel — esattamente come nel codice R originale, che infatti
la chiama dentro un future_lapply per-pixel e non in forma vettoriale.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from scipy.optimize import minimize_scalar

from .psychrometrics import dewp2hurs, emis_atm, esat_buck_hpa

# ---------------------------------------------------------------------------
# costanti fisiche condivise (stessi nomi/valori del sorgente R)
# ---------------------------------------------------------------------------

_STEFANB = 0.000000056696
_CP = 1003.5
_M_AIR = 28.97
_M_H2O = 18.015
_R_GAS = 8314.34
_R_AIR = _R_GAS / _M_AIR
_RATIO = _CP * _M_AIR / _M_H2O
_PR = _CP / (_CP + (1.25 * _R_AIR))


def deg_to_rad(angle_deg: float) -> float:
    """Porting di HeatStress::degToRad.R"""
    return np.pi * angle_deg / 180.0


def rad_to_deg(angle_rad: float) -> float:
    """Porting di HeatStress::radToDeg.R"""
    return 180.0 * angle_rad / np.pi


def is_leapyear(year: int) -> bool:
    """Porting di HeatStress::is.leapyear.R"""
    return ((year % 4 == 0) and (year % 100 != 0)) or (year % 400 == 0)


def viscosity(Tk: float) -> float:
    """Porting di HeatStress::viscosity.R (BSL, p.23). Viscosita' aria, kg/(m s)."""
    omega = (Tk / 97 - 2.9) / 0.4 * (-0.034) + 1.048
    return 0.0000026693 * (28.97 * Tk) ** 0.5 / (3.617 ** 2 * omega)


def thermal_cond(Tk: float) -> float:
    """Porting di HeatStress::thermal_cond.R (BSL, p.257). Conducibilita' termica aria, W/(m K)."""
    return (_CP + 1.25 * _R_AIR) * viscosity(Tk)


def diffusivity(Tk: float, Pair: float) -> float:
    """Porting di HeatStress::diffusivity.R (BSL, p.505). Diffusivita' vapore acqueo in aria, m2/s."""
    pcrit13 = (36.4 * 218) ** (1 / 3)
    tcrit512 = (132 * 647.3) ** (5 / 12)
    tcrit12 = (132 * 647.3) ** 0.5
    mmix = (1 / 28.97 + 1 / 18.015) ** 0.5
    return 0.000364 * (Tk / tcrit12) ** 2.334 * pcrit13 * tcrit512 * mmix / (Pair / 1013.25) * 0.0001


def h_sphere_in_air(Tk: float, Pair: float, speed: float, min_speed: float, diam_globe: float) -> float:
    """Porting di HeatStress::h_sphere_in_air.R (BSL, p.409). Coeff. scambio convettivo per sfera, W/(m2 K)."""
    pr = _CP / (_CP + 1.25 * _R_AIR)
    therm_con = thermal_cond(Tk)
    density = Pair * 100 / (_R_AIR * Tk)
    speed = max(speed, min_speed)
    re = speed * density * diam_globe / viscosity(Tk)
    nu = 2 + 0.6 * re ** 0.5 * pr ** 0.3333
    return nu * therm_con / diam_globe


def h_cylinder_in_air(Tk: float, Pair: float, speed: float, min_speed: float, diam_wick: float) -> float:
    """Porting di HeatStress::h_cylinder_in_air.R (Bedingfield & Drew, eqn 32). Coeff. scambio convettivo cilindro."""
    pr = _CP / (_CP + (1.25 * _R_AIR))
    therm_con = thermal_cond(Tk)
    density = Pair * 100 / (_R_AIR * Tk)
    speed = max(speed, min_speed)
    re = speed * density * diam_wick / viscosity(Tk)
    nu = 0.281 * re ** 0.6 * pr ** 0.44
    return nu * therm_con / diam_wick


def h_evap(Tk: float) -> float:
    """Porting di HeatStress::h_evap.R (Van Wylen & Sonntag, tab. A.1.1). Calore di evaporazione, J/(kg K)."""
    return (313.15 - Tk) / 30 * (-71100) + 2407300


def cal_zenith(date: datetime, lon: float, lat: float, hour: bool = False) -> float:
    """
    Porting di HeatStress::calZenith.R.
    Angolo zenitale solare in gradi. Se hour=False usa le 12 UTC come
    nel default R (dato giornaliero); se hour=True usa l'ora esatta di
    `date` (dato orario, come nell'uso effettivo in meteosalute_op_dir_mol01.r).
    """
    EQTIME1, EQTIME2, EQTIME3 = 229.18, 0.000075, 0.001868
    EQTIME4, EQTIME5, EQTIME6 = 0.032077, 0.014615, 0.040849
    DECL1, DECL2, DECL3, DECL4 = 0.006918, 0.399912, 0.070257, 0.006758
    DECL5, DECL6, DECL7 = 0.000907, 0.002697, 0.00148

    if hour:
        utc_hour = date.hour + date.minute / 60 + date.second / 3600
    else:
        utc_hour = 12

    year = date.year
    doy = date.timetuple().tm_yday
    dpy = 366 if is_leapyear(year) else 365

    rad_lon = deg_to_rad(lon)
    rad_lat = deg_to_rad(lat)

    gamma = 2 * np.pi * ((doy - 1) + (utc_hour / 24)) / dpy

    equ_time = EQTIME1 * (
        EQTIME2
        + EQTIME3 * np.cos(gamma)
        - EQTIME4 * np.sin(gamma)
        - EQTIME5 * np.cos(2 * gamma)
        - EQTIME6 * np.sin(2 * gamma)
    )

    decli = (
        DECL1
        - DECL2 * np.cos(gamma)
        + DECL3 * np.sin(gamma)
        - DECL4 * np.cos(2 * gamma)
        + DECL5 * np.sin(2 * gamma)
        - DECL6 * np.cos(3 * gamma)
        + DECL7 * np.sin(3 * gamma)
    )

    time_offset = 0.0  # come nel sorgente R (calcolo satellitare disattivato)
    true_solar_time = (utc_hour * 60) + time_offset

    ha_deg = (true_solar_time / 4) - 180
    ha_rad = deg_to_rad(ha_deg)

    cos_zen = np.sin(rad_lat) * np.sin(decli) + np.cos(rad_lat) * np.cos(decli) * np.cos(ha_rad)
    cos_zen = min(1.0, max(-1.0, cos_zen))

    sza_rad = np.arccos(cos_zen)
    return rad_to_deg(sza_rad)


def f_tg(
    tas: float,
    relh: float,
    Pair: float,
    wind: float,
    min_speed: float,
    radiation: float,
    prop_direct: float,
    zenith: float,
    surf_albedo: float = 0.4,
    tolerance: float = 1e-4,
) -> float:
    """
    Porting di HeatStress::fTg.R. Temperatura del globo nero (degC),
    calcolata per minimizzazione iterativa (bilancio radiativo/convettivo).
    """
    emis_globe = 0.95
    alb_globe = 0.05
    diam_globe = 0.05
    emis_sfc = 0.999
    alb_sfc = surf_albedo

    if zenith <= 0:
        zenith = 1e-10
    if radiation > 0 and zenith > 1.57:
        zenith = 1.57
    if radiation > 15 and zenith > 1.54:
        zenith = 1.54
    if radiation > 900 and zenith > 1.52:
        zenith = 1.52
    if radiation < 10 and zenith == 1.57:
        radiation = 0

    tair = tas + 273.15
    rh = relh * 0.01
    cza = np.cos(zenith)
    tsfc = tair

    def fr(tglobe_prev):
        tref = 0.5 * (tglobe_prev + tair)
        h = h_sphere_in_air(tref, Pair, wind, min_speed, diam_globe)
        tglobe = (
            0.5 * (emis_atm(tair, rh) * tair ** 4 + emis_sfc * tsfc ** 4)
            - h / (emis_globe * _STEFANB) * (tglobe_prev - tair)
            + radiation / (2 * emis_globe * _STEFANB) * (1 - alb_globe) * (prop_direct * (1 / (2 * cza) - 1) + 1 + alb_sfc)
        ) ** 0.25
        return abs(tglobe - tglobe_prev)

    lo, hi = tair - 2, tair + 10
    result = minimize_scalar(fr, bounds=(lo, hi), method="bounded", options={"xatol": tolerance})
    return result.x - 273.15


def f_tnwb(
    tas: float,
    dewp: float,
    relh: float,
    Pair: float,
    wind: float,
    min_speed: float,
    radiation: float,
    prop_direct: float,
    zenith: float,
    irad: float = 1,
    surf_albedo: float = 0.4,
    tolerance: float = 1e-4,
) -> float:
    """
    Porting di HeatStress::fTnwb.R. Temperatura naturale di bulbo umido
    (degC), calcolata per minimizzazione iterativa.
    """
    emis_wick = 0.95
    alb_wick = 0.4
    diam_wick = 0.007
    len_wick = 0.0254

    emis_sfc = 0.999
    alb_sfc = surf_albedo

    if zenith <= 0:
        zenith = 1e-10
    if radiation > 0 and zenith > 1.57:
        zenith = 1.57
    if radiation > 15 and zenith > 1.54:
        zenith = 1.54
    if radiation > 900 and zenith > 1.52:
        zenith = 1.52
    if radiation < 10 and zenith == 1.57:
        radiation = 0

    tdew = dewp + 273.15
    tair = tas + 273.15
    rh = relh * 0.01

    eair = rh * esat_buck_hpa(tair)
    emis_atm_val = emis_atm(tair, rh)
    tsfc = tair
    density = Pair * 100 / (tair * _R_AIR)

    def fr(twb_prev):
        tref = 0.5 * (twb_prev + tair)
        fatm = _STEFANB * emis_wick * (
            0.5 * (emis_atm_val * tair ** 4 + emis_sfc * tsfc ** 4) - twb_prev ** 4
        ) + (1 - alb_wick) * radiation * (
            (1 - prop_direct) * (1 + 0.25 * diam_wick / len_wick)
            + ((np.tan(zenith) / 3.1416) + 0.25 * diam_wick / len_wick) * prop_direct
            + alb_sfc
        )

        sc = viscosity(tair) / (density * diffusivity(tref, Pair))
        h = h_cylinder_in_air(twb_prev, Pair, wind, min_speed, diam_wick)
        ewick = esat_buck_hpa(twb_prev)
        evap = h_evap(twb_prev)

        twb = (
            tair
            - evap / _RATIO * (ewick - eair) / (Pair - ewick) * (_PR / sc) ** 0.56
            + fatm / h * irad
        )
        return abs(twb - twb_prev)

    lo, hi = tdew - 1, tair + 1
    result = minimize_scalar(fr, bounds=(lo, hi), method="bounded", options={"xatol": tolerance})
    return result.x - 273.15


@dataclass
class WBGTResult:
    data: np.ndarray  # WBGT (degC)
    Tnwb: np.ndarray  # temperatura naturale di bulbo umido (degC)
    Tg: np.ndarray  # temperatura del globo (degC)


def wbgt_liljegren(
    tas: np.ndarray,
    dewp: np.ndarray,
    wind: np.ndarray,
    radiation: np.ndarray,
    dates,
    lon: np.ndarray,
    lat: np.ndarray,
    tolerance: float = 1e-4,
    hour: bool = True,
) -> WBGTResult:
    """
    Porting di HeatStress::wbgt.Liljegren (modello outdoor completo).

    Parameters
    ----------
    tas, dewp, wind, radiation : array-like della stessa lunghezza
        Temperatura aria (degC), temperatura di rugiada (degC), vento (m/s),
        radiazione solare globale (W/m2).
    dates : array-like di datetime.datetime (uno per elemento)
    lon, lat : array-like o scalari (gradi)
    hour : bool, default True
        RISOLTO (in precedenza il default era False per fedelta' bug-
        compatibile con meteosalute_op_dir_mol01.r, che non passava mai
        hour= e quindi usava il default R hour=FALSE — angolo zenitale
        sempre a mezzogiorno UTC fisso anche per dati orari). Il sorgente
        R e' stato corretto per passare hour=TRUE esplicitamente (dato
        che `timeh` e' gia' un timestamp orario reale), e questo default
        Python e' stato allineato di conseguenza. Passare hour=False solo
        se serve riprodurre il vecchio comportamento bug-compatibile per
        confronto/debug.

    Returns
    -------
    WBGTResult con data (WBGT), Tnwb, Tg — stessa struttura della lista
    R $data/$Tnwb/$Tg.

    Note
    ----
    Come nel sorgente R, il calcolo e' intrinsecamente per-elemento (loop
    Python qui, future_lapply per-pixel in R) per via della minimizzazione
    1D di fTg/fTnwb: non e' vettorizzabile in numpy puro.
    """
    tas = np.atleast_1d(np.asarray(tas, dtype=float))
    dewp = np.atleast_1d(np.asarray(dewp, dtype=float))
    wind = np.atleast_1d(np.asarray(wind, dtype=float)).copy()
    radiation = np.atleast_1d(np.asarray(radiation, dtype=float)).copy()
    lon = np.broadcast_to(np.asarray(lon, dtype=float), tas.shape)
    lat = np.broadcast_to(np.asarray(lat, dtype=float), tas.shape)

    n = len(tas)
    prop_direct = 0.8
    pair = 1010.0
    min_wind_speed = 0.1

    radiation[radiation < 0] = 0
    wind[wind < 0] = 0

    # noNAs=TRUE, swap=FALSE (default R): impedisce dewp > tas forzando dewp=tas
    noway = (dewp - tas) > tolerance
    dewp = np.where(noway, tas, dewp)

    relh = dewp2hurs(tas, dewp)

    tnwb = np.full(n, np.nan)
    tg = np.full(n, np.nan)

    mask = ~np.isnan(tas + dewp + wind + radiation)

    for i in np.where(mask)[0]:
        zenith_deg = cal_zenith(dates[i], float(lon[i]), float(lat[i]), hour=hour)
        zenith_angle = deg_to_rad(zenith_deg)

        tg[i] = f_tg(
            tas[i], relh[i], pair, wind[i], min_wind_speed, radiation[i],
            prop_direct, zenith_angle, tolerance=tolerance,
        )
        tnwb[i] = f_tnwb(
            tas[i], dewp[i], relh[i], pair, wind[i], min_wind_speed, radiation[i],
            prop_direct, zenith_angle, tolerance=tolerance,
        )

    wbgt = 0.7 * tnwb + 0.2 * tg + 0.1 * tas
    return WBGTResult(data=wbgt, Tnwb=tnwb, Tg=tg)
