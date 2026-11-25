"""indici_tif: GeoTIFF senza data/ora, con l'altezza del sole come ingresso."""

import numpy as np
import pytest

from salutepy.indices.heatstress_np import tashurs2dewp
from salutepy.indices.wbgt_fast import _date_parts, cal_zenith, wbgt_liljegren, wbgt_liljegren_elev


def test_elev_equals_datetime_version():
    rng = np.random.default_rng(3)
    n = 300
    t = rng.uniform(-5, 38, n)
    td = t - rng.uniform(0, 15, n)
    w = rng.uniform(0.2, 8, n)
    r = rng.uniform(0, 950, n)
    lat = rng.uniform(36, 47, n)
    lon = rng.uniform(6, 19, n)
    when = np.datetime64("2026-07-15T11:00", "s")
    ref = wbgt_liljegren(t, td, w, r, when, lon, lat, hour=True)
    y, doy, hr = _date_parts(when, t.shape)
    elev = 90 - np.array([cal_zenith(y[0], doy[0], hr[0], lo, la, True, False) for la, lo in zip(lat, lon)])
    got = wbgt_liljegren_elev(t, td, w, r, elev)
    for k in ("data", "Tg", "Tnwb"):
        np.testing.assert_allclose(got[k], ref[k], atol=1e-6)


def test_geotiff_roundtrip(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    from salutepy.raster.indici_tif import OUTPUTS, calcola, indici_da_geotiff

    shape = (2, 4, 5)  # due bande (es. due ore)
    rng = np.random.default_rng(1)
    data = {"t": rng.uniform(15, 35, shape), "rh": rng.uniform(20, 90, shape),
            "w": rng.uniform(0.5, 6, shape), "r": rng.uniform(0, 900, shape)}
    data["t"][0, 0, 0] = -9999  # nodata
    prof = dict(driver="GTiff", height=4, width=5, count=2, dtype="float32", crs="EPSG:4326",
                transform=from_origin(10, 45, 0.1, 0.1), nodata=-9999)
    paths = {}
    for k, a in data.items():
        paths[k] = tmp_path / f"{k}.tif"
        with rasterio.open(paths[k], "w", **prof) as d:
            d.write(a.astype("float32"))
    files = indici_da_geotiff(paths["t"], paths["rh"], paths["w"], paths["r"], 50, tmp_path / "out", verbose=False)
    assert len(files) == len(OUTPUTS)
    f32 = {k: np.where(a == -9999, np.nan, a.astype("float32").astype(float)) for k, a in data.items()}
    exp = calcola(f32["t"], f32["rh"], f32["w"], f32["r"], 50.0)
    for name in ("utci", "pet", "wbgtsun", "wbgtsha", "airtiwtl"):
        with rasterio.open(tmp_path / "out" / f"{name}.tif") as d:
            got = d.read(masked=True)
            assert d.count == 2 and d.crs.to_epsg() == 4326
        assert got.mask[0, 0, 0]
        np.testing.assert_allclose(got.filled(np.nan), exp[name].astype("float32"), atol=1e-4, equal_nan=True)


def test_vpd_units_tif_and_engine(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    import xarray as xr
    from rasterio.transform import from_origin

    from salutepy.indices.psychrometrics import deficitsat
    from salutepy.raster.engine import compute
    from salutepy.raster.indici_tif import indici_da_geotiff

    t = np.array([[[20.0, 30.0], [-5.0, 10.0]]])
    rh = np.array([[[50.0, 40.0], [80.0, 100.0]]])
    ref = deficitsat(t, rh)
    np.testing.assert_allclose(ref[0, 0], [11.705, 25.494], atol=1e-3)
    prof = dict(driver="GTiff", height=2, width=2, count=1, dtype="float32", crs="EPSG:4326",
                transform=from_origin(10, 45, 0.1, 0.1), nodata=-9999)
    for k, a in (("t", t), ("rh", rh)):
        with rasterio.open(tmp_path / f"{k}.tif", "w", **prof) as d:
            d.write(a.astype("float32"))
    for unit, div in (("hPa", 1.0), ("kPa", 10.0)):
        out = tmp_path / unit
        indici_da_geotiff(tmp_path / "t.tif", tmp_path / "rh.tif", None, None, None, out, indices=["vpd"],
                          vpd_unit=unit, verbose=False)
        with rasterio.open(out / "vpd.tif") as d:
            assert d.tags()["units"] == unit
            np.testing.assert_allclose(d.read(), ref / div, atol=1e-4)
    with pytest.raises(ValueError):
        indici_da_geotiff(tmp_path / "t.tif", tmp_path / "rh.tif", None, None, None, tmp_path / "x", indices=["utci"],
                          verbose=False)
    std = xr.Dataset({"t": (("y", "x"), t[0]), "rh": (("y", "x"), rh[0])})
    for unit, div in (("hPa", 1.0), ("kPa", 10.0)):
        o = compute(std, ["vpd"], workers=1, params={"vpd": {"unit": unit}})
        assert o.vpd.attrs["units"] == unit
        np.testing.assert_allclose(o.vpd.values, ref[0] / div)
