"""
End-to-end tests of salutepy.raster on synthetic GRIB2, NetCDF and GeoTIFF files.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from salutepy import raster
from salutepy.indices import biometeo_np as b
from salutepy.indices.pet import pet_hoppe


def synthetic(nt=2, ny=6, nx=7):
    rng = np.random.default_rng(0)
    times = pd.date_range("2026-07-15T10:00", periods=nt, freq="3h").values
    lat = np.linspace(46.0, 44.0, ny)
    lon = np.linspace(10.0, 12.0, nx)
    shape = (nt, ny, nx)
    t2m = 273.15 + rng.uniform(15, 38, shape)
    d2m = t2m - rng.uniform(2, 15, shape)
    ds = xr.Dataset(
        {
            "t2m": (("time", "latitude", "longitude"), t2m, {"units": "K"}),
            "d2m": (("time", "latitude", "longitude"), d2m, {"units": "K"}),
            "u10": (("time", "latitude", "longitude"), rng.uniform(-6, 6, shape), {"units": "m s**-1"}),
            "v10": (("time", "latitude", "longitude"), rng.uniform(-6, 6, shape), {"units": "m s**-1"}),
            "sp": (("time", "latitude", "longitude"), rng.uniform(95000, 102000, shape), {"units": "Pa"}),
            "sdswrf": (("time", "latitude", "longitude"), rng.uniform(100, 900, shape), {"units": "W m**-2"}),
        },
        coords={"time": times, "latitude": lat, "longitude": lon},
    )
    return ds


def expected(std):
    t, rh, w, p = std.t.values, std.rh.values, std.wind.values, std.pair.values
    return {"humidex": b.humidex(t, rh), "net": b.net_index(t, rh, w), "thi": b.thi(t, rh),
            "pet_shade": pet_hoppe(t, rh, w, t, p)}


def test_standardize_units():
    std = raster.standardize(synthetic())
    assert set(["t", "td", "rh", "wind", "pair", "solar"]) <= set(std.data_vars)
    assert 10 < float(std.t.mean()) < 45
    assert 900 < float(std.pair.mean()) < 1100
    assert float(std.rh.max()) <= 100


def test_netcdf_roundtrip_and_parallel(tmp_path):
    ds = synthetic()
    f = tmp_path / "in.nc"
    ds.to_netcdf(f)
    std = raster.standardize(raster.open_raster(f))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        one = raster.compute(std, "utci,pet,humidex,net,thi,windchill,heatindex,mrt_solar,sunelev", workers=1)
        two = raster.compute(std, "utci,pet,humidex,net,thi,windchill,heatindex,mrt_solar,sunelev", workers=3, min_block=1)
    for v in one.data_vars:
        np.testing.assert_array_equal(one[v].values, two[v].values)
    exp = expected(std)
    np.testing.assert_allclose(one.humidex.values, exp["humidex"])
    np.testing.assert_allclose(one.net.values, exp["net"])
    # utci uses tmrt from solar radiation (no tmrt in the file)
    tm = b.mrt_solar_proj(std.t.values, std.rh.values, std.solar.values, one.sunelev.values)
    np.testing.assert_allclose(one.utci.values, b.utci(std.t.values, std.rh.values, std.wind.values, tm))
    assert one.utci_class.dtype == np.int8 and "labels_it" in one.utci_class.attrs
    out = tmp_path / "out.nc"
    raster.write_netcdf(one, out)
    back = xr.open_dataset(out)
    np.testing.assert_allclose(back.pet.values, one.pet.values)


def _write_grib2(path, fields, lat, lon, when):
    """Write 2D fields to GRIB2 with ecCodes: fields = [(paramId, level, typeOfLevel, 2D array)]."""
    eccodes = pytest.importorskip("eccodes")
    with open(path, "wb") as fout:
        for param, level, tol, arr in fields:
            h = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
            eccodes.codes_set(h, "Ni", len(lon))
            eccodes.codes_set(h, "Nj", len(lat))
            eccodes.codes_set(h, "latitudeOfFirstGridPointInDegrees", float(lat[0]))
            eccodes.codes_set(h, "latitudeOfLastGridPointInDegrees", float(lat[-1]))
            eccodes.codes_set(h, "longitudeOfFirstGridPointInDegrees", float(lon[0]))
            eccodes.codes_set(h, "longitudeOfLastGridPointInDegrees", float(lon[-1]))
            eccodes.codes_set(h, "iDirectionIncrementInDegrees", float(abs(lon[1] - lon[0])))
            eccodes.codes_set(h, "jDirectionIncrementInDegrees", float(abs(lat[1] - lat[0])))
            eccodes.codes_set(h, "dataDate", int(when.strftime("%Y%m%d")))
            eccodes.codes_set(h, "dataTime", int(when.strftime("%H%M")))
            eccodes.codes_set(h, "typeOfLevel", tol)
            eccodes.codes_set(h, "level", level)
            eccodes.codes_set(h, "paramId", param)
            eccodes.codes_set_values(h, np.asarray(arr, dtype=float).ravel())
            eccodes.codes_write(h, fout)
            eccodes.codes_release(h)


def test_grib2(tmp_path):
    pytest.importorskip("cfgrib")
    ds = synthetic(nt=1).isel(time=0)
    f = tmp_path / "in.grb2"
    when = pd.Timestamp(ds.time.values).to_pydatetime()
    lat, lon = ds.latitude.values, ds.longitude.values
    # ECMWF paramIds: 2t=167, 2d=168, 10u=165, 10v=166, sp=134
    _write_grib2(f, [(167, 2, "heightAboveGround", ds.t2m.values), (168, 2, "heightAboveGround", ds.d2m.values),
                     (165, 10, "heightAboveGround", ds.u10.values), (166, 10, "heightAboveGround", ds.v10.values),
                     (134, 0, "surface", ds.sp.values)], lat, lon, when)
    opened = raster.open_raster(f)
    std = raster.standardize(opened)
    assert {"t", "rh", "wind", "pair"} <= set(std.data_vars)
    np.testing.assert_allclose(std.t.values.squeeze(), ds.t2m.values - 273.15, atol=0.01)
    res = raster.compute(std, ["humidex", "net", "pet", "sunelev"], workers=2, min_block=1)
    np.testing.assert_allclose(res.humidex.values, b.humidex(std.t.values, std.rh.values))
    assert np.isfinite(res.sunelev.values).all()


def test_geotiff(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    ds = synthetic(nt=1).isel(time=0)
    std0 = raster.standardize(ds)
    tr = from_origin(10.0, 46.0, 2.0 / 7, 2.0 / 6)
    files = {}
    for name, arr in (("t", std0.t.values), ("rh", std0.rh.values), ("wind", std0.wind.values),
                      ("tmrt", std0.t.values + 12)):
        p = tmp_path / f"{name}.tif"
        with rasterio.open(p, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                           dtype="float64", crs="EPSG:4326", transform=tr) as dst:
            dst.write(arr, 1)
        files[name] = p
    std = raster.standardize(raster.open_rasters(files))
    res = raster.compute(std, ["utci", "pet"], workers=2, min_block=1)
    np.testing.assert_allclose(res.utci.values, b.utci(std0.t.values, std0.rh.values, std0.wind.values,
                                                         std0.t.values + 12))
    out = raster.write_geotiff(res.utci, tmp_path / "utci.tif")
    with rasterio.open(out) as src:
        np.testing.assert_allclose(src.read(1), res.utci.values.astype("float32"))
        assert src.crs.to_epsg() == 4326


def test_cli(tmp_path):
    from salutepy.raster.__main__ import main

    f = tmp_path / "in.nc"
    synthetic().to_netcdf(f)
    out = tmp_path / "res.nc"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert main([str(f), "-i", "utci,humidex,thi", "-o", str(out), "-w", "2"]) == 0
    res = xr.open_dataset(out)
    assert {"utci", "utci_class", "humidex", "thi_class"} <= set(res.data_vars)


def test_heatstress_indices_on_grid(tmp_path):
    from salutepy.indices import heatstress_np as hs
    from salutepy.indices.wbgt_fast import wbgt_bernard, wbgt_liljegren

    std = raster.standardize(synthetic())
    names = ["wbgt", "wbgt_bernard", "swbgt", "wbt_stull", "apparent_temp", "discomfort_index",
             "effective_temp", "heatindex_hs", "humidex_hs"]
    one = raster.compute(std, names, workers=1)
    two = raster.compute(std, names, workers=3, min_block=1)
    for v in one.data_vars:
        np.testing.assert_array_equal(one[v].values, two[v].values)
    t, td, rh, w, s = std.t.values, std.td.values, std.rh.values, std.wind.values, std.solar.values
    lat, lon = xr.broadcast(std.latitude, std.longitude)
    times = np.broadcast_to(std.time.values[:, None, None], t.shape)
    exp = wbgt_liljegren(t, td, w, s, times, np.broadcast_to(lon.values, t.shape), np.broadcast_to(lat.values, t.shape),
                         hour=True)["data"]
    np.testing.assert_allclose(one.wbgt.values, exp)
    np.testing.assert_allclose(one.wbgt_bernard.values, wbgt_bernard(t, td)["data"])
    np.testing.assert_allclose(one.swbgt.values, hs.swbgt(t, rh))
    assert "discomfort_index_class" in one and "effective_temp_class" in one
