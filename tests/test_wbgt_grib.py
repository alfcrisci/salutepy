"""wbgt_from_grib on a synthetic GRIB2 with the MOLOCH variables."""

import numpy as np
import pandas as pd
import pytest

from salutepy.indices.misc import reducewind
from salutepy.indices.wbgt_fast import wbgt_liljegren
from salutepy.pipeline.wbgt_grib import wbgt_from_grib

from .test_raster import _write_grib2, synthetic


def test_wbgt_from_grib(tmp_path):
    pytest.importorskip("cfgrib")
    ds = synthetic(nt=1).isel(time=0)
    f = tmp_path / "mol.grb2"
    when = pd.Timestamp(ds.time.values).to_pydatetime()
    # 2t, 2d, 10u, 10v e radiazione globale (ssrd come flusso: paramId 169 non va, uso dswrf di NCEP? -> uso 260087 sdswrf)
    _write_grib2(f, [(167, 2, "heightAboveGround", ds.t2m.values), (168, 2, "heightAboveGround", ds.d2m.values),
                     (165, 10, "heightAboveGround", ds.u10.values), (166, 10, "heightAboveGround", ds.v10.values),
                     (260087, 0, "surface", ds.sdswrf.values)], ds.latitude.values, ds.longitude.values, when)
    out = wbgt_from_grib(f, timesteps=None, workers=2)
    t = ds.t2m.values - 273.15
    td = ds.d2m.values - 273.15
    w2 = reducewind(np.hypot(ds.u10.values, ds.v10.values))
    lat, lon = np.meshgrid(ds.latitude.values, ds.longitude.values, indexing="ij")
    exp = wbgt_liljegren(t, td, w2, ds.sdswrf.values, np.datetime64(when, "s"), lon, lat, hour=True)
    np.testing.assert_allclose(out.wbgtsun.values[0], exp["data"], atol=1e-4)
    np.testing.assert_allclose(out.wbgtsha.values[0], 0.7 * exp["Tnwb"] + 0.3 * t, atol=1e-4)
