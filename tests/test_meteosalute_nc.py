"""Funzioni aggiunte per i NetCDF completi: PMV/clo compilati, frostime, posizione del sole."""

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from salutepy.indices import biometeo_np as B
from salutepy.indices import comfort_fast as CF
from salutepy.indices.sun_position import sunposition_rbiometeo

JS = Path(__file__).resolve().parents[1] / "salutepy" / "data" / "js" / "biometeo.js"


def _node(expr_rows):
    if shutil.which("node") is None:
        pytest.skip("node non disponibile")
    code = (
        "const fs=require('fs'),vm=require('vm');const c={Math,Date,isNaN,console};vm.createContext(c);"
        f"vm.runInContext(fs.readFileSync({json.dumps(str(JS))},'utf8'),c);"
        f"const rows={json.dumps(expr_rows)};"
        "console.log(JSON.stringify(rows.map(r=>[c.PMV_ISO7730(...r,1.1),c.clomin_7730(...r),c.clomax_7730(...r),"
        "c.frostime(r[0],r[2])])));"
    )
    out = subprocess.run(["node", "-e", code], capture_output=True, text=True, check=True).stdout
    return np.array([[np.nan if (x is None or isinstance(x, str) or x == 9999) else x for x in r]
                     for r in json.loads(out)], float)


def test_pmv_clo_frostime_vs_js():
    rng = np.random.default_rng(0)
    n = 400
    t = rng.uniform(-40, 35, n)
    rh = rng.uniform(5, 100, n)
    w = rng.uniform(0.1, 15, n)
    tr = t + rng.uniform(-5, 25, n)
    M = rng.uniform(58, 160, n)
    rows = [[float(a), float(b), float(c), float(d), float(e), 0.0] for a, b, c, d, e in zip(t, rh, w, tr, M)]
    ref = _node(rows)
    np.testing.assert_allclose(CF.pmv_iso7730(t, rh, w, tr, M, 0, 1.1), ref[:, 0], atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(CF.clomin_7730(t, rh, w, tr, M, 0), ref[:, 1], atol=1e-9)
    np.testing.assert_allclose(CF.clomax_7730(t, rh, w, tr, M, 0), ref[:, 2], atol=1e-9)
    np.testing.assert_allclose(B.frostime(t, w), ref[:, 3], atol=1e-9, equal_nan=True)


def test_sunposition_noon():
    # equinozio, mezzogiorno a Greenwich sull'equatore: sole quasi allo zenit
    el = sunposition_rbiometeo(datetime(2026, 3, 20, 12, 7), np.array([0.0, 45.0]), np.array([0.0, 0.0]))
    assert el[0] > 89.0 and 44.0 < el[1] < 46.0
