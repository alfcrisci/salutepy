"""
Checks the NumPy/numba ports against biometeo.js itself (run with Node.js).
Skipped if node is not installed.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from salutepy.indices import biometeo_np as b
from salutepy.indices.pet import pet_hoppe

JS = Path(__file__).resolve().parents[1] / "salutepy" / "data" / "js" / "biometeo.js"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

RNG = np.random.default_rng(42)
N = 4000


def run_js(fun, *cols):
    data = json.dumps([np.asarray(c, dtype=float).tolist() for c in cols])
    code = f"""
const fs=require('fs'),vm=require('vm');const c={{}};vm.createContext(c);
vm.runInContext(fs.readFileSync({json.dumps(str(JS))},'utf8'),c);
const a=JSON.parse(fs.readFileSync(0,'utf8'));
const out=a[0].map((_,i)=>{{const v=c.{fun}(...a.map(x=>x[i]));return (typeof v==='number'&&isFinite(v)&&Math.abs(v)!=9999)?v:null;}});
process.stdout.write(JSON.stringify(out));"""
    res = subprocess.run(["node", "-e", code], input=data, capture_output=True, text=True, check=True)
    return np.array([np.nan if v is None else v for v in json.loads(res.stdout)], dtype=float)


def check(py, js, tol=1e-9):
    py = np.asarray(py, dtype=float)
    assert np.array_equal(np.isnan(py), np.isnan(js))
    ok = ~np.isnan(js)
    assert np.max(np.abs(py[ok] - js[ok])) <= tol


T = RNG.uniform(-45, 55, N)
RH = RNG.uniform(1, 100, N)
W = RNG.uniform(0, 20, N)
TMRT = T + RNG.uniform(-35, 75, N)


def test_utci():
    check(b.utci(T, RH, W, TMRT), run_js("UTCI", T, RH, W, TMRT))


def test_heatindex():
    check(b.heatindex(T, RH), run_js("heatindex", T, RH))


def test_humidex():
    check(b.humidex(T, RH), run_js("humidex", T, RH))


def test_net():
    check(b.net_index(T, RH, W), run_js("net_index", T, RH, W))


def test_thi():
    check(b.thi(T, RH), run_js("THI", T, RH), 1e-12)


def test_windchill():
    check(b.windchill(T, W), run_js("windchill", T, W), 1e-9)


def test_mrt_globe():
    tg = T + RNG.uniform(-5, 25, N)
    d = RNG.uniform(0.02, 0.15, N)
    check(b.mrt_globe(T, tg, W, d, 0.95), run_js("mrt_globe", T, tg, W, d, np.full(N, 0.95)), 1e-7)


def test_mrt_solar():
    solar = RNG.uniform(0, 1000, N)
    elev = RNG.uniform(0, 90, N)
    check(b.mrt_solar_proj(T, RH, solar, elev), run_js("mrt_solar_proj", T, RH, solar, elev), 1e-7)


def test_pet():
    n = 1500
    t, rh, w = T[:n].clip(-30, 45), RH[:n], W[:n].clip(0, 12)
    tm = t + RNG.uniform(-15, 40, n)
    p = RNG.uniform(950, 1030, n)
    check(pet_hoppe(t, rh, w, tm, p), run_js("PET_Hoppe", t, rh, w, tm, p), 1e-9)


@pytest.mark.parametrize(
    "py_class,js_fun,args",
    [
        (lambda: b.utci_class(b.utci(T, RH, W, TMRT)), "utci_class", (T, RH, W, TMRT)),
        (lambda: b.heatindex_class(b.heatindex(T, RH)), "heatindex_class", (T, RH)),
        (lambda: b.humidex_class(b.humidex(T, RH)), "humidex_class", (T, RH)),
        (lambda: b.net_class(b.net_index(T, RH, W)), "net_class", (T, RH, W)),
        (lambda: b.thi_class(b.thi(T, RH)), "thi_class", (T, RH)),
        (lambda: b.windchill_class(b.windchill(T, W)), "windchill_cla", (T, W)),
    ],
)
def test_classes(py_class, js_fun, args):
    check(py_class(), run_js(js_fun, *args), 0)


def test_pet_class():
    p = RNG.uniform(-10, 50, N)
    check(b.pet_class(p), run_js("pet_class", p), 0)


def test_class_text():
    assert b.class_text([1, np.nan, 10], "utci", "en").tolist() == ["extreme cold stress", "", "extreme heat stress"]
