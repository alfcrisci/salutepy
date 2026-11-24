"""
Checks heatstress_np and wbgt_fast against the R package HeatStress (run with Rscript).
Skipped if Rscript or the HeatStress package are not available. Set HEATSTRESS_RLIB to the R
library where HeatStress is installed, if it is not in the default library.
"""

import os
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
import pytest

from salutepy.indices import heatstress_np as hs
from salutepy.indices.wbgt_fast import wbgt_bernard, wbgt_liljegren

RLIB = os.environ.get("HEATSTRESS_RLIB", "")


def _have_r():
    if shutil.which("Rscript") is None:
        return False
    lib = f'.libPaths(c("{RLIB}", .libPaths()));' if RLIB else ""
    r = subprocess.run(["Rscript", "-e", lib + 'cat(requireNamespace("HeatStress", quietly=TRUE))'],
                       capture_output=True, text=True)
    return r.stdout.strip().endswith("TRUE")


pytestmark = pytest.mark.skipif(not _have_r(), reason="Rscript with HeatStress not available")


def run_r(expr, df):
    with tempfile.TemporaryDirectory() as d:
        fin, fout = os.path.join(d, "in.csv"), os.path.join(d, "out.csv")
        df.to_csv(fin, index=False)
        lib = f'.libPaths(c("{RLIB}", .libPaths()));' if RLIB else ""
        code = (lib + f'suppressMessages(library(HeatStress)); d <- read.csv("{fin}"); '
                f'if ("dates" %in% names(d)) d$dates <- as.POSIXct(d$dates, tz="UTC"); '
                f'r <- {expr}; write.csv(data.frame(r = r), "{fout}", row.names = FALSE)')
        subprocess.run(["Rscript", "-e", code], check=True, capture_output=True, text=True)
        return pd.read_csv(fout)["r"].to_numpy(dtype=float)


RNG = np.random.default_rng(7)
N = 3000
TAS = RNG.uniform(-25, 45, N)
HURS = RNG.uniform(3, 100, N)
WIND = RNG.uniform(0, 15, N)
DF = pd.DataFrame({"tas": TAS, "hurs": HURS, "wind": WIND})


def close(a, b, tol):
    a = np.asarray(a, dtype=float)
    assert np.array_equal(np.isnan(a), np.isnan(b))
    ok = ~np.isnan(b)
    assert np.max(np.abs(a[ok] - b[ok])) <= tol


@pytest.mark.parametrize("py,r", [
    (lambda: hs.tashurs2vap_pres(TAS, HURS), "HeatStress:::tashurs2vap.pres(d$tas, d$hurs)"),
    (lambda: hs.tashurs2dewp(TAS, HURS), "HeatStress:::tashurs2dewp(d$tas, d$hurs)"),
    (lambda: hs.swbgt(TAS, HURS), "swbgt(d$tas, d$hurs)"),
    (lambda: hs.wbt_stull(TAS, HURS), "wbt.Stull(d$tas, d$hurs)"),
    (lambda: hs.apparent_temp(TAS, HURS, WIND), "apparentTemp(d$tas, d$hurs, d$wind)"),
    (lambda: hs.discomfort_index(TAS, HURS), "discomInd(d$tas, d$hurs)"),
    (lambda: hs.effective_temp(TAS, HURS, WIND), "effectiveTemp(d$tas, d$hurs, d$wind)"),
    (lambda: hs.hi(TAS, HURS), "hi(d$tas, d$hurs)"),
    (lambda: hs.humidex(TAS, HURS), "humidex(d$tas, d$hurs)"),
])
def test_simple_indices(py, r):
    close(py(), run_r(r, DF), 1e-9)


def test_dewp2hurs():
    dewp = TAS - RNG.uniform(-2, 20, N)
    close(hs.dewp2hurs(TAS, dewp), run_r("HeatStress:::dewp2hurs(d$tas, d$dewp)", DF.assign(dewp=dewp)), 1e-9)


def test_wbgt_bernard():
    dewp = TAS - RNG.uniform(-1, 20, N)
    py = wbgt_bernard(TAS, dewp)["data"]
    close(py, run_r("wbgt.Bernard(d$tas, d$dewp)$data", DF.assign(dewp=dewp)), 1e-6)


@pytest.mark.parametrize("hour", [False, True])
def test_wbgt_liljegren(hour):
    n = 400
    tas = RNG.uniform(-5, 42, n)
    dewp = tas - RNG.uniform(-1, 20, n)
    wind = RNG.uniform(0, 10, n)
    rad = RNG.uniform(0, 1000, n)
    dates = pd.Timestamp("2026-07-15") + pd.to_timedelta(RNG.integers(0, 24 * 60, n), unit="min") \
        + pd.to_timedelta(RNG.integers(0, 200, n), unit="D")
    df = pd.DataFrame({"tas": tas, "dewp": dewp, "wind": wind, "rad": rad,
                       "dates": dates.strftime("%Y-%m-%d %H:%M:%S")})
    py = wbgt_liljegren(tas, dewp, wind, rad, dates.values, 11.2, 43.8, hour=hour)
    h = "TRUE" if hour else "FALSE"
    r_w = run_r(f"wbgt.Liljegren(d$tas, d$dewp, d$wind, d$rad, d$dates, 11.2, 43.8, hour={h})$data", df)
    close(py["data"], r_w, 1e-6)
