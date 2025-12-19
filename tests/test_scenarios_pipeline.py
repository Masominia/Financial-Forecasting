import json

import numpy as np
import pandas as pd
import pytest

from plancast import data
from plancast.pipeline import run
from plancast.scenarios import (
    HAVE_RUST,
    ConditionalModel,
    LintnerModel,
    bootstrap_indices,
    scenario_paths,
    simulate_arx,
    simulate_arx_numpy,
    summarise,
)


def _args(n=300, h=9, seed=0):
    rng = np.random.default_rng(seed)
    return (
        0.1,
        np.array([0.6, -0.1]),
        np.array([0.3, 0.0, 1.4, 0.07]),
        rng.standard_normal((h, 4)),
        np.array([1.0, 2.0]),
        rng.standard_normal(150),
        4.0,
        rng.random((n, h, 2)),
        n,
        h,
    )


def test_numpy_kernel_deterministic_ar1():
    out = simulate_arx_numpy(1.0, [0.5], [], np.zeros((3, 0)), [0.0], [0.0], 4.0, np.full((1, 3, 2), 0.1), 1, 3)
    np.testing.assert_allclose(out[0], [1.0, 1.5, 1.75])


@pytest.mark.skipif(not HAVE_RUST, reason="Rust engine not built")
def test_rust_kernel_is_bit_identical():
    a = _args()
    assert np.array_equal(simulate_arx(*a, engine="rust"), simulate_arx(*a, engine="numpy"))


def test_bootstrap_indices_match_kernel():
    a = list(_args())
    m = 150
    a[0], a[1], a[2], a[3], a[4], a[5] = (
        0.0,
        np.zeros(0),
        np.zeros(0),
        np.zeros((9, 0)),
        np.zeros(0),
        np.arange(m, dtype=float),
    )
    idx = simulate_arx(*a).astype(np.int64)
    assert np.array_equal(idx, bootstrap_indices(a[7], m, 4.0))
    # stationary bootstrap: mean block length close to 4
    cont = (np.diff(idx, axis=1) == 1).mean()
    assert 0.70 < cont < 0.80


def test_conditional_models_fit_and_project():
    df = data.load_panel()
    cm = ConditionalModel.fit(df, "earnings")
    assert cm.beta["gdp_g"] > 0
    xp = scenario_paths({"gdp_g": -8.0, "eq_ret": -20.0}, 4)
    bad = cm.project(df, xp, 2000, 4.0, 1)
    good = cm.project(df, scenario_paths({"gdp_g": 3.0, "eq_ret": 3.0}, 4), 2000, 4.0, 1)
    assert np.median(bad.sum(1)) < np.median(good.sum(1))
    lint = LintnerModel.fit(df)
    assert lint.params["payout_l1"] < 0 and lint.params["gE_l1"] > 0
    U = np.random.default_rng(0).random((500, 4, 2))
    common = lint.resid.index.intersection(cm.resid_index)
    eps = cm.project(df, xp, 500, 4.0, 0, U=U, resid=cm.residuals_on(common))
    dps = lint.project(df, eps, bootstrap_indices(U, len(common), 4.0), np.zeros(len(common)))
    assert dps.shape == (500, 4) and np.isfinite(dps).all()


def test_summarise():
    s = summarise(np.array([[-10.0, 0.0, 10.0]]), 100.0)
    assert s["median 9q change %"] == pytest.approx(0.0, abs=1e-9)
    assert s["median peak-to-trough %"] == pytest.approx(100 * (np.exp(-0.1) - 1))


def test_pipeline_smoke(tmp_path):
    m = run(out=tmp_path, fast=True)
    for f in ("results.md", "monitoring.md", "metrics.json"):
        assert (tmp_path / f).exists()
    assert len(list((tmp_path / "figures").glob("*.png"))) == 7
    j = json.loads((tmp_path / "metrics.json").read_text())
    assert not [c for c in j["checks"] if c["status"] == "FAIL"]
    assert m["regimes"]["auc_full"] > 0.95
    assert isinstance(pd.DataFrame(j["scenarios"]), pd.DataFrame)
