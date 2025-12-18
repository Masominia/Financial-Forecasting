import numpy as np
import pandas as pd
import pytest

from plancast import data
from plancast.conformal import _conformal_quantile, intervals
from plancast.forecast import diebold_mariano, recession_window, rolling_forecasts, target


@pytest.fixture(scope="module")
def panel():
    return data.load_panel()


def test_target_is_sum_of_future_growth(panel):
    y = target(panel, "dividends", 4)
    t = panel.index.get_loc(pd.Period("2010Q1", "Q"))
    assert y.iloc[t] == pytest.approx(panel["dividends"].iloc[t + 1 : t + 5].sum())


def test_no_look_ahead(panel):
    """Changing data after the origin must not change the forecast made at the origin."""
    ev1 = rolling_forecasts(panel.loc[:"2012Q4"], "dividends", 1, "2012Q1", winsor=0.025)
    shocked = panel.copy()
    shocked.loc["2012Q3":, ["dividends", "gdp_g", "eq_ret"]] *= 5.0
    ev2 = rolling_forecasts(shocked.loc[:"2012Q4"], "dividends", 1, "2012Q1", winsor=0.025)
    # identical up to the optimiser tolerance of the regime model; a leak would move them by orders of magnitude
    np.testing.assert_allclose(
        ev1.preds.loc["2012Q1":"2012Q2"].to_numpy(), ev2.preds.loc["2012Q1":"2012Q2"].to_numpy(), rtol=1e-4, atol=1e-6
    )


def test_diebold_mariano_sign_and_size():
    rng = np.random.default_rng(0)
    good, bad = rng.normal(0, 1, 300), rng.normal(0, 2, 300)
    stat, p = diebold_mariano(good, bad, 1)
    assert stat < 0 and p < 0.001
    stat2, p2 = diebold_mariano(rng.normal(0, 1, 300), rng.normal(0, 1, 300), 1)
    assert p2 > 0.01


def test_recession_window():
    r = pd.Series([0, 0, 1, 0, 0], index=pd.period_range("2000Q1", periods=5, freq="Q"))
    assert recession_window(r, 2).tolist() == [1, 1, 0, 0, 0]


def test_conformal_quantile_finite_sample():
    s = np.arange(1, 11, dtype=float)
    assert _conformal_quantile(s, 0.9) == 10.0  # k = ceil(11 * 0.9) = 10
    assert _conformal_quantile(s, 0.5) == 6.0  # k = ceil(5.5) = 6
    assert _conformal_quantile(s, 0.95) == 15.0  # k = 11 > n: unbounded, reported as 1.5 x max


def _series(x):
    return pd.Series(x, index=pd.period_range("1950Q1", periods=len(x), freq="Q"))


def test_conformal_coverage_iid_and_aci_under_shift():
    rng = np.random.default_rng(3)
    n = 400
    actual = _series(np.r_[rng.normal(0, 1, 200), rng.normal(0, 3, 200)])  # volatility triples halfway
    pred = _series(np.zeros(n))
    sigma = _series(np.ones(n))
    iv = intervals(pred, actual, sigma, h=1, alpha=0.1, window=40, gamma=0.05).dropna(subset=["split_hw"])
    late = iv.iloc[220:]
    assert late["gauss_cover"].mean() < 0.6  # fixed-scale intervals collapse
    assert late["aci_cover"].mean() > 0.85  # ACI recovers nominal coverage
    calm = iv.iloc[40:190]
    assert 0.83 < calm["split_cover"].mean() < 0.97
