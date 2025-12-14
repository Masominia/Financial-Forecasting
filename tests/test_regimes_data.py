import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from plancast import data
from plancast.regimes import _refilter_path, cusum, hamilton_regimes, multiple_breaks


@pytest.fixture(scope="module")
def panel():
    return data.load_panel()


def test_panel(panel):
    assert str(panel.index[0]) == "1960Q1" and str(panel.index[-1]) == "2023Q2"
    assert not panel[data.ITEMS + data.DRIVERS].isna().any().any()
    # growth is the log change of the level
    g = 100 * np.log(panel["earnings_lvl"]).diff()
    np.testing.assert_allclose(g.iloc[1:], panel["earnings"].iloc[1:])
    # 2008Q4: the write-down quarter
    assert panel.loc["2008Q4", "earnings"] < -100


def test_nber_flags(panel):
    r = panel["recession"]
    assert r.loc["2008Q1":"2009Q2"].all() and not r.loc["2007Q4"] and not r.loc["2009Q3"]
    assert r.loc["2020Q1":"2020Q2"].all() and r.sum() == 32


def test_hamilton_matches_nber(panel):
    fit = hamilton_regimes(panel["gdp_g"])
    assert fit.recession_mean < 0 < fit.expansion_mean
    assert roc_auc_score(panel["recession"], fit.filtered) > 0.95
    # our fixed-parameter filter reproduces statsmodels' filtered probabilities
    np.testing.assert_allclose(_refilter_path(fit, panel["gdp_g"]).to_numpy(), fit.filtered.to_numpy(), atol=1e-6)


def test_multiple_breaks_recovers_known_dates():
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(0, 1, 60), rng.normal(3, 1, 50), rng.normal(-1, 1, 70)]
    s = pd.Series(x, index=pd.period_range("1950Q1", periods=len(x), freq="Q"))
    b = multiple_breaks(s, 4, 12)
    pos = [s.index.get_loc(d) for d in b.dates]
    assert len(pos) == 2 and abs(pos[0] - 60) <= 2 and abs(pos[1] - 110) <= 2
    assert multiple_breaks(pd.Series(rng.normal(0, 1, 150), index=s.index[:150]), 4, 12).dates == []


def test_cusum_detects_break_only_when_present():
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(200), rng.normal(size=200)])
    y_stable = X @ [0.5, 1.0] + rng.normal(0, 1, 200)
    y_break = y_stable + np.r_[np.zeros(120), np.full(80, 2.0)]
    assert not cusum(y_stable, X)[2]
    assert cusum(y_break, X)[2]
