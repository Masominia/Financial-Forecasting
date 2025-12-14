"""Regime changes and structural breaks.

* ``hamilton_regimes``: Hamilton (1989) two-state Markov-switching model of GDP
  growth (switching mean, common variance, impulse dummies for the 2020Q2-Q3
  lockdown outliers). Its *filtered* recession probability uses only data up
  to each quarter, so it can be used as a real-time forecasting feature.
* ``multiple_breaks``: Bai and Perron (2003) style least-squares estimation of
  m mean breaks by dynamic programming, with the number of breaks chosen by
  BIC (LWZ-type penalty) and a minimum segment length. Applied to |x - mean|
  it detects variance breaks (e.g. the Great Moderation).
* ``cusum``: Brown-Durbin-Evans recursive-residual CUSUM test for parameter
  stability of a linear model, with the 5% boundary.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

COVID = ("2020Q2", "2020Q3")


def _exog_coef(params: pd.Series, i: int) -> float:
    """Coefficient of the i-th exogenous dummy (statsmodels names it 'x{i+1}' or 'x{i+1}[k]')."""
    for name in (f"x{i + 1}", f"x{i + 1}[0]", f"x{i + 1}[1]"):
        if name in params.index:
            return float(params[name])
    return 0.0


def covid_dummies(index: pd.PeriodIndex) -> pd.DataFrame:
    return pd.DataFrame({f"d{q}": (index == pd.Period(q, "Q")).astype(float) for q in COVID}, index=index)


@dataclass
class RegimeFit:
    params: pd.Series
    filtered: pd.Series  # P(recession regime | data up to t)
    smoothed: pd.Series  # P(recession regime | all data)
    recession_mean: float
    expansion_mean: float
    p_stay_recession: float
    p_stay_expansion: float

    @property
    def expected_duration(self) -> tuple[float, float]:
        return 1 / (1 - self.p_stay_recession), 1 / (1 - self.p_stay_expansion)


def hamilton_regimes(gdp_g: pd.Series, seed: int = 0, search_reps: int = 30) -> RegimeFit:
    X = covid_dummies(gdp_g.index)
    X = X.loc[:, X.sum() > 0]
    np.random.seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sm.tsa.MarkovRegression(
            gdp_g, k_regimes=2, exog=X if X.shape[1] else None, switching_exog=False, switching_variance=False
        ).fit(disp=False, search_reps=search_reps)
    c = np.array([res.params["const[0]"], res.params["const[1]"]])
    lo, hi = int(np.argmin(c)), int(np.argmax(c))
    P = res.regime_transition[:, :, 0]  # P[i, j] = P(s_t = i | s_{t-1} = j)
    return RegimeFit(
        res.params,
        pd.Series(res.filtered_marginal_probabilities[lo], index=gdp_g.index),
        pd.Series(res.smoothed_marginal_probabilities[lo], index=gdp_g.index),
        float(c[lo]),
        float(c[hi]),
        float(P[lo, lo]),
        float(P[hi, hi]),
    )


def filtered_probability_realtime(gdp_g: pd.Series, origins, min_obs: int = 60, refit_every: int = 4) -> pd.Series:
    """Recession probability at each origin, re-estimating on data up to the origin only."""
    out, fit = {}, None
    for i, o in enumerate(origins):
        hist = gdp_g.loc[:o]
        if len(hist) < min_obs:
            continue
        if fit is None or i % refit_every == 0:
            fit = hamilton_regimes(hist, search_reps=10)
            out[o] = float(fit.filtered.iloc[-1])
        else:  # re-filter with fixed parameters on the extended sample
            out[o] = float(_refilter(fit, hist))
    return pd.Series(out, name="p_recession_rt")


def _refilter(fit: RegimeFit, y: pd.Series) -> float:
    """Hamilton filter with fixed parameters (no re-estimation): last probability."""
    return float(_refilter_path(fit, y).iloc[-1])


# Breaks
@dataclass
class Breaks:
    dates: list
    segment_means: list
    bic: dict


def multiple_breaks(x: pd.Series, max_breaks: int = 5, min_size: int = 16) -> Breaks:
    """Global least-squares break dates in the mean of x (dynamic programming)."""
    v = x.to_numpy(float)
    n = len(v)
    cs, cs2 = np.r_[0, np.cumsum(v)], np.r_[0, np.cumsum(v**2)]

    def ssr(i, j):  # segment v[i:j]
        m = j - i
        s = cs[j] - cs[i]
        return cs2[j] - cs2[i] - s * s / m

    INF = np.inf
    # cost[k][j]: min SSR of v[:j] with k breaks
    cost = np.full((max_breaks + 1, n + 1), INF)
    arg = np.zeros((max_breaks + 1, n + 1), dtype=int)
    for j in range(min_size, n + 1):
        cost[0, j] = ssr(0, j)
    for k in range(1, max_breaks + 1):
        for j in range((k + 1) * min_size, n + 1):
            best, bi = INF, -1
            for i in range(k * min_size, j - min_size + 1):
                c = cost[k - 1, i] + ssr(i, j)
                if c < best:
                    best, bi = c, i
            cost[k, j], arg[k, j] = best, bi
    bic = {}
    for k in range(max_breaks + 1):
        if np.isfinite(cost[k, n]):
            bic[k] = n * np.log(cost[k, n] / n) + (2 * k + 1) * np.log(n)  # k break dates + k + 1 means
    k_best = min(bic, key=bic.get)
    cuts, j = [], n
    for k in range(k_best, 0, -1):
        j = arg[k, j]
        cuts.append(j)
    cuts = sorted(cuts)
    bounds = [0, *cuts, n]
    means = [float(v[a:b].mean()) for a, b in zip(bounds[:-1], bounds[1:], strict=True)]
    return Breaks([x.index[c] for c in cuts], means, bic)


def cusum(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Recursive-residual CUSUM path, 5% boundary and whether it is crossed."""
    n, k = X.shape
    w = []
    for t in range(k + 1, n):
        Xt, yt = X[:t], y[:t]
        XtX_inv = np.linalg.pinv(Xt.T @ Xt)
        b = XtX_inv @ Xt.T @ yt
        f = 1 + X[t] @ XtX_inv @ X[t]
        w.append((y[t] - X[t] @ b) / np.sqrt(f))
    w = np.asarray(w)
    s = w.std(ddof=1)
    path = np.cumsum(w) / s
    m = len(w)
    r = np.arange(1, m + 1)
    bound = 0.948 * (np.sqrt(m) + 2 * r / np.sqrt(m))
    return path, bound, bool(np.any(np.abs(path) > bound))


def _refilter_path(fit: RegimeFit, y: pd.Series) -> pd.Series:
    """Filtered recession probabilities for every date of ``y`` with fixed parameters."""
    mu = np.array([fit.recession_mean, fit.expansion_mean])
    s2 = float(fit.params["sigma2"])
    X = covid_dummies(y.index)
    adj = y.to_numpy() - sum(_exog_coef(fit.params, i) * X.iloc[:, i].to_numpy() for i in range(X.shape[1]))
    P = np.array([[fit.p_stay_recession, 1 - fit.p_stay_expansion], [1 - fit.p_stay_recession, fit.p_stay_expansion]])
    p_rr, p_ee = fit.p_stay_recession, fit.p_stay_expansion
    xi = np.array([1 - p_ee, 1 - p_rr]) / (2 - p_rr - p_ee)  # ergodic start, as statsmodels
    out = np.empty(len(adj))
    for i, yt in enumerate(adj):
        pred = P @ xi
        lik = np.exp(-0.5 * (yt - mu) ** 2 / s2)
        xi = pred * lik / max(np.sum(pred * lik), 1e-300)
        out[i] = xi[0]
    return pd.Series(out, index=y.index)
