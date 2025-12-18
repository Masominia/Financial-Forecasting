"""Direct multi-horizon forecasting of a line item with a pseudo-real-time
rolling-origin evaluation.

Target at origin t and horizon h: the cumulative log growth of the item over
the next h quarters, y_t(h) = sum_{j=1..h} g_{t+j} (in %). Every model is
re-estimated at every origin on data available at t only (rows whose target
is fully observed by t); the regime feature is the *real-time* filtered
recession probability from a Hamilton model re-estimated up to t.

Models
------
naive (no change), mean (historical drift), ar (own lags), ardl (own lags +
macro drivers), regime_ardl (ardl + recession probability), gbm (shallow
gradient boosting on all features), ensemble (equal-weight average of ar,
regime_ardl and gbm).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

from .regimes import _refilter_path, hamilton_regimes

AR_FEATS = ["g0", "g1"]
ARDL_FEATS = AR_FEATS + ["gdp_g", "infl", "d_ust10y", "eq_ret", "eq_ret_4q"]
REGIME_FEATS = ARDL_FEATS + ["p_rec"]
MODELS = ["naive", "mean", "ar", "ardl", "regime_ardl", "gbm", "ensemble"]


def features(df: pd.DataFrame, item: str) -> pd.DataFrame:
    g = df[item]
    return pd.DataFrame(
        {
            "g0": g,
            "g1": g.shift(1),
            "g4q": g.rolling(4).sum(),
            "gdp_g": df["gdp_g"],
            "infl": df["infl"],
            "d_ust10y": df["d_ust10y"],
            "ust10y": df["ust10y"],
            "eq_ret": df["eq_ret"],
            "eq_ret_4q": df["eq_ret"].rolling(4).sum(),
        }
    )


def target(df: pd.DataFrame, item: str, h: int) -> pd.Series:
    g = df[item]
    return sum(g.shift(-j) for j in range(1, h + 1)).rename(f"y{h}")


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return b


def _ols_predict(b, x):
    return float(b[0] + np.asarray(x) @ b[1:])


def gbm_model(seed: int = 0) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_depth=2, learning_rate=0.05, max_iter=150, min_samples_leaf=12, l2_regularization=1.0, random_state=seed
    )


@dataclass
class Evaluation:
    item: str
    h: int
    preds: pd.DataFrame  # origin x model
    actual: pd.Series
    sigma: pd.DataFrame  # in-sample residual sd at each origin (for Gaussian intervals)
    p_rec: pd.Series

    def errors(self) -> pd.DataFrame:
        return self.preds.sub(self.actual, axis=0)


def rolling_forecasts(
    df: pd.DataFrame,
    item: str,
    h: int,
    first_origin: str,
    min_train: int = 60,
    refit_regime_every: int = 4,
    seed: int = 0,
    winsor: float | None = None,
) -> Evaluation:
    X_all = features(df, item)
    y_all = target(df, item, h)
    idx = df.index
    origins = idx[(idx >= pd.Period(first_origin, "Q")) & (np.arange(len(idx)) <= len(idx) - 1 - h)]
    preds, sig, prec = {}, {}, {}
    regime = None
    for k, t in enumerate(origins):
        hist = df.loc[:t]
        if regime is None or k % refit_regime_every == 0:
            regime = hamilton_regimes(hist["gdp_g"], search_reps=10)
            p_path = regime.filtered
        else:
            p_path = _refilter_path(regime, hist["gdp_g"])
        X = X_all.loc[:t].assign(p_rec=p_path)
        pos_t = idx.get_loc(t)
        train_rows = idx[: pos_t - h + 1]  # targets fully realised by t
        tr = X.loc[train_rows].join(y_all.loc[train_rows]).dropna()
        tr = tr.iloc[-max(len(tr), min_train) :]
        x_t = X.loc[t].copy()
        y_raw = tr[f"y{h}"].to_numpy()
        y_tr = y_raw.copy()
        if winsor is not None:  # outlier treatment estimated on the training window only
            lo, hi = np.quantile(y_tr, [winsor, 1 - winsor])
            y_tr = np.clip(y_tr, lo, hi)
            for c in ("g0", "g1", "g4q"):
                clo, chi = np.quantile(tr[c], [winsor, 1 - winsor])
                tr[c] = tr[c].clip(clo, chi)
                x_t[c] = float(np.clip(x_t[c], clo, chi))
        out = {"naive": 0.0, "mean": float(y_tr.mean())}
        s = {}
        for name, cols in (("ar", AR_FEATS), ("ardl", ARDL_FEATS), ("regime_ardl", REGIME_FEATS)):
            b = _ols(tr[cols].to_numpy(), y_tr)
            out[name] = _ols_predict(b, x_t[cols].to_numpy())
            fitted = b[0] + tr[cols].to_numpy() @ b[1:]
            s[name] = float(np.std(y_raw - fitted, ddof=len(cols) + 1))  # raw targets: honest residual scale
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = gbm_model(seed).fit(tr[REGIME_FEATS + ["g4q", "ust10y"]], y_tr)
        out["gbm"] = float(m.predict(x_t[REGIME_FEATS + ["g4q", "ust10y"]].to_frame().T)[0])
        s["gbm"] = float(np.std(y_raw - m.predict(tr[REGIME_FEATS + ["g4q", "ust10y"]])))
        out["ensemble"] = float(np.mean([out["ar"], out["regime_ardl"], out["gbm"]]))
        s["naive"] = float(np.sqrt(np.mean(y_raw**2)))
        s["mean"] = float(np.std(y_raw))
        s["ensemble"] = float(np.mean([s["ar"], s["regime_ardl"], s["gbm"]]))
        preds[t], sig[t], prec[t] = out, s, float(p_path.iloc[-1])
    P = pd.DataFrame(preds).T[MODELS]
    return Evaluation(item, h, P, y_all.loc[P.index], pd.DataFrame(sig).T[MODELS], pd.Series(prec))


# Evaluation
def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int) -> tuple[float, float]:
    """DM test of equal squared-error loss with the Harvey-Leybourne-Newbold
    small-sample correction; negative statistic = model 1 more accurate."""
    d = e1**2 - e2**2
    n = len(d)
    dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[: n - k] - dbar)) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return float("nan"), float("nan")
    dm = dbar / np.sqrt(var)
    hln = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(hln), float(2 * stats.t.sf(abs(hln), n - 1))


def score_table(ev: Evaluation, recession: pd.Series, start: str) -> pd.DataFrame:
    E = ev.errors().loc[pd.Period(start, "Q") :].dropna()
    rec = recession.reindex(E.index).fillna(0).astype(bool)
    # a forecast "covers" a recession if any quarter of its target window is in recession
    rows = {}
    base = E["naive"].to_numpy()
    for m in MODELS:
        e = E[m].to_numpy()
        dm, p = diebold_mariano(e, base, ev.h) if m != "naive" else (np.nan, np.nan)
        rows[m] = {
            "RMSE": np.sqrt(np.mean(e**2)),
            "MAE": np.mean(np.abs(e)),
            "RMSE vs naive": np.sqrt(np.mean(e**2)) / np.sqrt(np.mean(base**2)),
            "MAE, recession quarters": np.mean(np.abs(e[rec.to_numpy()])) if rec.any() else np.nan,
            "MAE, expansion quarters": np.mean(np.abs(e[~rec.to_numpy()])),
            "DM vs naive": dm,
            "DM p": p,
        }
    return pd.DataFrame(rows).T


def recession_window(recession: pd.Series, h: int) -> pd.Series:
    """1 if any quarter in (t, t + h] is a recession quarter."""
    r = recession.astype(int)
    return sum(r.shift(-j).fillna(0) for j in range(1, h + 1)).clip(0, 1)
