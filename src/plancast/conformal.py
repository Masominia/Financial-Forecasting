"""Prediction intervals with calibrated coverage for time series.

Given point forecasts made at origins t (target realised at t + h), three
interval constructions at nominal level 1 - alpha:

* ``gaussian``: forecast +/- z * in-sample residual standard deviation;
* ``split``: rolling split-conformal, the finite-sample-corrected (1 - alpha)
  quantile of the last W *realised* out-of-sample absolute errors;
* ``aci``: adaptive conformal inference (Gibbs and Candes, 2021). When the
  working level exceeds 1 the exact interval is unbounded; widths then count it
  as 1.5 x the largest calibration error. ACI: the
  quantile level is updated online, alpha_{k+1} = alpha_k + gamma (alpha - err_k),
  so coverage self-corrects under distribution shift. With horizon h an error
  is only known h quarters after the forecast, so updates use delayed feedback.

Only information available at the origin is used: the calibration set at t
contains errors of forecasts made at origins <= t - h.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _conformal_quantile(scores: np.ndarray, level: float) -> float:
    """k-th smallest score with k = ceil((n + 1) * level): the finite-sample-valid
    conformal quantile. When k > n the exact interval is unbounded; it is
    reported as 1.5 x the largest score to keep widths finite."""
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    k = int(np.ceil((n + 1) * level))
    if level >= 1.0 or k > n:
        return float(s[-1]) * 1.5
    return float(s[max(k, 1) - 1])


def intervals(
    pred: pd.Series,
    actual: pd.Series,
    sigma: pd.Series,
    h: int,
    alpha: float = 0.1,
    window: int = 40,
    gamma: float = 0.05,
    min_calib: int = 20,
) -> pd.DataFrame:
    idx = pred.index
    err = (actual - pred).abs()
    z = stats.norm.ppf(1 - alpha / 2)
    rows = []
    a_t = alpha
    pending = []  # (realisation position, covered flag) for delayed ACI updates
    for k, t in enumerate(idx):
        # ACI updates for forecasts whose outcome is known at t (made at origins <= t - h)
        still = []
        for pos, cov in pending:
            if pos <= k:
                a_t = a_t + gamma * (alpha - (0.0 if cov else 1.0))
            else:
                still.append((pos, cov))
        pending = still
        calib = err.iloc[max(0, k - h - window + 1) : max(0, k - h + 1)].dropna().to_numpy()
        row = {"origin": t, "pred": pred.iloc[k], "actual": actual.iloc[k], "gauss_hw": z * sigma.iloc[k]}
        if len(calib) >= min_calib:
            row["split_hw"] = _conformal_quantile(calib, 1 - alpha)
            row["aci_hw"] = _conformal_quantile(calib, min(max(1 - a_t, 0.0), 1.0))
            row["aci_alpha"] = a_t
            covered = abs(actual.iloc[k] - pred.iloc[k]) <= row["aci_hw"]
            pending.append((k + h, covered))
        rows.append(row)
    df = pd.DataFrame(rows).set_index("origin")
    for m in ("gauss", "split", "aci"):
        if f"{m}_hw" in df:
            df[f"{m}_cover"] = (df["actual"] - df["pred"]).abs() <= df[f"{m}_hw"]
    return df


def coverage_table(iv: pd.DataFrame, rec_window: pd.Series, start: str) -> pd.DataFrame:
    d = iv.loc[pd.Period(start, "Q") :].dropna(subset=["split_hw"])
    rec = rec_window.reindex(d.index).fillna(0).astype(bool)
    out = {}
    for m, lab in (("gauss", "Gaussian"), ("split", "Split conformal"), ("aci", "Adaptive conformal (ACI)")):
        c = d[f"{m}_cover"].astype(float)
        out[lab] = {
            "coverage": c.mean(),
            "coverage, recession windows": c[rec].mean() if rec.any() else np.nan,
            "coverage, expansions": c[~rec].mean(),
            "mean width": 2 * d[f"{m}_hw"].mean(),
            "median width": 2 * d[f"{m}_hw"].median(),
            "worst 20q rolling coverage": c.rolling(20).mean().min(),
        }
    return pd.DataFrame(out).T
