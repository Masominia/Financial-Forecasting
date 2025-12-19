"""Scenario-conditional projections of line items with Monte Carlo fan charts.

Conditional model (one per line item), quarterly growth in %:

    g_t = c + phi_1 g_{t-1} + phi_2 g_{t-2} + beta' x_t + e_t,
    x_t = (GDP growth, inflation, change in 10Y yield, equity return)

Coefficients are estimated by Huber M-estimation (robust to the 2008-09
write-down quarters); uncertainty comes from re-sampling the *raw* residuals
with a stationary block bootstrap, so crisis-sized shocks and their clustering
stay in the simulated distribution. The simulation kernel is written in Rust
(``rust/``) with a bit-for-bit identical NumPy fallback.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

X_COLS = ["gdp_g", "infl", "d_ust10y", "eq_ret"]

try:  # optional compiled engine
    import plancast_engine as _rust

    HAVE_RUST = True
except ImportError:  # pragma: no cover - exercised when the wheel is not built
    _rust = None
    HAVE_RUST = False


def simulate_arx_numpy(c, phi, beta, x_path, y_hist, resid, mean_block, uniforms, n, h) -> np.ndarray:
    """Reference implementation of the Rust kernel (same algorithm, same inputs)."""
    phi, beta, resid = np.asarray(phi, float), np.asarray(beta, float), np.asarray(resid, float)
    x = np.asarray(x_path, float).reshape(h, len(beta)) if len(beta) else np.zeros((h, 0))
    U = np.asarray(uniforms, float).reshape(n, h, 2)
    p, m = len(phi), len(resid)
    lags = np.tile(np.asarray(y_hist, float)[::-1], (n, 1)) if p else np.zeros((n, 0))
    out = np.zeros((n, h))
    idx = np.zeros(n, dtype=np.int64)
    for t in range(h):
        new = (U[:, t, 0] < 1.0 / mean_block) if t else np.ones(n, bool)
        start = np.minimum((U[:, t, 1] * m).astype(np.int64), m - 1)
        idx = np.where(new, start, (idx + 1) % m)
        y = c + resid[idx]  # same summation order as the Rust kernel -> bit-identical output
        for i in range(p):
            y = y + phi[i] * lags[:, i]
        for j in range(len(beta)):
            y = y + beta[j] * x[t, j]
        out[:, t] = y
        if p:
            lags = np.column_stack([y, lags[:, :-1]])
    return out


def simulate_arx(*args, engine: str = "auto") -> np.ndarray:
    n, h = args[8], args[9]
    if engine == "rust" or (engine == "auto" and HAVE_RUST):
        if not HAVE_RUST:
            raise RuntimeError("Rust engine not built: run `maturin develop -m rust/Cargo.toml`")
        a = list(args)
        for i in (1, 2, 3, 4, 5, 7):  # contiguous float64 vectors, borrowed by Rust without copying
            a[i] = np.ascontiguousarray(np.ravel(a[i]), dtype=np.float64)
        return _rust.simulate_arx(*a).reshape(n, h)
    return simulate_arx_numpy(*args)


def bootstrap_indices(U: np.ndarray, m: int, mean_block: float) -> np.ndarray:
    """Stationary-bootstrap residual indices implied by the uniforms U (n, h, 2):
    the same rule as the simulation kernel, so a second line item can re-use
    exactly the same historical dates (joint resampling keeps cross-correlation)."""
    n, h, _ = U.shape
    idx = np.zeros((n, h), dtype=np.int64)
    for t in range(h):
        start = np.minimum((U[:, t, 1] * m).astype(np.int64), m - 1)
        new = (U[:, t, 0] < 1.0 / mean_block) if t else np.ones(n, bool)
        idx[:, t] = np.where(new, start, (idx[:, t - 1] + 1) % m) if t else start
    return idx


@dataclass
class ConditionalModel:
    item: str
    c: float
    phi: np.ndarray
    beta: pd.Series
    resid: np.ndarray
    bse: pd.Series

    resid_index: pd.PeriodIndex | None = None

    @classmethod
    def fit(cls, df: pd.DataFrame, item: str) -> ConditionalModel:
        g = df[item]
        X = pd.concat([g.shift(1).rename("g_l1"), g.shift(2).rename("g_l2"), df[X_COLS]], axis=1)
        d = pd.concat([g.rename("y"), X], axis=1).dropna()
        res = sm.RLM(d["y"], sm.add_constant(d.drop(columns="y")), M=sm.robust.norms.HuberT()).fit()
        b = res.params
        fitted = sm.add_constant(d.drop(columns="y")) @ b
        return cls(
            item,
            float(b["const"]),
            b[["g_l1", "g_l2"]].to_numpy(),
            b[X_COLS],
            (d["y"] - fitted).to_numpy(),
            res.bse,
            d.index,
        )

    def residuals_on(self, dates) -> np.ndarray:
        return pd.Series(self.resid, index=self.resid_index).loc[dates].to_numpy()

    def project(
        self,
        df: pd.DataFrame,
        x_path: pd.DataFrame,
        n: int,
        mean_block: float,
        seed: int,
        engine: str = "auto",
        U: np.ndarray | None = None,
        resid: np.ndarray | None = None,
    ) -> np.ndarray:
        h = len(x_path)
        y_hist = df[self.item].iloc[-2:].to_numpy()  # oldest first
        U = np.random.default_rng(seed).random((n, h, 2)) if U is None else U
        resid = self.resid if resid is None else resid
        return simulate_arx(
            self.c,
            self.phi,
            self.beta.to_numpy(),
            x_path[X_COLS].to_numpy(),
            y_hist,
            resid,
            mean_block,
            U,
            n,
            h,
            engine=engine,
        )


@dataclass
class LintnerModel:
    """Dividends linked to earnings (Lintner, 1956): partial adjustment towards a
    target payout ratio, with lagged earnings growth and the lagged log payout
    ratio ln(D/E) as drivers,

        g^D_t = c + a1 g^D_{t-1} + a2 g^D_{t-2} + sum_{j=1..4} b_j g^E_{t-j} + k ln(D/E)_{t-1} + e_t.

    It is simulated *on top of* the simulated earnings paths, with residuals
    drawn on the same historical dates as the earnings shocks."""

    params: pd.Series
    tvalues: pd.Series
    resid: pd.Series

    @staticmethod
    def design(df: pd.DataFrame) -> pd.DataFrame:
        gD, gE = df["dividends"], df["earnings"]
        cols = {"gD_l1": gD.shift(1), "gD_l2": gD.shift(2)}
        cols.update({f"gE_l{j}": gE.shift(j) for j in range(1, 5)})
        cols["payout_l1"] = np.log(df["dividends_lvl"] / df["earnings_lvl"]).shift(1)
        return pd.DataFrame(cols)

    @classmethod
    def fit(cls, df: pd.DataFrame) -> LintnerModel:
        d = pd.concat([df["dividends"].rename("y"), cls.design(df)], axis=1).dropna()
        X = sm.add_constant(d.drop(columns="y"))
        res = sm.RLM(d["y"], X, M=sm.robust.norms.HuberT()).fit()
        return cls(res.params, res.tvalues, d["y"] - X @ res.params)

    def project(self, df: pd.DataFrame, eps_sims: np.ndarray, idx: np.ndarray, resid: np.ndarray) -> np.ndarray:
        """eps_sims: simulated earnings growth (n, h); idx: bootstrap indices (n, h)."""
        b = self.params
        n, h = eps_sims.shape
        gE = np.column_stack([np.tile(df["earnings"].iloc[-4:].to_numpy(), (n, 1)), eps_sims])  # (n, 4 + h)
        gD = np.tile(df["dividends"].iloc[-2:].to_numpy(), (n, 1))
        lnD = np.full(n, np.log(df["dividends_lvl"].iloc[-1]))
        lnE = np.full(n, np.log(df["earnings_lvl"].iloc[-1]))
        out = np.zeros((n, h))
        for t in range(h):
            y = (
                b["const"]
                + b["gD_l1"] * gD[:, -1]
                + b["gD_l2"] * gD[:, -2]
                + sum(b[f"gE_l{j}"] * gE[:, 4 + t - j] for j in range(1, 5))
                + b["payout_l1"] * (lnD - lnE)
                + resid[idx[:, t]]
            )
            out[:, t] = y
            gD = np.column_stack([gD[:, 1:], y])
            lnD = lnD + y / 100
            lnE = lnE + eps_sims[:, t] / 100
        return out


def scenario_paths(spec: dict, h: int) -> pd.DataFrame:
    out = {}
    for v in X_COLS:
        val = spec.get(v, 0.0)
        out[v] = np.full(h, float(val)) if np.isscalar(val) else np.asarray(val, float)[:h]
    return pd.DataFrame(out, index=pd.RangeIndex(1, h + 1, name="h"))


def summarise(sims: np.ndarray, level0: float) -> dict:
    """Level fan chart and drawdown statistics from simulated quarterly growth (in %)."""
    lvl = level0 * np.exp(np.cumsum(sims, axis=1) / 100)
    full = np.column_stack([np.full(len(lvl), level0), lvl])
    dd = (full / np.maximum.accumulate(full, axis=1)).min(axis=1) - 1
    q = np.quantile(lvl, [0.05, 0.5, 0.95], axis=0)
    return {
        "p5": q[0],
        "median": q[1],
        "p95": q[2],
        "median 9q change %": 100 * (q[1][-1] / level0 - 1),
        "median peak-to-trough %": 100 * np.median(dd),
        "5% worst peak-to-trough %": 100 * np.quantile(dd, 0.05),
    }
