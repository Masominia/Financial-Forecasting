"""End-to-end run: regimes and breaks -> rolling-origin forecasts -> intervals ->
explanations -> live forecast -> scenarios -> monitoring -> report."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.metrics import roc_auc_score

from . import data, plotting
from .conformal import _conformal_quantile, coverage_table, intervals
from .forecast import REGIME_FEATS, _ols, features, gbm_model, recession_window, rolling_forecasts, score_table, target
from .regimes import cusum, filtered_probability_realtime, hamilton_regimes, multiple_breaks
from .report import md_table
from .scenarios import (
    HAVE_RUST,
    ConditionalModel,
    LintnerModel,
    bootstrap_indices,
    scenario_paths,
    simulate_arx,
    summarise,
)

LABEL = {"earnings": "Earnings (EPS)", "dividends": "Dividends (DPS)"}


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (pd.Period, pd.Timestamp)):
        return str(o)
    if hasattr(o, "item"):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def run(cfg_path=data.ROOT / "configs" / "default.yaml", out=data.ROOT / "reports", fast: bool = False) -> dict:
    cfg = yaml.safe_load(open(cfg_path))
    fc = cfg["forecast"]
    if fast:
        fc = {**fc, "first_origin": "2008Q1", "evaluation_start": "2014Q1"}
        cfg["scenarios"]["paths"] = 2000
    out = Path(out)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    df = data.load_panel(cfg["sample"]["start"], cfg["sample"]["end"])
    md, metrics, checks = [], {}, []

    def check(area, name, value, threshold, ok, warn_only=False):
        checks.append(
            {
                "area": area,
                "check": name,
                "value": value,
                "threshold": threshold,
                "status": "PASS" if ok else ("WARN" if warn_only else "FAIL"),
            }
        )

    # 1. Regimes and breaks
    reg = hamilton_regimes(df["gdp_g"])
    auc_full = roc_auc_score(df["recession"], reg.filtered)
    origins = df.index[df.index >= pd.Period(fc["evaluation_start"], "Q")]
    p_rt = filtered_probability_realtime(df["gdp_g"], origins)
    auc_rt = roc_auc_score(df["recession"].loc[p_rt.index], p_rt)
    g = df["gdp_g"].drop([pd.Period(q, "Q") for q in ("2020Q2", "2020Q3")])
    vb = multiple_breaks((g - g.mean()).abs(), cfg["breaks"]["max_breaks"], cfg["breaks"]["min_size"])
    eb = multiple_breaks(
        (df["earnings"] - df["earnings"].mean()).abs(), cfg["breaks"]["max_breaks"], cfg["breaks"]["min_size"]
    )
    vol_segments = [
        float(np.std(g.loc[a:b]))
        for a, b in zip([g.index[0], *vb.dates], [*[d - 1 for d in vb.dates], g.index[-1]], strict=True)
    ]
    plotting.regimes(df, reg.filtered, p_rt, vb.dates, figs / "regimes.png")
    d_rec, d_exp = reg.expected_duration
    metrics["regimes"] = {
        "recession_mean": reg.recession_mean,
        "expansion_mean": reg.expansion_mean,
        "auc_full": auc_full,
        "auc_realtime": auc_rt,
        "gdp_vol_breaks": vb.dates,
        "earnings_vol_breaks": eb.dates,
    }
    md += [
        "## 1. Regimes and structural breaks",
        "",
        f"Panel {df.index[0]} to {df.index[-1]} ({len(df)} quarters).",
        "",
        "**Hamilton (1989) Markov-switching model of real GDP growth** (switching mean, impulse dummies for "
        "2020Q2-Q3):",
        "",
        "| | recession regime | expansion regime |",
        "|---|---|---|",
        f"| mean growth, % ann. | {reg.recession_mean:.2f} | {reg.expansion_mean:.2f} |",
        f"| expected duration, quarters | {d_rec:.1f} | {d_exp:.1f} |",
        "",
        f"Agreement with NBER recession dates (AUC): **{auc_full:.3f}** with full-sample parameters, "
        f"**{auc_rt:.3f}** in real time over {p_rt.index[0]} to {p_rt.index[-1]} (parameters re-estimated with "
        "data available at each quarter only).",
        "",
        f"**Variance breaks in GDP growth** (Bai-Perron dynamic programming, BIC): {', '.join(map(str, vb.dates))}; "
        f"growth volatility by segment {', '.join(f'{v:.2f}' for v in vol_segments)} pp (the Great Moderation). "
        f"**Variance breaks in earnings growth**: {', '.join(map(str, eb.dates))} (the 2008-09 write-down era).",
        "",
    ]
    check("Regimes", "Real-time recession probability vs NBER (AUC)", f"{auc_rt:.3f}", ">= 0.85", auc_rt >= 0.85)

    # 2. Rolling-origin forecasts
    evs, tables = {}, {}
    for item in fc["items"]:
        for h in fc["horizons"]:
            ev = rolling_forecasts(
                df, item, h, fc["first_origin"], fc["min_train"], seed=cfg["seed"], winsor=fc["winsor"]
            )
            evs[(item, h)] = ev
            tables[(item, h)] = score_table(ev, recession_window(df["recession"], h), fc["evaluation_start"])
    plotting.accuracy(
        {f"{LABEL[i].split()[0]}, h={h}q": t for (i, h), t in tables.items()}, figs / "forecast_accuracy.png"
    )
    metrics["forecast"] = {f"{i}_h{h}": t.reset_index().to_dict(orient="records") for (i, h), t in tables.items()}
    md += [
        "## 2. Forecast accuracy (pseudo real time)",
        "",
        f"Direct forecasts of cumulative log growth over the next h quarters; every model is re-estimated at "
        f"every origin from {fc['first_origin']} with data available at that date. Scores from "
        f"{fc['evaluation_start']}. DM = Diebold-Mariano (Harvey-Leybourne-Newbold) test against the no-change "
        "forecast; negative = better. Recession columns use forecasts whose target window contains an NBER "
        "recession quarter.",
        "",
    ]
    for (item, h), t in tables.items():
        md += [f"**{LABEL[item]}, h = {h} quarter{'s' if h > 1 else ''}**", "", md_table(t, 3), ""]

    # 3. Intervals
    pm = fc["production_model"]
    ic = cfg["intervals"]
    cov, ivs = {}, {}
    for (item, h), ev in evs.items():
        iv = intervals(ev.preds[pm], ev.actual, ev.sigma[pm], h, ic["alpha"], ic["window"], ic["gamma"])
        ivs[(item, h)] = iv
        cov[(item, h)] = coverage_table(iv, recession_window(df["recession"], h), fc["evaluation_start"])
    plotting.coverage(
        {f"{LABEL[i]}, h={h}q": ivs[(i, h)] for i, h in [("earnings", 1), ("dividends", 1)]},
        fc["evaluation_start"],
        ic["alpha"],
        figs / "coverage.png",
    )
    plotting.interval_band(
        ivs[("earnings", 1)],
        df["recession"],
        fc["evaluation_start"],
        "Earnings: next-quarter forecasts with adaptive conformal 90% intervals",
        figs / "earnings_intervals.png",
    )
    metrics["coverage"] = {f"{i}_h{h}": c.reset_index().to_dict(orient="records") for (i, h), c in cov.items()}
    md += [
        "## 3. Uncertainty: 90% prediction intervals",
        "",
        f"Production model `{pm}`. Gaussian = +/- 1.645 in-sample residual sd; split conformal = rolling "
        f"{ic['window']}-quarter calibration of realised out-of-sample errors; ACI = adaptive conformal "
        f"inference (Gibbs and Candes, 2021), step {ic['gamma']}, with delayed feedback for h > 1.",
        "",
    ]
    for (item, h), c in cov.items():
        md += [f"**{LABEL[item]}, h = {h}**", "", md_table(c, 3), ""]
        aci = c.loc["Adaptive conformal (ACI)", "coverage"]
        check(
            "Intervals",
            f"ACI coverage, {item} h={h}",
            f"{aci:.3f}",
            "0.85 to 0.95",
            0.85 <= aci <= 0.95,
            warn_only=True,
        )

    # 4. Live forecast + explanations
    live, contribs = {}, {}
    p_full = reg.filtered
    for item in fc["items"]:
        h = 4
        X = features(df, item).assign(p_rec=p_full)
        y = target(df, item, h)
        tr = X.join(y).dropna()
        y_raw = tr[f"y{h}"].to_numpy()
        lo, hi = np.quantile(y_raw, [fc["winsor"], 1 - fc["winsor"]])
        y_w = np.clip(y_raw, lo, hi)
        trw = tr.copy()
        x_last = X.iloc[-1].copy()
        for c in ("g0", "g1", "g4q"):
            clo, chi = np.quantile(tr[c], [fc["winsor"], 1 - fc["winsor"]])
            trw[c] = tr[c].clip(clo, chi)
            x_last[c] = float(np.clip(x_last[c], clo, chi))
        b = _ols(trw[REGIME_FEATS].to_numpy(), y_w)
        means = trw[REGIME_FEATS].mean()
        point = float(b[0] + x_last[REGIME_FEATS].to_numpy() @ b[1:])
        c = pd.Series(b[1:] * (x_last[REGIME_FEATS] - means).to_numpy(), index=REGIME_FEATS)
        base = float(b[0] + means.to_numpy() @ b[1:])
        names = {
            "g0": "growth, last q",
            "g1": "growth, q-1",
            "gdp_g": "GDP growth",
            "infl": "inflation",
            "d_ust10y": "10Y change",
            "eq_ret": "equities, last q",
            "eq_ret_4q": "equities, 4q",
            "p_rec": "recession prob.",
        }
        contribs[f"{LABEL[item]}: 4-quarter forecast from {df.index[-1]}"] = pd.concat(
            [pd.Series({"average growth": base}), c.rename(index=names), pd.Series({"forecast": point})]
        )
        # Interval for the live forecast: realised errors up to now, current ACI level
        iv = ivs[(item, h)]
        err = (iv["actual"] - iv["pred"]).abs().dropna()
        calib = err.iloc[-ic["window"] :].to_numpy()
        a_t = float(iv["aci_alpha"].dropna().iloc[-1])
        hw = _conformal_quantile(calib, min(max(1 - a_t, 0.0), 1.0))
        hw_s = _conformal_quantile(calib, 1 - ic["alpha"])
        live[item] = {
            "origin": str(df.index[-1]),
            "level now": float(df[f"{item}_lvl"].iloc[-1]),
            "point %": point,
            "split conformal 90%": f"[{point - hw_s:+.1f}, {point + hw_s:+.1f}]",
            "ACI 90%": f"[{point - hw:+.1f}, {point + hw:+.1f}]" if 1 - a_t < 1 else "unbounded",
            "ACI quantile level": 1 - a_t,
        }
    # SHAP for the gradient-boosting model, earnings h=1, full sample
    X = features(df, "earnings").assign(p_rec=p_full)
    tr = X.join(target(df, "earnings", 1)).dropna()
    cols = REGIME_FEATS + ["g4q", "ust10y"]
    yv = tr["y1"].to_numpy()
    lo, hi = np.quantile(yv, [fc["winsor"], 1 - fc["winsor"]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gbm = gbm_model(cfg["seed"]).fit(tr[cols], np.clip(yv, lo, hi))
        expl = shap.TreeExplainer(gbm)
        sv = expl.shap_values(tr[cols])
    add_err = float(np.max(np.abs(sv.sum(1) + expl.expected_value - gbm.predict(tr[cols]))))
    mean_abs = pd.Series(np.abs(sv).mean(0), index=cols).rename(
        index={
            "g0": "EPS growth, last q",
            "g1": "EPS growth, q-1",
            "g4q": "EPS growth, 4q",
            "gdp_g": "GDP growth",
            "infl": "inflation",
            "d_ust10y": "10Y change",
            "ust10y": "10Y level",
            "eq_ret": "equities, last q",
            "eq_ret_4q": "equities, 4q",
            "p_rec": "recession prob.",
        }
    )
    plotting.shap_bar(mean_abs, "Gradient boosting, next-quarter EPS growth: SHAP importance", figs / "shap.png")
    plotting.attribution(contribs, figs / "attribution.png")
    check(
        "Explainability",
        "SHAP additivity (sum of SHAP + base = prediction)",
        f"max error {add_err:.1e}",
        "< 1e-6",
        add_err < 1e-6,
    )
    metrics["live"] = live
    lt = pd.DataFrame(live).T
    md += [
        "## 4. Live forecast and explanations",
        "",
        "Production model re-estimated on the full sample; 4-quarter log growth (%) from the last observation, "
        "with 90% intervals from the errors realised to date. For earnings, after the large 2020-2022 misses, "
        "ACI has pushed its working quantile level above 1: no finite interval built from the last 40 "
        "quarters of errors would have delivered 90% coverage, so ACI reports the interval as unbounded and the "
        "drift check in section 6 flags the same model. The attribution "
        "figure decomposes each forecast exactly into driver contributions (linear model: coefficient x "
        "deviation from the sample mean).",
        "",
        md_table(lt, 2),
        "",
        "SHAP importance of the gradient-boosting model (TreeSHAP, exact):",
        "",
        md_table(mean_abs.sort_values(ascending=False).to_frame("mean |SHAP|").T, 2),
        "",
    ]

    # 5. Scenarios
    sc = cfg["scenarios"]
    fans, srows, coefs = {}, {}, {}
    cm = ConditionalModel.fit(df, "earnings")
    lint = LintnerModel.fit(df)
    common = lint.resid.index.intersection(cm.resid_index)  # joint resampling of both items' shocks
    rE, rD = cm.residuals_on(common), lint.resid.loc[common].to_numpy()
    eb_ = pd.concat([pd.Series({"const": cm.c, "g_l1": cm.phi[0], "g_l2": cm.phi[1]}), cm.beta])
    coefs = {"coef": eb_, "t": eb_ / cm.bse.reindex(eb_.index)}
    fans = {"earnings": {}, "dividends": {}}
    for k, (name, spec) in enumerate(sc["definitions"].items()):
        U = np.random.default_rng(cfg["seed"] + k).random((sc["paths"], sc["horizon"], 2))
        eps = cm.project(df, scenario_paths(spec, sc["horizon"]), sc["paths"], sc["mean_block"], 0, U=U, resid=rE)
        dps = lint.project(df, eps, bootstrap_indices(U, len(common), sc["mean_block"]), rD)
        for item, sims in (("earnings", eps), ("dividends", dps)):
            s_ = summarise(sims, float(df[f"{item}_lvl"].iloc[-1]))
            fans[item][name] = s_
            srows[(LABEL[item], name)] = {k2: v for k2, v in s_.items() if not isinstance(v, np.ndarray)}
    plotting.fan_charts(fans, {i: df[f"{i}_lvl"] for i in fc["items"]}, figs / "scenarios.png")
    st = pd.DataFrame(srows).T
    st = st.loc[[k for item in ("earnings", "dividends") for k in st.index if k[0] == LABEL[item]]]
    metrics["scenarios"] = st.reset_index().to_dict(orient="records")
    # Engine parity
    rng = np.random.default_rng(0)
    U = rng.random((500, 9, 2))
    args = (
        0.1,
        np.array([0.6, -0.1]),
        np.array([0.3, 0.0, 1.4, 0.07]),
        rng.standard_normal((9, 4)),
        np.array([1.0, 2.0]),
        rng.standard_normal(200),
        4.0,
        U,
        500,
        9,
    )
    if HAVE_RUST:
        same = bool(np.array_equal(simulate_arx(*args, engine="rust"), simulate_arx(*args, engine="numpy")))
        check("Engine", "Rust kernel bit-identical to NumPy reference", str(same), "True", same)
    else:
        check("Engine", "Rust kernel available", "not built (NumPy fallback used)", "built", False, warn_only=True)
    md += [
        "## 5. Scenario projections",
        "",
        f"Earnings: conditional ARX model on the macro scenario (Huber M-estimation); {sc['paths']:,} Monte Carlo "
        f"paths with stationary block-bootstrap residuals (mean block {sc['mean_block']} quarters), "
        f"{sc['horizon']} quarters from "
        f"{df.index[-1]}. Simulation engine: {'Rust' if HAVE_RUST else 'NumPy'}.",
        "",
        "Earnings model (Huber M-estimation):",
        "",
        md_table(pd.DataFrame(coefs).T, 3),
        "",
        "Dividends are **linked** to the simulated earnings (Lintner partial adjustment to a target payout "
        "ratio), not to the macro drivers directly:",
        "",
        md_table(pd.DataFrame({"coef": lint.params, "t": lint.tvalues}).T, 3),
        "",
        md_table(st, 1),
        "",
    ]
    gdp_b = coefs["coef"]["gdp_g"]
    check("Scenario model", "Earnings respond positively to GDP growth", f"beta = {gdp_b:.3f}", "> 0", gdp_b > 0)
    ok_l = bool(lint.params["payout_l1"] < 0 and lint.params[[f"gE_l{j}" for j in (1, 2)]].min() > 0)
    check(
        "Scenario model",
        "Dividends: payout mean reversion (k < 0) and positive earnings pass-through",
        f"k = {lint.params['payout_l1']:.2f}",
        "k < 0, b1, b2 > 0",
        ok_l,
    )
    m_ = len(common)
    n_, h_ = U.shape[0], U.shape[1]
    idx_kernel = simulate_arx(
        0.0,
        np.zeros(0),
        np.zeros(0),
        np.zeros((h_, 0)),
        np.zeros(0),
        np.arange(m_, dtype=float),
        sc["mean_block"],
        U,
        n_,
        h_,
    )
    ok_idx = bool(np.array_equal(idx_kernel.astype(np.int64), bootstrap_indices(U, m_, sc["mean_block"])))
    check(
        "Engine",
        "Joint resampling: dividend shocks use the earnings shocks' historical dates",
        str(ok_idx),
        "True",
        ok_idx,
    )

    # 6. Monitoring
    for item in fc["items"]:
        g_ = df[item]
        Xc = np.column_stack(
            [
                np.ones(len(g_) - 2),
                g_.shift(1).iloc[2:],
                g_.shift(2).iloc[2:],
                df[["gdp_g", "infl", "d_ust10y", "eq_ret"]].iloc[2:].to_numpy(),
            ]
        )
        _, _, crossed = cusum(g_.iloc[2:].to_numpy(), Xc)
        check(
            "Stability",
            f"CUSUM test, {item} conditional model",
            "boundary crossed" if crossed else "within bounds",
            "within 5% bounds",
            not crossed,
            warn_only=True,
        )
    for (item, h), ev in evs.items():
        e = ev.errors()[pm].dropna()
        z = (e / e.rolling(20, min_periods=8).std().shift(h)).dropna()
        recent = float(z.iloc[-8:].abs().max())
        check(
            "Drift",
            f"{item} h={h}: max |standardised error|, last 8 forecasts",
            f"{recent:.2f}",
            "< 3",
            recent < 3,
            warn_only=True,
        )
    for v in ("gdp_g", "infl", "d_ust10y", "eq_ret"):
        ref, new = df[v].iloc[:-20].to_numpy(), df[v].iloc[-20:].to_numpy()
        edges = np.quantile(ref, np.linspace(0, 1, 6))
        edges[0], edges[-1] = -np.inf, np.inf
        r = np.clip(np.histogram(ref, edges)[0] / len(ref), 1e-4, None)
        c_ = np.clip(np.histogram(new, edges)[0] / len(new), 1e-4, None)
        psi = float(np.sum((c_ - r) * np.log(c_ / r)))
        check(
            "Input drift",
            f"PSI of {v}, last 20 quarters vs history",
            f"{psi:.3f}",
            "< 0.25",
            psi < 0.25,
            warn_only=True,
        )
    ch = pd.DataFrame(checks)
    counts = ch["status"].value_counts().to_dict()
    mon = (
        f"**{counts.get('PASS', 0)} PASS, {counts.get('WARN', 0)} WARN, {counts.get('FAIL', 0)} FAIL**\n\n"
        + md_table(ch, index=False)
    )
    (out / "monitoring.md").write_text(
        "# Model monitoring report\n\nGenerated by `python -m plancast run`.\n\n" + mon + "\n"
    )
    metrics["checks"] = checks
    md += ["## 6. Model monitoring", "", mon, ""]
    head = [
        "# PlanCast results",
        "",
        "Generated by `python -m plancast run` from the data snapshot in `data/raw`.",
        "",
    ]
    (out / "results.md").write_text("\n".join(head + md) + "\n")
    (out / "metrics.json").write_text(json.dumps(_clean(metrics), indent=2))
    return metrics
