"""Static figures for the report (matplotlib, PNG for the README)."""

from __future__ import annotations

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
NEUTRAL, SHADE = "#8a8985", "#ecebe7"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "figure.dpi": 150,
        "axes.axisbelow": True,
    }
)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _ts(idx):
    return idx.to_timestamp() if isinstance(idx, pd.PeriodIndex) else idx


def _shade(ax, recession: pd.Series):
    r = recession.astype(bool)
    t = _ts(r.index)
    for i in np.where(r.to_numpy())[0]:
        ax.axvspan(t[i] - pd.Timedelta(days=46), t[i] + pd.Timedelta(days=46), color=SHADE, lw=0)


def regimes(df: pd.DataFrame, p_full: pd.Series, p_rt: pd.Series, breaks: list, path):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    _shade(a1, df["recession"])
    a1.plot(_ts(df.index), df["gdp_g"].clip(-12, 12), color=INK2, lw=1.0)
    for i, b in enumerate(breaks):
        a1.axvline(b.to_timestamp(), color=SERIES[1], ls="--", lw=1.2)
        a1.annotate(
            f"break {b}",
            (b.to_timestamp(), 11.0 - 2.2 * i),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=8,
            color=SERIES[1],
            backgroundcolor=SURFACE,
        )
    a1.set_ylabel("real GDP growth, % ann. (clipped at +/-12)")
    a1.set_title("GDP growth, NBER recessions (shaded) and estimated volatility breaks", loc="left")
    _shade(a2, df["recession"])
    a2.plot(_ts(p_full.index), p_full, color=SERIES[0], lw=1.2, label="filtered, full-sample parameters")
    a2.plot(_ts(p_rt.index), p_rt, color=SERIES[1], lw=1.6, label="real time (re-estimated each quarter)")
    a2.set_ylim(-0.02, 1.02)
    a2.set_ylabel("P(recession regime)")
    a2.set_title("Hamilton regime model: recession probability", loc="left")
    a2.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.12))
    _save(fig, path)


def accuracy(tables: dict[str, pd.DataFrame], path):
    fig, axes = plt.subplots(1, len(tables), figsize=(3.0 * len(tables), 3.8), sharey=True)
    for ax, (name, t) in zip(np.atleast_1d(axes), tables.items(), strict=True):
        r = t["RMSE vs naive"].drop("naive")
        colors = [SERIES[0] if v < 1 else NEUTRAL for v in r]
        ax.barh(r.index[::-1], r.to_numpy()[::-1], color=colors[::-1], height=0.6)
        ax.axvline(1.0, color=INK2, lw=0.8, ls=":")
        for y, (m, v) in enumerate(r[::-1].items()):
            star = "*" if t.loc[m, "DM p"] < 0.05 else ""
            ax.annotate(f"{v:.2f}{star}", (v, y), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8)
        ax.set_title(name, loc="left")
        ax.set_xlim(0, max(1.25, r.max() * 1.15))
        ax.set_xlabel("RMSE / naive")
    _save(fig, path)


def coverage(ivs: dict[str, pd.DataFrame], start, alpha: float, path):
    fig, axes = plt.subplots(len(ivs), 1, figsize=(10, 2.9 * len(ivs)), sharex=True)
    for ax, (name, iv) in zip(np.atleast_1d(axes), ivs.items(), strict=True):
        d = iv.loc[pd.Period(start, "Q") :].dropna(subset=["split_hw"])
        for i, (m, lab) in enumerate(
            (("gauss", "Gaussian"), ("split", "split conformal"), ("aci", "adaptive conformal"))
        ):
            ax.plot(
                _ts(d.index),
                d[f"{m}_cover"].astype(float).rolling(20, min_periods=12).mean(),
                color=SERIES[i],
                label=lab,
            )
        ax.axhline(1 - alpha, color=INK2, lw=0.8, ls=":")
        ax.set_ylim(0.2, 1.02)
        ax.set_ylabel("rolling 20q coverage")
        ax.set_title(f"{name}: 90% interval coverage", loc="left")
    np.atleast_1d(axes)[0].legend(loc="lower left", ncol=3)
    _save(fig, path)


def interval_band(iv: pd.DataFrame, recession: pd.Series, start, title: str, path):
    d = iv.loc[pd.Period(start, "Q") :].dropna(subset=["aci_hw"])
    t = _ts(d.index)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    _shade(ax, recession.reindex(d.index).fillna(0))
    ax.fill_between(
        t,
        d["pred"] - d["aci_hw"],
        d["pred"] + d["aci_hw"],
        color=SERIES[0],
        alpha=0.15,
        lw=0,
        label="90% adaptive conformal interval",
    )
    ax.plot(t, d["pred"], color=SERIES[0], lw=1.4, label="forecast")
    out = ~d["aci_cover"].astype(bool)
    ax.plot(t, d["actual"], "o", ms=3, color=NEUTRAL, label="actual")
    ax.plot(t[out.to_numpy()], d["actual"][out], "o", ms=5, color=SERIES[1], label="outside interval")
    lim = np.nanquantile(np.abs(d["actual"]), 0.97) * 1.5
    ax.set_ylim(-lim, lim)
    ax.set_ylabel("% growth")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.12))
    _save(fig, path)


def shap_bar(mean_abs: pd.Series, title: str, path):
    s = mean_abs.sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.barh(s.index, s.to_numpy(), color=SERIES[0], height=0.6)
    for y, v in enumerate(s.to_numpy()):
        ax.annotate(f"{v:.2f}", (v, y), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8)
    ax.set_xlabel("mean |SHAP value|, % growth")
    ax.set_title(title, loc="left")
    _save(fig, path)


def attribution(contribs: dict[str, pd.Series], path):
    fig, axes = plt.subplots(1, len(contribs), figsize=(5.4 * len(contribs), 3.8))
    for ax, (name, c) in zip(np.atleast_1d(axes), contribs.items(), strict=True):
        c = c[c.abs() > 1e-9]
        cum = 0.0
        levels = [0.0]
        for i, (k, v) in enumerate(c.items()):
            color = NEUTRAL if k in ("average growth", "forecast") else (SERIES[0] if v >= 0 else SERIES[1])
            if k == "forecast":
                ax.bar(i, v, color=INK2, width=0.6)
                ax.annotate(
                    f"{v:+.1f}",
                    (i, v),
                    xytext=(0, 3 if v >= 0 else -10),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
            else:
                ax.bar(i, v, bottom=cum, color=color, width=0.6)
                ax.annotate(
                    f"{v:+.1f}",
                    (i, cum + v),
                    xytext=(0, 3 if v >= 0 else -10),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
                cum += v
                levels.append(cum)
        ax.axhline(0, color=INK2, lw=0.6)
        lo_, hi_ = min(levels + [float(c.iloc[-1])]), max(levels + [float(c.iloc[-1])])
        pad = 0.18 * (hi_ - lo_ if hi_ > lo_ else 1.0)
        ax.set_ylim(min(lo_, 0) - pad, hi_ + pad)
        ax.set_xticks(range(len(c)), c.index, rotation=30, ha="right")
        ax.set_ylabel("% growth over 4 quarters")
        ax.set_title(name, loc="left")
    _save(fig, path)


def fan_charts(fans: dict[str, dict[str, dict]], history: dict[str, pd.Series], path):
    fig, axes = plt.subplots(1, len(fans), figsize=(5.6 * len(fans), 4.0))
    for ax, (item, scen) in zip(np.atleast_1d(axes), fans.items(), strict=True):
        hist = history[item].iloc[-12:]
        x_hist = np.arange(-len(hist) + 1, 1)
        ax.plot(x_hist, hist.to_numpy(), color=NEUTRAL, lw=1.4, label="history")
        h0 = float(hist.iloc[-1])
        for i, (name, s) in enumerate(scen.items()):
            x = np.arange(0, len(s["median"]) + 1)
            if name == "Severely adverse":
                ax.fill_between(
                    x,
                    np.r_[h0, s["p5"]],
                    np.r_[h0, s["p95"]],
                    color=SERIES[i],
                    alpha=0.15,
                    lw=0,
                    label=f"{name}, 5-95%",
                )
            ax.plot(x, np.r_[h0, s["median"]], color=SERIES[i], label=f"{name}, median")
        ax.axvline(0, color=INK2, lw=0.6, ls=":")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_xlabel("quarters from jump-off (2023Q2)")
        ax.set_ylabel("trailing-12-month, USD per index share")
        ax.set_title(f"S&P 500 {item}: scenario projections", loc="left")
    h, lab = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    _save(fig, path)
