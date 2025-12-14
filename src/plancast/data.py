"""Quarterly planning panel: financial line items and their macro drivers.

Line items (R. Shiller's S&P 500 series, trailing-twelve-month, per index share):

* ``earnings``: reported EPS, cyclical and regime-dependent (the "revenue-like" item);
* ``dividends``: DPS, sticky and slow-moving (the "expense-like" item).

Drivers: real GDP growth, CPI inflation, 10Y Treasury yield (level and change),
S&P 500 price return. Growth rates are 100 x dln per quarter for the line items
and 400 x dln (annualised) for GDP and CPI.

NBER recession quarters (peak excluded, trough included) are used only to
*evaluate* regime detection and forecasts by phase, never as a model input.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

# NBER business-cycle reference dates, quarterly (peak, trough).
NBER = [
    ("1960Q2", "1961Q1"),
    ("1969Q4", "1970Q4"),
    ("1973Q4", "1975Q1"),
    ("1980Q1", "1980Q3"),
    ("1981Q3", "1982Q4"),
    ("1990Q3", "1991Q1"),
    ("2001Q1", "2001Q4"),
    ("2007Q4", "2009Q2"),
    ("2019Q4", "2020Q2"),
]

DRIVERS = ["gdp_g", "infl", "ust10y", "d_ust10y", "eq_ret"]
ITEMS = ["earnings", "dividends"]


def _q(series: pd.Series, how: str = "last") -> pd.Series:
    s = series.resample("QS")
    out = s.last() if how == "last" else s.mean()
    out.index = out.index.to_period("Q")
    return out


def recession_flags(index: pd.PeriodIndex) -> pd.Series:
    flag = pd.Series(0, index=index, dtype=int)
    for peak, trough in NBER:
        p, t = pd.Period(peak, "Q"), pd.Period(trough, "Q")
        flag[(index > p) & (index <= t)] = 1
    return flag.rename("recession")


def load_panel(start: str = "1960Q1", end: str = "2023Q2", raw: Path = RAW) -> pd.DataFrame:
    sh = pd.read_csv(raw / "sp500_monthly.csv", parse_dates=["Date"]).set_index("Date")
    sh = sh[sh.index.month.isin([3, 6, 9, 12])]
    gdp = pd.read_csv(raw / "gdp_quarterly.csv", parse_dates=["date"]).set_index("date")["level-chained"]
    cpi = pd.read_csv(raw / "cpi_monthly.csv", parse_dates=["Date"]).set_index("Date")["Index"]
    y10 = pd.read_csv(raw / "ust10y_monthly.csv", parse_dates=["Date"]).set_index("Date")["Rate"]
    gdp.index = gdp.index.to_period("Q")
    eps, dps, px = (_q(sh[c].where(sh[c] > 0)) for c in ("Earnings", "Dividend", "SP500"))
    cpi_q = _q(cpi, "mean")
    infl = 400 * np.log(cpi_q).diff()
    infl = infl - infl.groupby(infl.index.quarter).transform("mean") + infl.mean()  # CPI snapshot is NSA
    y10q = _q(y10, "mean")
    df = pd.DataFrame(
        {
            "earnings_lvl": eps,
            "dividends_lvl": dps,
            "earnings": 100 * np.log(eps).diff(),
            "dividends": 100 * np.log(dps).diff(),
            "gdp_g": 400 * np.log(gdp).diff(),
            "infl": infl,
            "ust10y": y10q,
            "d_ust10y": y10q.diff(),
            "eq_ret": 100 * np.log(px).diff(),
        }
    )
    df = df.loc[start:end].dropna()
    df["recession"] = recession_flags(df.index)
    return df
