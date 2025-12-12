"""Refresh the raw data snapshot in data/raw (public sources via the datahub.io core datasets on GitHub).

* sp500_monthly.csv: R. Shiller's S&P 500 price, dividends, earnings (datasets/s-and-p-500)
* gdp_quarterly.csv: BEA real GDP (datasets/gdp-us)
* cpi_monthly.csv: BLS CPI-U (datasets/cpi-us)
* ust10y_monthly.csv: 10Y Treasury yield (datasets/bond-yields-us-10y)
"""

from __future__ import annotations

import argparse
import io
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://raw.githubusercontent.com/datasets"
SOURCES = {
    "sp500_monthly.csv": f"{BASE}/s-and-p-500/main/data/data.csv",
    "gdp_quarterly.csv": f"{BASE}/gdp-us/main/data/quarter.csv",
    "cpi_monthly.csv": f"{BASE}/cpi-us/main/data/cpiai.csv",
    "ust10y_monthly.csv": f"{BASE}/bond-yields-us-10y/main/data/monthly.csv",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "raw"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 (fixed https URLs)
            df = pd.read_csv(io.BytesIO(r.read()))
        df.to_csv(out / name, index=False)
        print(f"{name:20s} {len(df):6d} rows, last {df.iloc[-1, 0]}")


if __name__ == "__main__":
    main()
