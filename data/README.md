# Data

Snapshot taken with `python scripts/fetch_data.py`. The pipeline reads only this folder.

| File | Content | Coverage used | Source |
|---|---|---|---|
| `sp500_monthly.csv` | S&P 500 price, trailing-12-month dividends and reported earnings per index share | 1960-2023Q2 (earnings and dividends end in June 2023) | R. Shiller, via datahub.io (`datasets/s-and-p-500`) |
| `gdp_quarterly.csv` | Real GDP, chained dollars | 1960Q1-2023Q2 | BEA via datahub.io (`datasets/gdp-us`) |
| `cpi_monthly.csv` | CPI-U all items, not seasonally adjusted | 1960-2023 | BLS via datahub.io (`datasets/cpi-us`) |
| `ust10y_monthly.csv` | 10-year Treasury yield, monthly average | 1960-2023 | Federal Reserve via datahub.io |

Notes:

* Shiller's monthly earnings and dividends are interpolated from quarterly data; only quarter-end months are used.
* Quarterly CPI inflation is computed from the NSA index and its quarter-of-year means are removed.
* GDP growth is recomputed from the chained level (the snapshot's growth column is shifted by one quarter).
* All series are latest-vintage: GDP revisions mean the pseudo-real-time evaluation is not a true real-time one
  (that would require ALFRED vintages).
* NBER recession dates are hard-coded in `src/plancast/data.py` from the NBER business-cycle chronology and used
  only for evaluation.
