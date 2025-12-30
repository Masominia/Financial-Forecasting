# Methodology

## Data and targets

Quarterly, 1960Q1-2023Q2. Line items are R. Shiller's trailing-twelve-month S&P 500 earnings and dividends per
index share at quarter-end months; growth `g_t = 100 dln(level)`. Drivers: real GDP growth `400 dln`, CPI
inflation `400 dln` with quarter-of-year means removed (NSA index), 10Y yield level and change, S&P 500 price
return `100 dln`. Target at origin t and horizon h: `y_t(h) = g_{t+1} + ... + g_{t+h}`.

NBER recession quarters (peak excluded, trough included) are used only for evaluation.

## Regimes

Hamilton (1989) Markov-switching regression of GDP growth with two regimes, switching intercept, common
variance, and impulse dummies for 2020Q2 and 2020Q3 (statsmodels `MarkovRegression`, multiple random starts).
The filtered probability `P(s_t = recession | y_1..y_t)` is used as a real-time forecasting feature; between
re-estimations (every four quarters) the Hamilton filter is run with fixed parameters from the ergodic
distribution (our filter reproduces statsmodels' to 1e-15).

## Structural breaks

Bai and Perron (2003) global minimisation of the sum of squared residuals over all partitions with `m` breaks
and minimum segment length 16 quarters, by dynamic programming in `O(m n^2)`; `m` is chosen by
`BIC = n ln(SSR / n) + (2m + 1) ln n`. Applied to `|x - mean(x)|` the procedure dates variance breaks.
Parameter stability of the scenario models is checked with the Brown, Durbin and Evans (1975) recursive-residual
CUSUM and its 5% boundary `0.948 (sqrt(m) + 2r / sqrt(m))`.

## Forecasting models

All estimated at every origin on rows whose target is fully observed by the origin (`s + h <= t`):

| Model | Specification |
|---|---|
| naive | `y_t(h) = 0` (no change) |
| mean | historical mean of the target |
| ar | OLS on `g_t, g_{t-1}` |
| ardl | + GDP growth, inflation, 10Y change, equity return (1q and 4q) |
| regime_ardl | + real-time recession probability |
| gbm | histogram gradient boosting, depth 2, 150 trees, learning rate 0.05, min leaf 12, L2 1.0 |
| ensemble | equal-weight mean of ar, regime_ardl, gbm |

Outlier treatment: own-growth features and the target are winsorised at the 2.5% / 97.5% quantiles of each
training window (sensitivity: at 5% the same models come out best); evaluation uses raw outcomes.

Comparison: Diebold and Mariano (1995) test of equal squared-error loss with the Harvey, Leybourne and Newbold
(1997) small-sample correction and `h - 1` autocovariances.

## Intervals

At origin t, the calibration set is the absolute errors of forecasts made at origins `<= t - h` (realised by t),
last W = 40.

* Gaussian: `pred +/- 1.645 sd`, sd from in-sample residuals against raw targets.
* Split conformal: the `k`-th smallest calibration score with `k = ceil((n + 1)(1 - alpha))`.
* Adaptive conformal inference (Gibbs and Candes, 2021): `alpha_{j+1} = alpha_j + gamma (alpha - err_j)`,
  gamma = 0.05, the update applied when the outcome becomes known (delayed by h). If `k > n` the exact interval is
  unbounded; it is reported as such for the live forecast and counted as 1.5 x the largest score in average widths.

## Explanations

Linear models: `pred - (b0 + b' xbar) = sum_j b_j (x_j - xbar_j)`, an exact additive decomposition (the Shapley
values of a linear model with independent features). Gradient boosting: TreeSHAP (Lundberg et al., 2020), exact
for trees; additivity is checked at run time.

## Scenario models

Earnings: `g_t = c + phi_1 g_{t-1} + phi_2 g_{t-2} + beta' x_t + e_t`, Huber M-estimation (statsmodels `RLM`).

Dividends (Lintner, 1956): `g^D_t = c + a_1 g^D_{t-1} + a_2 g^D_{t-2} + sum_{j=1..4} b_j g^E_{t-j} + k ln(D/E)_{t-1} + e_t`,
Huber M-estimation, simulated on top of each simulated earnings path with the payout ratio updated path by path.

Shocks: raw residuals resampled with the stationary bootstrap (Politis and Romano, 1994), mean block 4: at each
step a new block starts with probability 1/4. Both items use the same bootstrap indices on their common sample,
so cross-item correlation is preserved. 20,000 paths, 9 quarters.

## Rust kernel

`rust/src/lib.rs` implements the ARX recursion with stationary-bootstrap shocks; uniforms are passed in from
Python so outputs are bit-identical to `simulate_arx_numpy` (same algorithm, same summation order). Built with
maturin; PyO3 abi3 wheel for Python 3.10+; rust-numpy for zero-copy inputs.

## Limitations

* Latest-vintage macro data: GDP is revised, so pseudo-real-time is not true real time (ALFRED vintages would fix it).
* About 250 quarterly observations: limited room for non-linear models, which is visible in the results.
* S&P 500 aggregates stand in for a firm's own line items; the framework is generic but not calibrated to a bank.
* Scenario paths include shock uncertainty but not parameter uncertainty.
* The Hamilton model is univariate; a multivariate or time-varying-transition model would use more information.

## References

* Bai, J. and Perron, P. (2003). Computation and analysis of multiple structural change models. Journal of Applied Econometrics.
* Brown, R., Durbin, J. and Evans, J. (1975). Techniques for testing the constancy of regression relationships over time. JRSS B.
* Diebold, F. and Mariano, R. (1995). Comparing predictive accuracy. Journal of Business and Economic Statistics.
* Gibbs, I. and Candes, E. (2021). Adaptive conformal inference under distribution shift. NeurIPS.
* Hamilton, J. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. Econometrica.
* Harvey, D., Leybourne, S. and Newbold, P. (1997). Testing the equality of prediction mean squared errors. International Journal of Forecasting.
* Lintner, J. (1956). Distribution of incomes of corporations among dividends, retained earnings, and taxes. American Economic Review.
* Lundberg, S. et al. (2020). From local explanations to global understanding with explainable AI for trees. Nature Machine Intelligence.
* McConnell, M. and Perez-Quiros, G. (2000). Output fluctuations in the United States: what has changed since the early 1980s? American Economic Review.
* Politis, D. and Romano, J. (1994). The stationary bootstrap. Journal of the American Statistical Association.
* Shiller, R. Online data, http://www.econ.yale.edu/~shiller/data.htm
