# wayyLab — Financial Time Series Research Terminal

**Live app:** https://wayylab.streamlit.app/ · **Source:** https://github.com/rcgalbo/wayylab

---

## Overview

wayyLab is a nine-panel Streamlit dashboard for the display and modelling of
equity time series, styled after a Bloomberg terminal. It covers charting and
technical indicators, return distributions, stationarity testing, volatility
modelling, fractal regime detection, cross-sectional correlation and
cointegration, out-of-sample forecast evaluation, and a written analysis of
three specific names.

The design principle throughout is that **an indicator reading is not a signal
until it has been tested against a baseline**. Three panels exist specifically
to report negative results rather than hide them, and those are the findings I
consider most worth presenting.

## Data and methods

Daily OHLCV from `yfinance`; ten symbols frozen as 10-year parquet snapshots so
the app runs offline and the reported figures are reproducible. Statistics come
from `statsmodels`, `scipy` and `arch`. Methods implemented: SMA/EMA/Bollinger/
VWAP/RSI/MACD/stochastic/ATR; log-return moments with Jarque–Bera; ADF and KPSS
with ACF/PACF and STL decomposition; realised, EWMA (λ=0.94), Parkinson and
GARCH(1,1)-t volatility; Hurst exponent via rescaled range and detrended
fluctuation analysis with a moving-block bootstrap; PCA, Engle–Granger
cointegration with Ornstein–Uhlenbeck half-life, and Granger causality; ARIMA
with rolling-origin walk-forward evaluation, Theil's U and Diebold–Mariano.

## Three findings

**1. The Hurst estimator you choose determines the answer.** On the same AAPL
series, classical rescaled-range analysis returns H = 0.563 ("trending") while
detrended fluctuation analysis returns H = 0.456 ("mean-reverting") — opposite
conclusions across the 0.5 threshold. To decide which to trust I generated
fractional Brownian motion with known H by Davies–Harte circulant embedding and
measured both estimators against it: mean absolute error is **0.059 for R/S
against 0.009 for DFA**, and the R/S bias is largest at low H, meaning it
systematically under-detects mean reversion. The dashboard reports DFA with a
95% bootstrap interval and declines to name a regime when that interval
contains 0.5 — which it does for AAPL, at [0.383, 0.530].

**2. ARIMA does not beat a random walk.** Walk-forward evaluation refits the
model at each origin on a trailing window and forecasts one step ahead, so
nothing after the origin is ever visible to the fit. Against a naive
"tomorrow equals today" benchmark, ARIMA returns **Theil's U ≈ 1.008** — worse
than the benchmark — with a Diebold–Mariano p-value of **0.34**, meaning the
difference is not even distinguishable from sampling noise. In-sample AIC and a
confident-looking forecast fan chart say nothing about this; only the
out-of-sample comparison does.

**3. Most "cointegrated" pairs are multiple-testing artefacts.** Scanning an
eight-asset universe produces 28 pairs, of which 9 are significant at p < 0.05.
At that threshold roughly 1.4 false positives are expected by chance; applying
a Bonferroni correction (α = 0.0018) leaves **zero**. Relatedly, SPY/QQQ
regress with R² = 0.987 yet fail the cointegration test at p = 0.09 — high
explanatory power and a stationary spread are not the same property.

Supporting results: historical VaR is well calibrated (95% model, 5.48%
realised breach rate over 1,004 days, Kupiec LR = 0.47); daily returns show
excess kurtosis of +5.70 with two observations beyond 5σ where a normal
predicts zero; and returns are serially uncorrelated (Ljung–Box p = 0.61) while
squared returns are strongly so (p = 1.3e-30) — unforecastable direction,
forecastable magnitude, which is the justification for fitting GARCH at all
(persistence 0.979, shock half-life 32 days).

## Applied analysis

The Analysis panel evaluates three names by computing, for each signal, the
forward return distribution following every past occurrence *and the baseline
for all other days*, since the edge is the difference rather than the
conditional number alone.

**Buy — JPM** at RSI 34.5, z-score −2.07. Hurst below 0.5 licenses a
mean-reversion model; historically RSI < 35 on this name returned +4.90% over
21 days at a 72.7% hit rate against a +1.23% baseline; and GARCH conditional
volatility sits below its long-run level, indicating a drift lower on calm tape
rather than a repricing.

**Sell — AMD**, but not for the obvious reason. Every overbought box is ticked
(RSI 70.1, at its high, 69% above the 200-day), yet RSI > 70 on AMD has
historically been followed by **+11.67%** at a 63% hit rate — an +8.4 point
edge — consistent with its Hurst of 0.556. The sell case is therefore a risk
case, not a directional one: 56.9% annualised volatility, 5.6% daily VaR, a
−65% maximum drawdown and a 39-day volatility half-life mean the position
supports roughly a 13% weight against JPM's 26% on the same risk budget. The
recommendation is to trim to that budget while keeping the thesis.

**Control — TLT** is the most oversold name on the page (RSI 23.5) and the one
I would not buy: RSI < 35 there has returned −0.37% at a 46% hit rate, with a
Sharpe of −0.80 and no stable level to revert toward. Its inclusion tests
whether the JPM argument is reasoning or merely a restated RSI rule.

## Implementation

Analytics live in a `lib/` package that never imports Streamlit, so the same
functions can be served from an API unchanged; caching is isolated to a single
module. The Hurst implementation prefers an external library when available and
otherwise falls back to pure-NumPy estimators that reproduce it to ~1e-14,
verified by direct comparison. Dependency majors are pinned to the tested lines
so the deployment rebuilds identically.

## Limitations

The conditional edges use overlapping 21-day forward windows, so effective
sample sizes are far smaller than the observation counts imply and none carry a
significance test. The five-year evidence window spans a single broad market
regime, and the conditional statistics are in-sample on the same window used to
select the names. Nothing here is investment advice.
