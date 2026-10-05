# wayyLab

A Bloomberg-style research terminal for equity time series, built in Streamlit.

Nine panels covering charting, return distributions, stationarity, volatility
modelling, fractal regime detection, cointegration, forecast evaluation, and a
written analysis of three names.

The organising idea is that **an indicator reading is not a signal until it has
been tested against a baseline**. Several panels exist specifically to report
negative results: ARIMA fails to beat a random walk, the classical Hurst
estimator is biased, and most "cointegrated" pairs do not survive a
multiple-testing correction. Those are shown rather than hidden.

---

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens on <http://localhost:8501>. No API keys required — price data comes from
`yfinance`.

---

## Panels

| Key | Page | Contents |
|---|---|---|
| F1 | Overview | Quotes, candles + volume, risk readout, VaR backtest (Kupiec), drawdown, correlation |
| F2 | Chart | SMA/EMA/Bollinger/VWAP overlays, RSI, MACD, stochastic, ATR, crossover log |
| F3 | Returns | Cumulative return, moments, histogram vs normal, Q-Q, tail counts, ACF of returns vs squared returns |
| F4 | Structure | ADF + KPSS, ACF/PACF, differencing ladder, STL decomposition |
| F5 | Volatility | Realised / EWMA / Parkinson / GARCH(1,1), conditional forecast, clustering evidence |
| F6 | Regime | Hurst exponent, DFA vs R/S, block-bootstrap CI, validation against fBm with known H |
| F7 | Correlation | Correlation matrix, PCA, pairs scanner, Engle–Granger cointegration, Granger causality |
| F8 | Forecast | ARIMA, order grid, walk-forward vs random walk, Theil's U, Diebold–Mariano |
| F9 | Analysis | Per-name signal track records and a written interpretation |

---

## Three results worth looking at

**The Hurst estimator you use changes the answer.** On the same AAPL series,
classical R/S returns 0.563 ("trending") while DFA returns 0.456 ("random
walk"). Validated against fractional Brownian motion with known H, mean
absolute error is **0.059 for R/S against 0.009 for DFA** — and the bias is
largest for mean-reverting series, so R/S systematically over-reports trends.
F6 shows the validation curve alongside the estimate.

**ARIMA does not beat a random walk.** Walk-forward evaluation on daily closes
gives Theil's U ≈ 1.01 with a Diebold–Mariano p-value around 0.34 — the
difference is not distinguishable from sampling noise. F8 reports this plainly
instead of showing an in-sample fan chart.

**Most cointegrated pairs are multiple-testing artefacts.** Scanning an
8-asset universe yields 28 pairs, ~9 significant at p < 0.05, and typically
**none** surviving a Bonferroni threshold. F7 shows both counts.

---

## Layout

```
app.py              boot, chrome, page dispatch
views/              one module per page, each exposing render(cfg)
lib/
  theme.py          design tokens + Plotly layout
  styles.css        the terminal skin
  chrome.py         title bar, F-key nav, panels, stat tiles, status bar
  data.py           OHLCV access                         PURE
  stats.py          risk statistics, VaR backtest        PURE
  indicators.py     SMA/EMA/RSI/MACD/Bollinger/ATR       PURE
  tsa.py            ADF/KPSS/ACF/PACF/Ljung-Box/STL      PURE
  vol.py            realised/EWMA/Parkinson/GARCH        PURE
  regime.py         Hurst + fBm validation               PURE
  pairs.py          correlation/PCA/cointegration/Granger PURE
  forecast.py       ARIMA, walk-forward, DM test         PURE
  signals.py        conditional signal track records     PURE
  charts.py         Plotly builders                      PURE
  cached.py         the only streamlit-aware module in lib/
analysis/           written interpretation, editable in-app
```

Everything marked PURE avoids importing `streamlit`, so the analytics can be
lifted into a plain API without modification. Caching lives solely in
`lib/cached.py`.

---

## Notes

**Hurst backend.** `lib/regime.py` prefers [fracTime](https://github.com/Wayy-Research/fracTime)
when installed and otherwise uses pure-numpy R/S and DFA implementations that
reproduce it to ~1e-14. fracTime is not in `requirements.txt` because its
dependency tree (pymc, prophet, xgboost) exceeds typical free hosting limits.
The F6 panel reports which backend is active. To use fracTime locally:

```bash
pip install --no-deps -e /path/to/fracTime
```

fracTime's `Analyzer` defaults to `method='rs'`, the biased estimator; this
project passes `method='dfa'` explicitly.

**Offline mode.** "Freeze snapshot" writes parquet to `data/snapshot/`;
switching Source to `snapshot` runs the whole dashboard with no network, which
is useful when presenting.

**Colour.** Amber `#FFB300` = up and red `#E53935` = down are reserved status
colours. Multi-series charts use a separate 6-slot categorical palette in
`theme.SERIES`, validated for contrast and colour-vision deficiency against the
`#111111` surface, and assigned in fixed order.

---

## Caveats

The conditional edges in F9 use overlapping forward windows, so effective
sample sizes are far smaller than the observation counts suggest, and none
carry a significance test. The evidence window covers a single broad market
regime. Nothing here is investment advice.
