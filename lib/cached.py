"""
Streamlit caching layer over lib/data.py.

This is the ONLY module in lib/ that imports streamlit. Keeping the cache
wrappers here means lib/data.py stays portable into the FastAPI backend.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from . import data

TTL = 300  # 5 min - long enough to survive a demo, short enough to feel live


def _trim(df: pd.DataFrame, years: float) -> pd.DataFrame:
    """
    Snapshots are frozen at maximum depth so every lookback is available from
    one file. Without this the Lookback control does nothing in snapshot mode
    and a 1Y selection silently shows 10Y.
    """
    if df.empty or years <= 0:
        return df
    cutoff = df.index.max() - pd.Timedelta(days=int(365.25 * years))
    out = df[df.index >= cutoff]
    return out if not out.empty else df


@st.cache_data(ttl=TTL, show_spinner=False)
def ohlcv(symbol: str, years: float = 5.0, interval: str = "1d",
          backend: str = "yfinance", source: str = "live") -> pd.DataFrame:
    if source == "snapshot":
        df = data.load_snapshot(symbol, interval)
        if not df.empty:
            return _trim(df, years)
    return data.fetch_ohlcv(symbol, years=years, interval=interval, backend=backend)


@st.cache_data(ttl=TTL, show_spinner=False)
def panel(symbols: tuple[str, ...], years: float = 5.0,
          backend: str = "yfinance", source: str = "live") -> pd.DataFrame:
    if source == "snapshot":
        series = {}
        for s in symbols:
            df = data.load_snapshot(s)
            if not df.empty:
                series[s.upper()] = _trim(df, years)["close"]
        if series:
            return pd.DataFrame(series).sort_index().dropna(how="all")
    return data.fetch_panel(list(symbols), years=years, backend=backend)


# ---------------------------------------------------------------------------
# expensive analytics
#
# These dominate page latency: the Hurst bootstrap refits the estimator a few
# hundred times, the pairs scanner runs k(k-1) regressions, and walk-forward
# refits ARIMA at every origin. Caching keeps tab-switching instant during a
# live demo. Series/DataFrame args are hashed by content by Streamlit.
# ---------------------------------------------------------------------------
@st.cache_data(ttl=TTL, show_spinner=False)
def hurst(prices: pd.Series, method: str = "dfa", n_boot: int = 250) -> dict:
    from . import regime
    return regime.hurst(prices, method=method, n_boot=n_boot)


@st.cache_data(ttl=TTL, show_spinner=False)
def hurst_compare(prices: pd.Series) -> pd.DataFrame:
    from . import regime
    return regime.compare_methods(prices)


@st.cache_data(ttl=TTL, show_spinner=False)
def rolling_hurst(prices: pd.Series, window: int, step: int, method: str) -> pd.Series:
    from . import regime
    return regime.rolling_hurst(prices, window=window, step=step, method=method)


@st.cache_data(ttl=3600, show_spinner=False)
def hurst_validation(reps: int, n: int) -> pd.DataFrame:
    """Synthetic-data validation - depends only on (reps, n, seed), so cache hard."""
    from . import regime
    return regime.validate(reps=reps, n=n)


@st.cache_data(ttl=TTL, show_spinner=False)
def garch(returns: pd.Series, dist: str = "t") -> dict:
    from . import vol
    return vol.fit_garch(returns, dist=dist)


@st.cache_data(ttl=TTL, show_spinner=False)
def scan_pairs(closes: pd.DataFrame) -> pd.DataFrame:
    from . import pairs
    return pairs.scan_pairs(closes)


@st.cache_data(ttl=TTL, show_spinner=False)
def walk_forward(prices: pd.Series, order: tuple[int, int, int],
                 train: int, step: int, max_points: int) -> dict:
    from . import forecast
    return forecast.walk_forward(prices, order=order, train=train,
                                 step=step, max_points=max_points)


@st.cache_data(ttl=TTL, show_spinner=False)
def auto_order(prices: pd.Series, max_p: int, max_q: int, d: int) -> dict:
    from . import forecast
    return forecast.auto_order(prices, max_p=max_p, max_q=max_q, d=d)


def clear() -> None:
    for fn in (ohlcv, panel, hurst, hurst_compare, rolling_hurst,
               hurst_validation, garch, scan_pairs, walk_forward, auto_order):
        fn.clear()
