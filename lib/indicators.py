"""
Technical indicators. PURE - no streamlit, no I/O.

Mirrors the set implemented in backend/cpp/src/signal_evaluator.cpp so the
dashboard and the C++ engine agree on definitions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window).mean()


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def bollinger(s: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    mid = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0)
    return pd.DataFrame({"mid": mid, "upper": mid + n_std * sd, "lower": mid - n_std * sd})


def rsi(s: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI (exponential smoothing, alpha = 1/window)."""
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(100.0).where(avg_loss.notna())


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(s, fast) - ema(s, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP over the supplied window."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = (tp * df["volume"]).cumsum()
    vv = df["volume"].cumsum().replace(0, np.nan)
    return pv / vv


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3) -> pd.DataFrame:
    low_k = df["low"].rolling(k).min()
    high_k = df["high"].rolling(k).max()
    pct_k = 100 * (df["close"] - low_k) / (high_k - low_k).replace(0, np.nan)
    return pd.DataFrame({"k": pct_k, "d": pct_k.rolling(d).mean()})


def zscore(s: pd.Series, window: int = 20) -> pd.Series:
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0).replace(0, np.nan)
    return (s - mu) / sd


# ---------------------------------------------------------------------------
# signal extraction
# ---------------------------------------------------------------------------
def crossover_signals(fast: pd.Series, slow: pd.Series) -> pd.DataFrame:
    """
    +1 where fast crosses above slow, -1 where it crosses below.
    Returned as a frame of the crossing points only.
    """
    rel = (fast > slow).astype(int)
    cross = rel.diff()
    up = cross[cross == 1]
    dn = cross[cross == -1]
    return pd.DataFrame({
        "when": list(up.index) + list(dn.index),
        "side": [1] * len(up) + [-1] * len(dn),
    }).sort_values("when").reset_index(drop=True)


def rsi_signals(r: pd.Series, low: float = 30.0, high: float = 70.0) -> pd.DataFrame:
    """Entries when RSI crosses back out of oversold/overbought."""
    oversold = (r < low).astype(int).diff()
    overbought = (r > high).astype(int).diff()
    buys = oversold[oversold == -1].index      # leaving oversold
    sells = overbought[overbought == -1].index  # leaving overbought
    return pd.DataFrame({
        "when": list(buys) + list(sells),
        "side": [1] * len(buys) + [-1] * len(sells),
    }).sort_values("when").reset_index(drop=True)
