"""
Equity time-series access. PURE - no streamlit import, so this module can be
lifted straight into backend/app_v2/api/quant/ later. Caching lives in
lib/cached.py.

Two backends:
  * "yfinance"  - direct, deep history, keyless. Default.
  * "datastream"- wayyFin's own 15-provider DataStream, proving the shared
                  data layer works. Provider is PINNED to yfinance because
                  the default equity chain starts at Alpha Vantage, whose
                  free tier silently truncates to the 100 most recent bars
                  (outputsize=full is premium-only).
"""
from __future__ import annotations

import os
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

OHLCV_COLS = ["open", "high", "low", "close", "volume"]
SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data" / "snapshot"
_BACKEND_DIR = Path(__file__).resolve().parents[2] / "backend"


# ---------------------------------------------------------------------------
# yfinance backend
# ---------------------------------------------------------------------------
def _fetch_yfinance(symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.Ticker(symbol).history(start=start, end=end, interval=interval, auto_adjust=False)
    if df.empty:
        return pd.DataFrame(columns=OHLCV_COLS)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "timestamp"
    return df.dropna(how="all")


# ---------------------------------------------------------------------------
# DataStream backend (wayyFin's shared data layer)
# ---------------------------------------------------------------------------
def _fetch_datastream(symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
    if str(_BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(_BACKEND_DIR))
    cwd = os.getcwd()
    try:
        os.chdir(_BACKEND_DIR)  # settings resolves .env relative to CWD
        from app_v2.services.market_data.stream import DataStream

        ds = DataStream()
        pl_df = ds.get(symbol, start=start, end=end, interval=interval, provider="yfinance")
    finally:
        os.chdir(cwd)

    if pl_df is None or pl_df.height == 0:
        return pd.DataFrame(columns=OHLCV_COLS)
    df = pl_df.to_pandas().set_index("timestamp").sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[OHLCV_COLS].dropna(how="all")


_BACKENDS = {"yfinance": _fetch_yfinance, "datastream": _fetch_datastream}


# ---------------------------------------------------------------------------
# public
# ---------------------------------------------------------------------------
def fetch_ohlcv(
    symbol: str,
    years: float = 5.0,
    interval: str = "1d",
    backend: str = "yfinance",
    end: str | None = None,
) -> pd.DataFrame:
    """Daily OHLCV indexed by naive timestamp. Empty frame on failure."""
    end_d = date.fromisoformat(end) if end else date.today() + timedelta(days=1)
    start_d = end_d - timedelta(days=int(365.25 * years) + 2)
    fn = _BACKENDS.get(backend, _fetch_yfinance)
    try:
        return fn(symbol.upper().strip(), start_d.isoformat(), end_d.isoformat(), interval)
    except Exception:
        return pd.DataFrame(columns=OHLCV_COLS)


def fetch_panel(
    symbols: list[str], years: float = 5.0, backend: str = "yfinance"
) -> pd.DataFrame:
    """Wide frame of aligned closes, one column per symbol. Drops empties."""
    series = {}
    for s in symbols:
        df = fetch_ohlcv(s, years=years, backend=backend)
        if not df.empty:
            series[s.upper()] = df["close"]
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).sort_index().dropna(how="all")


def quote_from(df: pd.DataFrame) -> dict:
    """Last close + day-over-day change from an OHLCV frame."""
    if df.empty or len(df) < 1:
        return {"price": None, "change_pct": None, "asof": None}
    last = float(df["close"].iloc[-1])
    prev = float(df["close"].iloc[-2]) if len(df) > 1 else last
    chg = (last / prev - 1.0) * 100.0 if prev else 0.0
    return {"price": last, "change_pct": chg, "asof": df.index[-1]}


def log_returns(close: pd.Series) -> pd.Series:
    """Log returns, NaN/inf-safe."""
    r = np.log(close.astype(float)).diff()
    return r.replace([np.inf, -np.inf], np.nan).dropna()


def is_market_open(now: datetime | None = None) -> bool:
    """Rough US equity regular-session check (ignores holidays)."""
    et = ZoneInfo("America/New_York")
    now = (now or datetime.now(et)).astimezone(et)
    if now.weekday() >= 5:
        return False
    return time(9, 30) <= now.time() <= time(16, 0)


# ---------------------------------------------------------------------------
# snapshot: freeze data so a live demo cannot fail on wifi or a rate limit
# ---------------------------------------------------------------------------
def snapshot_path(symbol: str, interval: str = "1d") -> Path:
    return SNAPSHOT_DIR / f"{symbol.upper()}_{interval}.parquet"


def save_snapshot(symbol: str, df: pd.DataFrame, interval: str = "1d") -> Path:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    p = snapshot_path(symbol, interval)
    df.to_parquet(p)
    return p


def load_snapshot(symbol: str, interval: str = "1d") -> pd.DataFrame:
    p = snapshot_path(symbol, interval)
    if not p.exists():
        return pd.DataFrame(columns=OHLCV_COLS)
    return pd.read_parquet(p)


def available_snapshots(interval: str = "1d") -> list[str]:
    if not SNAPSHOT_DIR.exists():
        return []
    suffix = f"_{interval}.parquet"
    return sorted(p.name[: -len(suffix)] for p in SNAPSHOT_DIR.glob(f"*{suffix}"))
