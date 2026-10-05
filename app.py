"""
wayyLab - Streamlit research dashboard for equity time series.

Run:  streamlit run app.py     (from the dashboard/ directory)
"""
from __future__ import annotations

import streamlit as st

from lib import cached, chrome, data
from views import RENDERERS

DEFAULT_SYMBOL = "AAPL"
FREEZE_SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TLT", "GLD", "XLE"]

chrome.boot()


def controls() -> dict:
    c1, c2, c3, c4, c5 = st.columns([1.1, 1, 1, 1, 2.4])
    with c1:
        symbol = st.text_input("Symbol", DEFAULT_SYMBOL, key="sym").upper().strip()
    with c2:
        years = st.selectbox("Lookback", [1, 2, 3, 5, 10], index=3, key="yrs")
    with c3:
        interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0, key="iv")
    with c4:
        source = st.selectbox("Source", ["live", "snapshot"], index=0, key="src")
    with c5:
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Refresh", width="stretch"):
                cached.clear()
                st.rerun()
        with b2:
            if st.button("Freeze snapshot", width="stretch",
                         help="Save current symbols to parquet for offline demo"):
                _freeze(years, interval)
    return {"symbol": symbol or DEFAULT_SYMBOL, "years": float(years),
            "interval": interval, "source": source}


def _freeze(years: int, interval: str) -> None:
    saved = []
    wanted = sorted(set(FREEZE_SYMBOLS + [st.session_state.get("sym", "")]))
    for s in wanted:
        if not s:
            continue
        df = cached.ohlcv(s, years=float(years), interval=interval, source="live")
        if not df.empty:
            data.save_snapshot(s, df, interval)
            saved.append(s)
    st.toast(f"Snapshot saved: {', '.join(saved)}" if saved else "Nothing to save")


source = st.session_state.get("src", "live")
chrome.titlebar(data.is_market_open(), source_label=source.upper(), source_kind=source)
page = chrome.function_bar("overview")
cfg = controls()

RENDERERS[page](cfg)

chrome.status_bar(
    left=[
        ("SYMBOL", cfg["symbol"]),
        ("LOOKBACK", f"{int(cfg['years'])}Y"),
        ("SOURCE", cfg["source"]),
        ("SNAPSHOTS", f"{len(data.available_snapshots())}"),
        ("BACKEND", "yfinance"),
    ],
    right="WAYY RESEARCH · wayyLab 0.1",
)
