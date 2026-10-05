"""
F9 - Analysis.

Live evidence for each name plus a written interpretation. The prose lives in
analysis/<TICKER>.md and is editable from inside the app, so the narrative
stays the author's own while the numbers beside it are recomputed every run
and can never go stale.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from lib import cached, charts, chrome, data, signals
from lib import indicators as ind
from lib import theme as T

CFG = T.PLOTLY_CONFIG
NOTES_DIR = Path(__file__).resolve().parent.parent / "analysis"
DEFAULT_NAMES = ["JPM", "AMD", "TLT"]


def render(cfg: dict) -> None:
    c1, c2, c3 = st.columns([2.4, 1, 1])
    with c1:
        raw = st.text_input("Names under analysis", ",".join(DEFAULT_NAMES), key="an_n")
    with c2:
        horizon = st.number_input("Forward horizon (bars)", 5, 126, 21, step=1, key="an_h")
    with c3:
        years = st.selectbox("Evidence window", [3, 5, 10], index=1, key="an_y")

    names = [s.strip().upper() for s in raw.split(",") if s.strip()]
    if not names:
        chrome.note("Enter at least one ticker.")
        return

    tabs = st.tabs(names + ["Comparison"])

    snaps: dict[str, dict] = {}
    for tab, sym in zip(tabs, names):
        with tab:
            snaps[sym] = _one_name(sym, float(years), int(horizon))

    with tabs[-1]:
        _comparison(snaps, int(horizon))


# ---------------------------------------------------------------------------
def _one_name(sym: str, years: float, horizon: int) -> dict:
    df = cached.ohlcv(sym, years=years, source="live")
    if df.empty or len(df) < 260:
        with chrome.panel(sym, key=f"an_err_{sym}"):
            chrome.note(f"Not enough history for {sym}.")
        return {}

    snap = signals.snapshot(df)
    active = signals.active_conditions(df)
    edges = signals.edge_table(df, horizon)
    rb = signals.risk_budget(snap)
    close = df["close"]

    # --- state tiles -------------------------------------------------------
    chrome.stat_tiles([
        {"label": f"{sym} last", "value": f"{snap['last']:,.2f}",
         "change_pct": float(close.pct_change().iloc[-1] * 100)},
        {"label": "RSI 14", "value": f"{snap['rsi']:.1f}",
         "change_pct": None, "sub": _rsi_word(snap["rsi"])},
        {"label": "Z-score 60d", "value": f"{snap['z60']:+.2f}",
         "change_pct": None, "sub": "σ from 60d mean"},
        {"label": "vs 200d SMA", "value": f"{snap['vs_sma200']:+.1%}",
         "change_pct": None, "sub": "trend context"},
        {"label": "Ann. vol", "value": f"{snap['vol_full']:.0%}",
         "change_pct": None, "sub": f"21d {snap['vol21']:.0%}"},
    ])

    left, right = st.columns([2.3, 1])
    with left:
        ov = {"SMA50": ind.sma(close, 50), "SMA200": ind.sma(close, 200)}
        bb = ind.bollinger(close, 20)
        ov["BB upper"] = bb["upper"]
        ov["BB lower"] = bb["lower"]
        view = df.tail(320)
        ov = {k: v.tail(320) for k, v in ov.items()}
        with chrome.panel("Price in Context", key=f"an_px_{sym}",
                          meta=f"{sym} · last 320 bars",
                          footer="Bollinger 20 ± 2σ with SMA 50/200 · range buttons to zoom"):
            st.plotly_chart(
                charts.price_volume(view, ov, height=360,
                                    uirevision=f"{sym}|{years}"),
                config=T.PLOTLY_CONFIG_PINNED, width="stretch", key=f"an_pv_{sym}")
    with right:
        with chrome.panel("State", key=f"an_state_{sym}", meta="as of last bar",
                          footer="position vs its own recent distribution"):
            chrome.kv([
                ("RSI 14", f"{snap['rsi']:.1f}",
                 "up" if snap["rsi"] < 35 else "dn" if snap["rsi"] > 70 else ""),
                ("MACD hist", f"{snap['macd_hist']:+.3f}",
                 "up" if snap["macd_hist"] > 0 else "dn"),
                ("Bollinger pos", f"{snap['bb_pos']:.0%}"),
                ("Z-score 60d", f"{snap['z60']:+.2f}",
                 "up" if snap["z60"] < -1.5 else "dn" if snap["z60"] > 1.5 else ""),
                ("vs SMA 50", f"{snap['vs_sma50']:+.1%}",
                 "up" if snap["vs_sma50"] > 0 else "dn"),
                ("vs SMA 200", f"{snap['vs_sma200']:+.1%}",
                 "up" if snap["vs_sma200"] > 0 else "dn"),
                ("Off 52w high", f"{snap['from_high']:+.1%}"),
                ("MA stacked", "yes" if snap["stacked"] else "no",
                 "up" if snap["stacked"] else "dn"),
            ])
        with chrome.panel("Risk", key=f"an_risk_{sym}", meta="full window",
                          footer=f"2×ATR stop ⇒ max {rb['weight_pct']:.0f}% weight at 1% account risk"
                          if rb.get("ok") else ""):
            chrome.kv([
                ("Ann. vol", f"{snap['vol_full']:.1%}"),
                ("21d vol", f"{snap['vol21']:.1%}",
                 "dn" if snap["vol_ratio"] > 1.15 else "up"),
                ("ATR 14", f"{snap['atr_pct']:.2%}"),
                ("VaR 95%", f"{snap['var95']:.2%}", "dn"),
                ("CVaR 95%", f"{snap['cvar95']:.2%}", "dn"),
                ("Max drawdown", f"{snap['max_dd']:.1%}", "dn"),
                ("Sharpe", f"{snap['sharpe']:.2f}",
                 "up" if snap["sharpe"] > 0 else "dn"),
            ])

    # --- the evidence that matters ----------------------------------------
    with chrome.panel("Signal Track Record", key=f"an_edge_{sym}",
                      meta=f"{sym} · forward {horizon} bars · {int(years)}y window",
                      footer="edge = conditional mean minus the mean of all other days"):
        show = edges.copy()
        show["mean_fwd"] = show["mean_fwd"].map(lambda v: f"{v:+.2%}" if pd.notna(v) else "—")
        show["base_mean"] = show["base_mean"].map(lambda v: f"{v:+.2%}" if pd.notna(v) else "—")
        show["edge"] = show["edge"].map(lambda v: f"{v:+.2%}" if pd.notna(v) else "—")
        show["hit_rate"] = show["hit_rate"].map(lambda v: f"{v:.0%}" if pd.notna(v) else "—")
        show = show.rename(columns={"mean_fwd": "mean fwd", "base_mean": "all other days",
                                    "hit_rate": "hit rate", "active": "ACTIVE NOW"})
        st.dataframe(show, width="stretch", height=350, hide_index=True)
        if active:
            chrome.note("Active now: " + "; ".join(active))
        else:
            chrome.note("No standard condition is currently triggered.")

    # --- the writeup -------------------------------------------------------
    _notes_panel(sym)
    return snap


# ---------------------------------------------------------------------------
def _notes_panel(sym: str) -> None:
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    path = NOTES_DIR / f"{sym}.md"
    key_edit = f"an_edit_{sym}"
    editing = st.session_state.get(key_edit, False)

    header_meta = path.name if path.exists() else "no note yet"
    with chrome.panel("Written Analysis", key=f"an_note_{sym}", meta=header_meta,
                      footer="stored in analysis/ — edit here or in your editor"):
        if editing:
            text = st.text_area("note", value=_read(path), height=420,
                                key=f"an_ta_{sym}", label_visibility="collapsed")
            b1, b2 = st.columns([1, 6])
            with b1:
                if st.button("Save", key=f"an_save_{sym}", type="primary",
                             width="stretch"):
                    path.write_text(text, encoding="utf-8")
                    st.session_state[key_edit] = False
                    st.rerun()
            with b2:
                if st.button("Cancel", key=f"an_cancel_{sym}"):
                    st.session_state[key_edit] = False
                    st.rerun()
        else:
            body = _read(path)
            if body.strip():
                st.markdown(body)
            else:
                chrome.pending("no analysis written yet")
            if st.button("Edit", key=f"an_editbtn_{sym}"):
                st.session_state[key_edit] = True
                st.rerun()


def _comparison(snaps: dict[str, dict], horizon: int) -> None:
    rows = {k: v for k, v in snaps.items() if v}
    if len(rows) < 2:
        chrome.pending("need at least two names")
        return
    t = pd.DataFrame(rows).T
    with chrome.panel("Side by Side", key="an_cmp", meta="current state",
                      footer="the same indicator means different things on different names"):
        disp = pd.DataFrame({
            "RSI": t["rsi"].map(lambda v: f"{v:.1f}"),
            "Z 60d": t["z60"].map(lambda v: f"{v:+.2f}"),
            "vs 200d": t["vs_sma200"].map(lambda v: f"{v:+.1%}"),
            "off high": t["from_high"].map(lambda v: f"{v:+.1%}"),
            "ann vol": t["vol_full"].map(lambda v: f"{v:.0%}"),
            "VaR 95%": t["var95"].map(lambda v: f"{v:.2%}"),
            "max DD": t["max_dd"].map(lambda v: f"{v:.0%}"),
            "Sharpe": t["sharpe"].map(lambda v: f"{v:.2f}"),
        })
        st.dataframe(disp, width="stretch", height=160)

    with chrome.panel("Volatility Budget", key="an_budget",
                      meta="position weight at 1% account risk, 2×ATR stop",
                      footer="identical conviction, very different position sizes"):
        labels, weights = [], []
        for sym, s in rows.items():
            rb = signals.risk_budget(s)
            if rb.get("ok"):
                labels.append(sym)
                weights.append(rb["weight_pct"])
        if labels:
            st.plotly_chart(charts.bars(labels, weights, height=200, yfmt=".0f"),
                            config=CFG, width="stretch")
            chrome.note(
                "Position size here is set by volatility alone: " + ", ".join(
                    f"{l} {w:.0f}%" for l, w in zip(labels, weights)) +
                ". A high-volatility name earns a smaller weight for the same "
                "dollar risk, which is why 'sell' can mean 'hold less'. "
                "This is a ceiling on size, not a view on direction — a quiet "
                "name in a downtrend scores a large budget and still should "
                "not be owned."
            )

    _notes_panel("_SUMMARY")


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _rsi_word(v: float) -> str:
    return "oversold" if v < 35 else "overbought" if v > 70 else "neutral"
