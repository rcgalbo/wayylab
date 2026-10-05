"""F2 - Chart & Technicals."""
from __future__ import annotations

import streamlit as st

from lib import cached, charts, chrome, indicators as ind
from lib import theme as T

CFG = T.PLOTLY_CONFIG


def render(cfg: dict) -> None:
    px_df = cached.ohlcv(cfg["symbol"], cfg["years"], cfg["interval"], source=cfg["source"])
    if px_df.empty:
        with chrome.panel("Chart", key="chart_err", meta=cfg["symbol"]):
            chrome.note(f"No data for {cfg['symbol']}.")
        return
    close = px_df["close"]

    # --- overlay controls -------------------------------------------------
    c1, c2, c3, c4 = st.columns([2.2, 1, 1, 1])
    with c1:
        overlays = st.multiselect(
            "Overlays", ["SMA 20", "SMA 50", "SMA 200", "EMA 12", "EMA 26",
                         "Bollinger 20", "VWAP"],
            default=["SMA 20", "SMA 50"], key="ov")
    with c2:
        rsi_len = st.number_input("RSI length", 2, 50, 14, key="rsil")
    with c3:
        show = st.selectbox("Lower panel", ["RSI", "MACD", "Stochastic", "ATR"], key="lower")
    with c4:
        bars = st.number_input("Bars shown", 60, 2000, min(400, len(px_df)), step=20, key="nbars")

    view = px_df.tail(int(bars))
    ov: dict = {}
    if "SMA 20" in overlays:
        ov["SMA20"] = ind.sma(close, 20).tail(int(bars))
    if "SMA 50" in overlays:
        ov["SMA50"] = ind.sma(close, 50).tail(int(bars))
    if "SMA 200" in overlays:
        ov["SMA200"] = ind.sma(close, 200).tail(int(bars))
    if "EMA 12" in overlays:
        ov["EMA12"] = ind.ema(close, 12).tail(int(bars))
    if "EMA 26" in overlays:
        ov["EMA26"] = ind.ema(close, 26).tail(int(bars))
    if "VWAP" in overlays:
        ov["VWAP"] = ind.vwap(px_df).tail(int(bars))
    if "Bollinger 20" in overlays:
        bb = ind.bollinger(close, 20)
        ov["BB upper"] = bb["upper"].tail(int(bars))
        ov["BB lower"] = bb["lower"].tail(int(bars))

    left, right = st.columns([2.8, 1])

    with left:
        rev = f"{cfg['symbol']}|{cfg['interval']}|{int(bars)}"
        with chrome.panel("Price", key="c_price",
                          meta=f"{cfg['symbol']} · {cfg['interval']} · {len(view)} bars",
                          footer="range buttons or drag a box to zoom · double-click to reset"):
            st.plotly_chart(
                charts.price_volume(view, ov, height=410, uirevision=rev),
                config=T.PLOTLY_CONFIG_PINNED, width="stretch", key="c_pricevol")

        if show == "RSI":
            r = ind.rsi(close, int(rsi_len)).tail(int(bars))
            with chrome.panel(f"RSI({int(rsi_len)})", key="c_rsi",
                              meta=f"last {r.dropna().iloc[-1]:.1f}" if r.notna().any() else "",
                              footer="amber band = oversold · red band = overbought"):
                st.plotly_chart(charts.osc(r, (30, 70), height=130), config=CFG, width="stretch")
        elif show == "MACD":
            m = ind.macd(close).tail(int(bars))
            with chrome.panel("MACD(12,26,9)", key="c_macd",
                              meta=f"hist {m['hist'].iloc[-1]:+.3f}",
                              footer="histogram = MACD − signal"):
                st.plotly_chart(charts.macd_panel(m, height=140), config=CFG, width="stretch")
        elif show == "Stochastic":
            s = ind.stochastic(px_df).tail(int(bars))
            with chrome.panel("Stochastic(14,3)", key="c_stoch",
                              meta=f"%K {s['k'].dropna().iloc[-1]:.1f}" if s["k"].notna().any() else "",
                              footer="%K solid · %D smoothed"):
                st.plotly_chart(charts.osc(s["k"].rename("%K"), (20, 80), height=130,
                                           extra=s["d"].rename("%D")),
                                config=CFG, width="stretch")
        else:
            a = ind.atr(px_df, 14).tail(int(bars))
            with chrome.panel("ATR(14)", key="c_atr",
                              meta=f"last {a.iloc[-1]:.2f}",
                              footer="average true range · absolute price units"):
                st.plotly_chart(charts.area_single(a, color=T.SERIES[3], height=130),
                                config=CFG, width="stretch")

    with right:
        last = float(close.iloc[-1])
        r_now = ind.rsi(close, int(rsi_len))
        m_now = ind.macd(close)
        bb_now = ind.bollinger(close, 20)
        s_now = ind.stochastic(px_df)
        bb_pos = ((last - bb_now["lower"].iloc[-1])
                  / max(1e-9, bb_now["upper"].iloc[-1] - bb_now["lower"].iloc[-1]))
        rv = float(r_now.dropna().iloc[-1]) if r_now.notna().any() else float("nan")

        with chrome.panel("Readout", key="c_read", meta=cfg["symbol"]):
            chrome.kv([
                ("Close", f"{last:,.2f}"),
                ("SMA 20", f"{ind.sma(close,20).iloc[-1]:,.2f}",
                 "up" if last > ind.sma(close, 20).iloc[-1] else "dn"),
                ("SMA 50", f"{ind.sma(close,50).iloc[-1]:,.2f}",
                 "up" if last > ind.sma(close, 50).iloc[-1] else "dn"),
                ("SMA 200", f"{ind.sma(close,200).iloc[-1]:,.2f}"
                 if close.size >= 200 else "n/a",
                 "up" if close.size >= 200 and last > ind.sma(close, 200).iloc[-1] else "dn"),
                (f"RSI {int(rsi_len)}", f"{rv:.1f}",
                 "dn" if rv > 70 else "up" if rv < 30 else ""),
                ("MACD hist", f"{m_now['hist'].iloc[-1]:+.3f}",
                 "up" if m_now["hist"].iloc[-1] >= 0 else "dn"),
                ("Stoch %K", f"{s_now['k'].dropna().iloc[-1]:.1f}"
                 if s_now["k"].notna().any() else "n/a"),
                ("BB position", f"{bb_pos:.0%}",
                 "dn" if bb_pos > 1 else "up" if bb_pos < 0 else ""),
                ("ATR 14", f"{ind.atr(px_df,14).iloc[-1]:,.2f}"),
            ])

        with chrome.panel("Signals", key="c_sig", meta="SMA 20/50 cross",
                          footer="crossovers are descriptive, not a strategy"):
            xo = ind.crossover_signals(ind.sma(close, 20), ind.sma(close, 50))
            if xo.empty:
                chrome.pending("no crossovers")
            else:
                recent = xo.tail(8).iloc[::-1]
                chrome.kv([
                    (f"{row.when:%Y-%m-%d}",
                     "golden" if row.side > 0 else "death",
                     "up" if row.side > 0 else "dn")
                    for row in recent.itertuples()
                ])

        with chrome.panel("Interpretation", key="c_interp"):
            notes = []
            if rv > 70:
                notes.append("RSI above 70 — overbought by the usual threshold.")
            elif rv < 30:
                notes.append("RSI below 30 — oversold by the usual threshold.")
            else:
                notes.append("RSI mid-range — no extreme reading.")
            if m_now["hist"].iloc[-1] >= 0:
                notes.append("MACD above signal — short-term momentum positive.")
            else:
                notes.append("MACD below signal — short-term momentum negative.")
            notes.append("These are threshold rules, not tested edges. "
                         "F8 shows what happens when a model is actually "
                         "evaluated out of sample.")
            chrome.note(" ".join(notes))
