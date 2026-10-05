"""F4 - Stationarity & Structure."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from lib import cached, charts, chrome, data, tsa
from lib import theme as T

CFG = T.PLOTLY_CONFIG


def render(cfg: dict) -> None:
    px_df = cached.ohlcv(cfg["symbol"], cfg["years"], cfg["interval"], source=cfg["source"])
    if px_df.empty:
        with chrome.panel("Structure", key="s_err", meta=cfg["symbol"]):
            chrome.note(f"No data for {cfg['symbol']}.")
        return
    close = px_df["close"]

    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        target = st.selectbox("Series under test",
                              ["price", "log price", "log returns", "differenced price"],
                              index=2, key="s_t")
    with c2:
        nlags = st.number_input("ACF/PACF lags", 10, 80, 40, step=5, key="s_l")
    with c3:
        period = st.number_input("STL period", 5, 252, 21, step=1, key="s_p")

    import numpy as np
    series = {
        "price": close,
        "log price": np.log(close),
        "log returns": data.log_returns(close),
        "differenced price": tsa.difference(close, 1),
    }[target]

    joint = tsa.joint_stationarity(series)
    adf, kpss = joint["adf"], joint["kpss"]

    # --- row 1: the series + verdict --------------------------------------
    a, b = st.columns([2, 1])
    with a:
        with chrome.panel("Series", key="s_series",
                          meta=f"{cfg['symbol']} · {target} · n={len(series):,}",
                          footer="a stationary series has a stable mean and variance"):
            st.plotly_chart(charts.area_single(series, color=T.SERIES[1], height=235),
                            config=CFG, width="stretch")
    with b:
        verdict_cls = {"stationary": "up", "non-stationary": "dn",
                       "inconclusive": "info"}[joint["conclusion"]]
        with chrome.panel("Stationarity", key="s_tests", meta="ADF + KPSS",
                          footer="opposite nulls — agreement is the strong result"):
            chrome.kv([
                ("ADF statistic", f"{adf['stat']:.3f}"),
                ("ADF p-value", f"{adf['p']:.4f}",
                 "up" if adf["stationary"] else "dn"),
                ("ADF 5% crit", f"{adf['crit'].get('5%', float('nan')):.3f}", "dim"),
                ("KPSS statistic", f"{kpss['stat']:.3f}"),
                ("KPSS p-value", f"{kpss['p']:.4f}",
                 "up" if kpss["stationary"] else "dn"),
                ("KPSS 5% crit", f"{kpss['crit'].get('5%', float('nan')):.3f}", "dim"),
                ("Verdict", joint["conclusion"].upper(), verdict_cls),
            ])
        with chrome.panel("What this means", key="s_mean"):
            chrome.note(
                f"ADF H₀ = unit root; p = {adf['p']:.4f} → {adf['verdict']}. "
                f"KPSS H₀ = stationary; p = {kpss['p']:.4f} → {kpss['verdict']}. "
                f"{joint['detail'].capitalize()}."
            )

    # --- row 2: ACF / PACF -------------------------------------------------
    ap = tsa.acf_pacf(series, int(nlags))
    d1, d2 = st.columns(2)
    with d1:
        with chrome.panel("ACF", key="s_acf", meta=f"{int(nlags)} lags · ±{ap['conf']:.3f}",
                          footer="slow decay ⇒ differencing needed (the d in ARIMA)"):
            st.plotly_chart(charts.stems(ap["lags"], ap["acf"], ap["conf"], height=215),
                            config=CFG, width="stretch")
    with d2:
        with chrome.panel("PACF", key="s_pacf", meta=f"{int(nlags)} lags · ±{ap['conf']:.3f}",
                          footer="a sharp cutoff at lag k suggests AR(k)"):
            st.plotly_chart(charts.stems(ap["lags"], ap["pacf"], ap["conf"], height=215),
                            config=CFG, width="stretch")

    # --- row 3: differencing ladder + STL ---------------------------------
    e1, e2 = st.columns([1, 1.6])
    with e1:
        rows = []
        for d in range(0, 3):
            s_d = close if d == 0 else tsa.difference(close, d)
            j = tsa.joint_stationarity(s_d)
            rows.append((f"d = {d}", j["conclusion"],
                         "up" if j["conclusion"] == "stationary"
                         else "dn" if j["conclusion"] == "non-stationary" else "info"))
        with chrome.panel("Differencing Ladder", key="s_diff", meta="on raw price",
                          footer="the first d that reaches stationarity is the ARIMA d"):
            chrome.kv(rows)
            chrome.note("Prices are integrated of order 1 in almost every "
                        "equity series: levels are non-stationary, first "
                        "differences are not. That is why models are fitted "
                        "to returns rather than prices.")
    with e2:
        try:
            stl = tsa.stl_decompose(close, int(period))
            with chrome.panel("STL Decomposition", key="s_stl",
                              meta=f"period {int(period)} · robust",
                              footer="trend / seasonal / residual — seasonality in daily equities is usually weak"):
                st.plotly_chart(
                    charts.lines(stl[["trend"]].rename(columns={"trend": "trend"}), height=105),
                    config=CFG, width="stretch")
                st.plotly_chart(
                    charts.lines(stl[["seasonal", "resid"]], height=130),
                    config=CFG, width="stretch")
        except Exception as exc:  # noqa: BLE001
            with chrome.panel("STL Decomposition", key="s_stl", meta="failed"):
                chrome.note(f"STL could not fit: {exc}")
