"""F3 - Returns & Distribution."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from lib import cached, charts, chrome, data, stats, tsa
from lib import theme as T

CFG = T.PLOTLY_CONFIG


def render(cfg: dict) -> None:
    px_df = cached.ohlcv(cfg["symbol"], cfg["years"], cfg["interval"], source=cfg["source"])
    if px_df.empty:
        with chrome.panel("Returns", key="r_err", meta=cfg["symbol"]):
            chrome.note(f"No data for {cfg['symbol']}.")
        return

    close = px_df["close"]
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        kind = st.selectbox("Return type", ["log", "simple"], key="rk")
    with c2:
        freq = st.selectbox("Frequency", ["daily", "weekly", "monthly"], key="rf")
    with c3:
        roll = st.number_input("Rolling window", 5, 252, 21, step=1, key="rw")

    s = close.resample({"daily": "D", "weekly": "W", "monthly": "ME"}[freq]).last().dropna() \
        if freq != "daily" else close
    rets = data.log_returns(s) if kind == "log" else s.pct_change().dropna()
    per = {"daily": 252, "weekly": 52, "monthly": 12}[freq]

    m = stats.moments(rets)
    cum = (np.exp(rets.cumsum()) if kind == "log" else (1 + rets).cumprod()) - 1.0

    # --- row 1 -------------------------------------------------------------
    a, b = st.columns([2, 1])
    with a:
        with chrome.panel("Cumulative Return", key="r_cum",
                          meta=f"{cfg['symbol']} · {kind} · {freq}",
                          footer=f"total {cum.iloc[-1]:+.1%} over {len(rets):,} {freq} periods"):
            st.plotly_chart(charts.area_single(cum, color=T.SERIES[0], height=240, hline=0.0),
                            config=CFG, width="stretch")
    with b:
        with chrome.panel("Moments", key="r_mom", meta=f"{kind} returns",
                          footer="JB tests normality; equities almost always reject"):
            chrome.kv([
                ("Mean", f"{m['mean']:+.5f}"),
                ("Std dev", f"{m['sd']:.5f}"),
                ("Annualised μ", f"{stats.ann_return(rets, per):+.2%}",
                 "up" if stats.ann_return(rets, per) >= 0 else "dn"),
                ("Annualised σ", f"{stats.ann_vol(rets, per):.2%}"),
                ("Skew", f"{m['skew']:+.3f}",
                 "dn" if m["skew"] < 0 else "up"),
                ("Excess kurtosis", f"{m['excess_kurtosis']:+.3f}",
                 "dn" if m["excess_kurtosis"] > 1 else ""),
                ("Jarque-Bera", f"{m['jb']:,.0f}"),
                ("JB p-value", f"{m['jb_p']:.2e}", "dn" if m["jb_p"] < 0.05 else "up"),
                ("Normality", "REJECTED" if m["jb_p"] < 0.05 else "not rejected",
                 "dn" if m["jb_p"] < 0.05 else "up"),
            ])

    # --- row 2: distribution ----------------------------------------------
    d1, d2, d3 = st.columns(3)
    with d1:
        with chrome.panel("Histogram", key="r_hist", meta="vs fitted normal",
                          footer="the peak is taller and the tails fatter than normal"):
            st.plotly_chart(charts.histogram(rets.values, bins=70, height=230),
                            config=CFG, width="stretch")
    with d2:
        with chrome.panel("Normal Q-Q", key="r_qq", meta="quantile comparison",
                          footer="points bending off the line at both ends = fat tails"):
            st.plotly_chart(charts.qq(rets.values, height=230), config=CFG, width="stretch")
    with d3:
        sd = m["sd"]
        tail_obs = {k: int((rets.abs() > k * sd).sum()) for k in (2, 3, 4, 5)}
        from scipy import stats as sps
        tail_exp = {k: float(2 * (1 - sps.norm.cdf(k)) * len(rets)) for k in (2, 3, 4, 5)}
        with chrome.panel("Tail Counts", key="r_tail", meta="observed vs normal",
                          footer="how many σ-events actually happened"):
            chrome.kv([(f"|r| > {k}σ", f"{tail_obs[k]:,} obs / {tail_exp[k]:.1f} exp",
                        "dn" if tail_obs[k] > tail_exp[k] * 1.5 else "")
                       for k in (2, 3, 4, 5)]
                      + [("Worst day", f"{rets.min():.2%}", "dn"),
                         ("Best day", f"{rets.max():+.2%}", "up")])

    # --- row 3: rolling + autocorrelation ---------------------------------
    e1, e2 = st.columns(2)
    with e1:
        roll_sd = (rets.rolling(int(roll)).std(ddof=1) * np.sqrt(per)).dropna()
        roll_mu = (rets.rolling(int(roll)).mean() * per).dropna()
        # Separate charts, not a second y-axis: annualised μ swings by ±2
        # while σ sits near 0.3, so a shared scale flattens σ to a line.
        with chrome.panel("Rolling Moments", key="r_roll",
                          meta=f"{int(roll)}-period · annualised · separate scales",
                          footer="μ is noisy and centres on zero; σ is persistent"):
            st.plotly_chart(
                charts.area_single(roll_sd.rename(f"σ ({int(roll)})"),
                                   color=T.SERIES[0], height=115),
                config=CFG, width="stretch")
            st.plotly_chart(
                charts.area_single(roll_mu.rename(f"μ ({int(roll)})"),
                                   color=T.SERIES[1], height=115, hline=0.0),
                config=CFG, width="stretch")
    with e2:
        ap = tsa.acf_pacf(rets, 30)
        ap2 = tsa.acf_pacf(rets.pow(2), 30)
        tab1, tab2 = st.tabs(["returns", "squared returns"])
        with tab1:
            with chrome.panel("ACF — returns", key="r_acf",
                              meta=f"±{ap['conf']:.3f} band",
                              footer="mostly inside the band: returns are close to unpredictable"):
                st.plotly_chart(charts.stems(ap["lags"], ap["acf"], ap["conf"], height=200),
                                config=CFG, width="stretch")
        with tab2:
            with chrome.panel("ACF — squared returns", key="r_acf2",
                              meta=f"±{ap2['conf']:.3f} band",
                              footer="persistently outside the band: volatility is predictable"):
                st.plotly_chart(charts.stems(ap2["lags"], ap2["acf"], ap2["conf"], height=200),
                                config=CFG, width="stretch")

    lb_r = tsa.ljung_box(rets, 20)
    lb_r2 = tsa.ljung_box(rets.pow(2), 20)
    with chrome.panel("Reading", key="r_read", meta="Ljung-Box, 20 lags"):
        chrome.note(
            f"Returns: Q p = {lb_r['p']:.3f} → {lb_r['verdict']}.  "
            f"Squared returns: Q p = {lb_r2['p']:.2e} → {lb_r2['verdict']}.  "
            "That asymmetry is the whole motivation for volatility models: the "
            "direction of the next move is near-unforecastable while its "
            "magnitude is not. F5 fits a GARCH to exactly this."
        )
