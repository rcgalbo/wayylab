"""F5 - Volatility Modelling."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from lib import cached, charts, chrome, data, stats, vol
from lib import theme as T

CFG = T.PLOTLY_CONFIG


def render(cfg: dict) -> None:
    px_df = cached.ohlcv(cfg["symbol"], cfg["years"], cfg["interval"], source=cfg["source"])
    if px_df.empty:
        with chrome.panel("Volatility", key="v_err", meta=cfg["symbol"]):
            chrome.note(f"No data for {cfg['symbol']}.")
        return

    close = px_df["close"]
    rets = data.log_returns(close)

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        window = st.number_input("Realised window", 5, 126, 21, step=1, key="v_w")
    with c2:
        lam = st.number_input("EWMA λ", 0.80, 0.995, 0.94, step=0.01, key="v_l")
    with c3:
        dist = st.selectbox("GARCH dist", ["t", "normal", "skewt"], key="v_d")
    with c4:
        horizon = st.number_input("Forecast days", 5, 126, 21, step=1, key="v_h")

    realised = vol.realised_vol(rets, int(window))
    ewma = vol.ewma_vol(rets, float(lam))
    park = vol.parkinson_vol(px_df, int(window))
    fit = cached.garch(rets, dist=dist)

    # --- row 1: the estimators together -----------------------------------
    comp = pd.DataFrame({
        f"realised {int(window)}d": realised,
        f"EWMA λ={lam:.2f}": ewma,
        "Parkinson": park,
    })
    if fit.get("ok"):
        comp["GARCH(1,1)"] = fit["cond_vol"]
    comp = comp.dropna(how="all")

    a, b = st.columns([2.4, 1])
    with a:
        with chrome.panel("Volatility Estimators", key="v_cmp",
                          meta=f"{cfg['symbol']} · annualised",
                          footer="all four annualised at √252 · same series, different estimators"):
            st.plotly_chart(charts.lines(comp, height=270, yfmt=".0%"),
                            config=CFG, width="stretch")
    with b:
        with chrome.panel("Current Levels", key="v_now", meta="latest",
                          footer="Parkinson uses the high-low range"):
            rows = [
                (f"Realised {int(window)}d", f"{_last(realised):.2%}"),
                (f"EWMA λ={lam:.2f}", f"{_last(ewma):.2%}"),
                ("Parkinson", f"{_last(park):.2%}"),
            ]
            if fit.get("ok"):
                rows += [
                    ("GARCH cond.", f"{_last(fit['cond_vol']):.2%}"),
                    ("GARCH long-run", f"{fit['long_run_vol']:.2%}", "info"),
                ]
            rows.append(("Full-sample", f"{stats.ann_vol(rets):.2%}", "dim"))
            chrome.kv(rows)

    # --- row 2: GARCH detail ----------------------------------------------
    d1, d2 = st.columns([1, 2])
    with d1:
        if fit.get("ok"):
            pers = fit["persistence"]
            with chrome.panel("GARCH Fit", key="v_fit", meta=fit["model"],
                              footer="α+β is persistence — how slowly shocks decay"):
                chrome.kv([
                    ("ω (omega)", f"{fit['omega']:.5f}"),
                    ("α₁", f"{fit['alpha'][0]:.4f}"),
                    ("β₁", f"{fit['beta'][0]:.4f}"),
                    ("Persistence α+β", f"{pers:.4f}",
                     "dn" if pers >= 0.99 else "up"),
                    ("Shock half-life", f"{fit['half_life']:.1f} d"
                     if np.isfinite(fit["half_life"]) else "∞"),
                    ("ν (t d.o.f.)", f"{fit['nu']:.2f}" if fit.get("nu") else "n/a"),
                    ("AIC", f"{fit['aic']:,.0f}", "dim"),
                    ("Stationary", "yes" if fit["stationary"] else "NO",
                     "up" if fit["stationary"] else "dn"),
                ])
        else:
            with chrome.panel("GARCH Fit", key="v_fit", meta="failed"):
                chrome.note(f"Could not fit: {fit.get('error')}")

    with d2:
        if fit.get("ok"):
            fc = vol.garch_forecast(fit, int(horizon))
            hist = fit["cond_vol"].tail(120)
            band = pd.DataFrame({"GARCH forecast": fc})
            with chrome.panel("Conditional Volatility Forecast", key="v_fc",
                              meta=f"{int(horizon)} days ahead",
                              footer=f"mean-reverts toward the long-run level {fit['long_run_vol']:.1%}"):
                st.plotly_chart(charts.lines(band, height=215, yfmt=".0%",
                                             hline=fit["long_run_vol"]),
                                config=CFG, width="stretch")
                chrome.note(
                    f"Starting from {_last(fit['cond_vol']):.1%}, the model pulls "
                    f"volatility toward {fit['long_run_vol']:.1%} with a half-life of "
                    f"{fit['half_life']:.0f} trading days. Persistence is "
                    f"{fit['persistence']:.3f}; at 1.0 shocks would never decay."
                )

    # --- row 3: the evidence for clustering --------------------------------
    cl = vol.clustering_evidence(rets)
    e1, e2 = st.columns([1.6, 1])
    with e1:
        sq = pd.DataFrame({"squared log return": rets.pow(2)})
        with chrome.panel("Volatility Clustering", key="v_clus",
                          meta="squared log returns",
                          footer="big moves arrive next to other big moves"):
            st.plotly_chart(charts.area_single(sq["squared log return"],
                                               color=T.SERIES[2], height=180),
                            config=CFG, width="stretch")
    with e2:
        with chrome.panel("Ljung-Box Evidence", key="v_lb", meta="20 lags",
                          footer="the asymmetry is the justification for GARCH"):
            chrome.kv([
                ("Returns Q", f"{cl['returns']['stat']:.1f}"),
                ("Returns p", f"{cl['returns']['p']:.3f}",
                 "dn" if cl["returns"]["p"] < 0.05 else "up"),
                ("Squared Q", f"{cl['squared']['stat']:,.0f}"),
                ("Squared p", f"{cl['squared']['p']:.2e}",
                 "dn" if cl["squared"]["p"] < 0.05 else "up"),
                ("Clustering", "YES" if cl["clustering"] else "no",
                 "dn" if cl["clustering"] else "up"),
            ])
            chrome.note(cl["summary"].capitalize() + ". Returns show little "
                        "autocorrelation while their squares show a great deal — "
                        "unforecastable direction, forecastable magnitude.")


def _last(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")
