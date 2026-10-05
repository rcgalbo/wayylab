"""F8 - Forecast Evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from lib import cached, charts, chrome
from lib import forecast as fc
from lib import theme as T

CFG = T.PLOTLY_CONFIG


def render(cfg: dict) -> None:
    px_df = cached.ohlcv(cfg["symbol"], cfg["years"], cfg["interval"], source=cfg["source"])
    if px_df.empty:
        with chrome.panel("Forecast", key="f_err", meta=cfg["symbol"]):
            chrome.note(f"No data for {cfg['symbol']}.")
        return
    close = px_df["close"]

    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    with c1:
        p = st.number_input("AR (p)", 0, 5, 1, key="f_p")
    with c2:
        d = st.number_input("Diff (d)", 0, 2, 1, key="f_d")
    with c3:
        q = st.number_input("MA (q)", 0, 5, 1, key="f_q")
    with c4:
        train = st.number_input("Train window", 120, 1000, 400, step=20, key="f_t")
    with c5:
        origins = st.number_input("Walk-fwd origins", 20, 250, 80, step=10, key="f_o")
    order = (int(p), int(d), int(q))

    # --- row 1: in-sample fit + forward forecast ---------------------------
    fit = fc.fit_arima(close, order=order, horizon=21)
    a, b = st.columns([2.2, 1])
    with a:
        with chrome.panel("ARIMA Forecast", key="f_fan",
                          meta=f"{cfg['symbol']} · ARIMA{order} · 21 steps",
                          footer="shaded band = 95% interval; it widens with horizon for a reason"):
            if not fit.get("ok"):
                chrome.note(fit.get("error", "fit failed"))
            else:
                st.plotly_chart(
                    charts.band(fit["forecast"], fit["lower"], fit["upper"],
                                color=T.SERIES[1], height=265,
                                history=close.tail(180)),
                    config=CFG, width="stretch")
    with b:
        with chrome.panel("Model", key="f_mod", meta=f"ARIMA{order}",
                          footer="AIC/BIC compare fit, not forecast skill"):
            if fit.get("ok"):
                chrome.kv([
                    ("Order", f"({order[0]},{order[1]},{order[2]})"),
                    ("AIC", f"{fit['aic']:,.1f}"),
                    ("BIC", f"{fit['bic']:,.1f}"),
                    ("Next step", f"{fit['forecast'].iloc[0]:,.2f}"),
                    ("Last close", f"{close.iloc[-1]:,.2f}", "dim"),
                    ("Implied move",
                     f"{fit['forecast'].iloc[0] / close.iloc[-1] - 1:+.2%}",
                     "up" if fit["forecast"].iloc[0] >= close.iloc[-1] else "dn"),
                    ("Band width (21d)",
                     f"±{(fit['upper'].iloc[-1] - fit['lower'].iloc[-1]) / 2 / close.iloc[-1]:.1%}"),
                ])

    # --- row 2: order search ----------------------------------------------
    with st.expander("Order search (AIC grid)", expanded=False):
        ao = cached.auto_order(close.tail(700), 3, 3, int(d))
        if ao.get("ok"):
            g1, g2 = st.columns([2, 1])
            with g1:
                st.dataframe(ao["table"].head(10).round(2), width="stretch",
                             height=220, hide_index=True)
            with g2:
                chrome.kv([("Best by AIC", f"ARIMA{ao['best']}"),
                           ("Orders tried", f"{len(ao['table'])}", "dim")])
                chrome.note(
                    "Note how little AIC separates the top orders. A flat "
                    "criterion means the data is not strongly preferring any "
                    "structure — itself a result worth reporting."
                )

    # --- row 3: walk-forward, the honest test ------------------------------
    with chrome.panel("Walk-Forward Evaluation", key="f_wf",
                      meta=f"rolling origin · train {int(train)} · {int(origins)} refits",
                      footer="out of sample: nothing after each origin is visible to the fit"):
        with st.spinner("refitting at each origin…"):
            wf = cached.walk_forward(close, order, int(train), 3, int(origins))
        if not wf.get("ok"):
            chrome.note(wf.get("error", "failed"))
            return

        w1, w2 = st.columns([2, 1])
        with w1:
            frame = wf["frame"][["actual", "arima", "naive"]]
            st.plotly_chart(charts.lines(frame, height=230), config=CFG, width="stretch")
        with w2:
            rows = []
            for name, label in (("arima", f"ARIMA{order}"), ("naive", "Random walk"),
                                ("drift", "Drift")):
                m = wf[name]
                rows.append((label, f"RMSE {m['rmse']:.3f}"))
            rows.append(("—", "—", "dim"))
            for name, label in (("arima", "ARIMA dir."), ("naive", "Naive dir."),
                                ("drift", "Drift dir.")):
                v = wf[name]["dir_acc"]
                rows.append((label, "n/a" if not np.isfinite(v) else f"{v:.1%}",
                             "up" if np.isfinite(v) and v > 0.5 else "dn"))
            chrome.kv(rows)

        v1, v2 = st.columns([1, 1])
        with v1:
            chrome.kv([
                ("Theil's U", f"{wf['theil_u']:.4f}",
                 "up" if wf["beats_naive"] else "dn"),
                ("Beats random walk", "YES" if wf["beats_naive"] else "NO",
                 "up" if wf["beats_naive"] else "dn"),
                ("DM statistic", f"{wf['dm'].get('stat', float('nan')):+.3f}"
                 if wf["dm"].get("ok") else "n/a"),
                ("DM p-value", f"{wf['dm'].get('p', float('nan')):.3f}"
                 if wf["dm"].get("ok") else "n/a"),
                ("Difference real?",
                 ("yes" if wf["dm"].get("significant") else "no")
                 if wf["dm"].get("ok") else "n/a",
                 "up" if wf["dm"].get("significant") else "dn"),
            ])
        with v2:
            u = wf["theil_u"]
            dm = wf["dm"]
            verdict = (
                f"Theil's U = {u:.3f}. "
                + ("Below 1, so ARIMA has lower out-of-sample error than simply "
                   "predicting no change. " if u < 1 else
                   "Above 1, so ARIMA is **worse** than simply predicting no "
                   "change. ")
            )
            if dm.get("ok"):
                verdict += (
                    f"Diebold–Mariano p = {dm['p']:.3f}, so the gap "
                    + ("is statistically significant."
                       if dm["significant"] else
                       "is **not** distinguishable from sampling noise.")
                )
            chrome.note(verdict)

        chrome.note(
            "This is the test that matters. In-sample fit and a confident-looking "
            "fan chart tell you nothing about forecast skill — a random walk is a "
            "genuinely hard benchmark on daily equity prices, and models that look "
            "impressive in-sample routinely fail to beat it out of sample. "
            "Reporting that honestly is the finding."
        )
