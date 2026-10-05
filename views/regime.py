"""F6 - Regime Detection (Hurst via fracTime)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from lib import cached, charts, chrome
from lib import regime as rg
from lib import theme as T

CFG = T.PLOTLY_CONFIG


def render(cfg: dict) -> None:
    px_df = cached.ohlcv(cfg["symbol"], cfg["years"], cfg["interval"], source=cfg["source"])
    if px_df.empty:
        with chrome.panel("Regime", key="g_err", meta=cfg["symbol"]):
            chrome.note(f"No data for {cfg['symbol']}.")
        return
    close = px_df["close"]

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        method = st.selectbox("Estimator", ["dfa", "rs"], index=0, key="g_m")
    with c2:
        n_boot = st.number_input("Bootstrap reps", 50, 1000, 250, step=50, key="g_b")
    with c3:
        window = st.number_input("Rolling window", 126, 756, 252, step=21, key="g_w")
    with c4:
        val_reps = st.number_input("Validation reps", 1, 10, 3, step=1, key="g_v")

    res = cached.hurst(close, method=method, n_boot=int(n_boot))

    # --- row 1: the estimate ----------------------------------------------
    a, b = st.columns([1, 2])
    with b:
        cmp_df = cached.hurst_compare(close)
        with chrome.panel("R/S vs DFA", key="g_cmp", meta="same series, same window",
                          footer="fracTime's Analyzer defaults to method='rs'"):
            # 0.5 is the only threshold that matters here, so anchor the
            # scale to it rather than to zero.
            lo = min(0.40, float(cmp_df["hurst"].min()) - 0.05)
            hi = max(0.62, float(cmp_df["hurst"].max()) + 0.06)
            st.plotly_chart(
                charts.bars(list(cmp_df["method"]), list(cmp_df["hurst"]),
                            height=180, threshold=0.5, hline=0.5,
                            yrange=(lo, hi)),
                config=CFG, width="stretch")
            rows = [(f"{r.method}", f"H = {r.hurst:.4f} → {r.regime}",
                     "up" if r.regime == "trending"
                     else "dn" if r.regime == "mean reverting" else "info")
                    for r in cmp_df.itertuples()]
            chrome.kv(rows)
            if cmp_df["regime"].nunique() > 1:
                chrome.note(
                    "The two estimators disagree on this series. Classical R/S "
                    "is biased upward (mean abs error 0.059 against fractional "
                    "Brownian motion with known H, versus 0.009 for DFA), and "
                    "the bias is largest for mean-reverting series — so R/S "
                    "systematically over-reports trends. Prefer DFA."
                )

    with a:
        if res.get("ok"):
            lo, hi = res["ci"]
            cls = ("up" if res["regime"] == "trending"
                   else "dn" if res["regime"] == "mean reverting" else "info")
            with chrome.panel("Hurst Exponent", key="g_h",
                              meta=f"{method.upper()} · n={res['n']:,} · {rg.backend()}",
                              footer="95% CI from a moving-block bootstrap"):
                chrome.kv([
                    ("H", f"{res['hurst']:.4f}"),
                    ("95% CI", f"[{lo:.3f}, {hi:.3f}]"),
                    ("Fractal dim.", f"{res['fractal_dimension']:.4f}"),
                    ("Regime", res["regime"].upper(), cls),
                    ("CI includes 0.5", "YES" if res["straddles_half"] else "no",
                     "dn" if res["straddles_half"] else "up"),
                    ("Bootstrap reps", f"{len(res['boot']):,}", "dim"),
                ])
                chrome.note(res["detail"].capitalize() + ".")
        else:
            with chrome.panel("Hurst Exponent", key="g_h", meta="failed"):
                chrome.note(res.get("error", "unknown error"))

    # --- row 2: bootstrap distribution + rolling ---------------------------
    d1, d2 = st.columns([1, 2])
    with d1:
        if res.get("ok") and len(res["boot"]):
            with chrome.panel("Bootstrap Distribution", key="g_boot",
                              meta=f"{len(res['boot']):,} resamples",
                              footer="0.5 inside the mass ⇒ cannot call a regime"):
                st.plotly_chart(
                    charts.histogram(res["boot"], bins=40, height=200,
                                     overlay_normal=False),
                    config=CFG, width="stretch")
    with d2:
        roll = cached.rolling_hurst(close, int(window), 5, method)
        with chrome.panel("Rolling Hurst", key="g_roll",
                          meta=f"{int(window)}-day window · {method.upper()}",
                          footer="0.5 = random walk; the window must stay long or this is noise"):
            if roll.empty:
                chrome.pending("not enough history for this window")
            else:
                st.plotly_chart(
                    charts.area_single(roll, color=T.SERIES[4], height=200, hline=0.5),
                    config=CFG, width="stretch")

    # --- row 3: validation -------------------------------------------------
    with chrome.panel("Estimator Validation", key="g_val",
                      meta=f"fBm with known H · {int(val_reps)} reps · n=2000",
                      footer="the error of the tool, measured before pointing it at a real asset"):
        val = cached.hurst_validation(int(val_reps), 2000)
        st.plotly_chart(
            charts.lines(val.set_index("true_H")[["RS", "DFA"]], height=210),
            config=CFG, width="stretch")
        rs_err = val["RS_err"].abs().mean()
        dfa_err = val["DFA_err"].abs().mean()
        v1, v2 = st.columns([1, 1])
        with v1:
            chrome.kv([(f"true H = {r.true_H:.1f}",
                        f"RS {r.RS:.3f} ({r.RS_err:+.3f})   DFA {r.DFA:.3f} ({r.DFA_err:+.3f})")
                       for r in val.itertuples()])
        with v2:
            chrome.kv([
                ("Mean |error| R/S", f"{rs_err:.4f}", "dn"),
                ("Mean |error| DFA", f"{dfa_err:.4f}", "up"),
                ("DFA improvement", f"{rs_err / dfa_err:.1f}×" if dfa_err > 0 else "n/a", "up"),
            ])
            chrome.note(
                "Series are generated by Davies–Harte circulant embedding, so "
                "the true H is known exactly. A diagonal line would be a "
                "perfect estimator. R/S sits above the diagonal at low H — it "
                "under-detects mean reversion, which is precisely the regime "
                "it would be used to find."
            )
