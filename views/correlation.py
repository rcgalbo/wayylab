"""F7 - Correlation, Cointegration & Pairs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from lib import cached, charts, chrome, pairs
from lib import theme as T

CFG = T.PLOTLY_CONFIG
DEFAULT_UNIVERSE = "SPY,QQQ,AAPL,MSFT,NVDA,TLT,GLD,XLE"


def render(cfg: dict) -> None:
    c1, c2 = st.columns([3, 1])
    with c1:
        raw = st.text_input("Universe (comma separated)", DEFAULT_UNIVERSE, key="x_u")
    with c2:
        method = st.selectbox("Correlation", ["pearson", "spearman"], key="x_m")

    syms = tuple(dict.fromkeys(
        s.strip().upper() for s in raw.split(",") if s.strip()))
    if len(syms) < 2:
        with chrome.panel("Universe", key="x_err"):
            chrome.note("Enter at least two symbols.")
        return

    closes = cached.panel(syms, years=cfg["years"], source=cfg["source"])
    if closes.shape[1] < 2:
        with chrome.panel("Universe", key="x_err2"):
            chrome.note("Could not fetch at least two of those symbols.")
        return

    # --- row 1: correlation + PCA -----------------------------------------
    a, b = st.columns([1.5, 1])
    with a:
        cm = pairs.corr_matrix(closes, method=method)
        with chrome.panel("Correlation Matrix", key="x_corr",
                          meta=f"{cm.shape[0]} assets · {method} · log returns",
                          footer="diverging scale · grey = uncorrelated"):
            st.plotly_chart(charts.heatmap(cm, height=290), config=CFG, width="stretch")
    with b:
        p = pairs.pca(closes)
        with chrome.panel("PCA", key="x_pca", meta="of the correlation matrix",
                          footer="PC1 is the common 'market' factor"):
            if not p.get("ok"):
                chrome.note(p.get("error", "failed"))
            else:
                st.plotly_chart(
                    charts.bars([f"PC{i+1}" for i in range(len(p["explained"]))],
                                list(p["explained"]), height=150, yfmt=".0%"),
                    config=CFG, width="stretch")
                chrome.kv([
                    ("PC1 explains", f"{p['pc1_share']:.1%}", "info"),
                    ("PC1–PC3", f"{p['cumulative'][min(2, len(p['cumulative'])-1)]:.1%}"),
                    ("Assets", f"{p['n_assets']}", "dim"),
                ])
                chrome.note(
                    f"A single factor accounts for {p['pc1_share']:.0%} of the "
                    "variance across this universe. Diversification claims should "
                    "be read against that number."
                )

    # --- row 2: pairs scanner ---------------------------------------------
    with chrome.panel("Pairs Scanner", key="x_scan",
                      meta="Engle–Granger on log prices, both directions",
                      footer="ranked by ADF p-value on the spread"):
        scan = cached.scan_pairs(closes)
        if scan.empty:
            chrome.pending("no pairs with enough overlapping history")
        else:
            bonf = scan.attrs.get("bonferroni", 0.05)
            n_tests = scan.attrs.get("n_tests", len(scan))
            show = scan.head(10).copy()
            show["survives"] = show["p"] < bonf
            st.dataframe(
                show[["pair", "beta", "p", "adf", "half_life", "z",
                      "cointegrated", "survives"]].round(4),
                width="stretch", height=240, hide_index=True)
            n_sig = int((scan["p"] < 0.05).sum())
            n_bonf = int((scan["p"] < bonf).sum())
            chrome.note(
                f"{n_tests} pairs tested. {n_sig} significant at p < 0.05, but "
                f"only {n_bonf} survive the Bonferroni threshold of {bonf:.5f}. "
                f"At α = 0.05 roughly {0.05 * n_tests:.1f} false positives are "
                "expected by chance alone — which is why the raw count of "
                "'cointegrated' pairs is not the interesting number."
            )

    # --- row 3: spread detail ---------------------------------------------
    opts = list(closes.columns)
    s1, s2, s3 = st.columns([1, 1, 1])
    with s1:
        ya = st.selectbox("Leg Y", opts, index=0, key="x_y")
    with s2:
        xb = st.selectbox("Leg X", opts, index=min(1, len(opts) - 1), key="x_x")
    with s3:
        zwin = st.number_input("Z-score window", 20, 252, 60, step=5, key="x_z")

    if ya == xb:
        with chrome.panel("Spread", key="x_sp"):
            chrome.note("Pick two different legs.")
    else:
        eg = pairs.engle_granger(closes[ya], closes[xb])
        d1, d2 = st.columns([2, 1])
        with d1:
            if eg.get("ok"):
                sp = pairs.spread_series(closes[ya], closes[xb], eg["beta"],
                                         eg["alpha"], int(zwin))
                with chrome.panel(f"Spread  {ya} − β·{xb}", key="x_spread",
                                  meta=f"β = {eg['beta']:.4f} · R² = {eg['r2']:.3f}",
                                  footer="z-score crossing ±2 is the classic entry rule"):
                    st.plotly_chart(
                        charts.area_single(sp["z"].dropna(), color=T.SERIES[3],
                                           height=215, hline=0.0),
                        config=CFG, width="stretch")
            else:
                with chrome.panel("Spread", key="x_spread", meta="failed"):
                    chrome.note(eg.get("error", "failed"))
        with d2:
            if eg.get("ok"):
                with chrome.panel("Cointegration", key="x_eg", meta="Engle–Granger",
                                  footer="ADF on the regression residual"):
                    chrome.kv([
                        ("Hedge ratio β", f"{eg['beta']:.4f}"),
                        ("R²", f"{eg['r2']:.4f}"),
                        ("ADF statistic", f"{eg['adf_stat']:.3f}"),
                        ("p-value", f"{eg['p']:.4f}",
                         "up" if eg["cointegrated"] else "dn"),
                        ("Half-life", f"{eg['half_life']:.1f} d"
                         if np.isfinite(eg["half_life"]) else "∞"),
                        ("Current z", f"{eg['zscore']:+.2f}",
                         "dn" if abs(eg["zscore"]) > 2 else ""),
                        ("Cointegrated", "YES" if eg["cointegrated"] else "no",
                         "up" if eg["cointegrated"] else "dn"),
                    ])

    # --- row 4: Granger ----------------------------------------------------
    g1, g2, g3 = st.columns([1, 1, 1])
    with g1:
        gc = st.selectbox("Cause", opts, index=0, key="x_gc")
    with g2:
        ge = st.selectbox("Effect", opts, index=min(1, len(opts) - 1), key="x_ge")
    with g3:
        glag = st.number_input("Max lag", 2, 20, 10, step=1, key="x_gl")

    with chrome.panel("Granger Causality", key="x_granger",
                      meta=f"{gc} → {ge} · on log returns",
                      footer="predictive precedence, not causation"):
        if gc == ge:
            chrome.note("Pick two different series.")
        else:
            rets = pairs.returns_panel(closes)
            g = pairs.granger(rets[gc], rets[ge], maxlag=int(glag))
            if not g.get("ok"):
                chrome.note(g.get("error", "failed"))
            else:
                gg1, gg2 = st.columns([2, 1])
                with gg1:
                    st.plotly_chart(
                        charts.bars([str(l) for l in g["table"]["lag"]],
                                    list(g["table"]["p"]), height=185,
                                    color=T.SERIES[1]),
                        config=CFG, width="stretch")
                with gg2:
                    chrome.kv([
                        ("Best lag", f"{g['best_lag']}"),
                        ("F statistic", f"{g['best_F']:.3f}"),
                        ("p-value", f"{g['best_p']:.4f}",
                         "up" if g["significant"] else "dn"),
                        ("Significant", "yes" if g["significant"] else "no",
                         "up" if g["significant"] else "dn"),
                        ("Bonferroni α", f"{g['bonferroni']:.4f}", "dim"),
                        ("Survives", "YES" if g["survives_bonferroni"] else "no",
                         "up" if g["survives_bonferroni"] else "dn"),
                    ])
                chrome.note(
                    f"Lowest p-value is {g['best_p']:.4f} at lag {g['best_lag']}. "
                    f"Because {int(glag)} lags were tested, the honest threshold is "
                    f"{g['bonferroni']:.4f}, which this "
                    f"{'does' if g['survives_bonferroni'] else 'does not'} clear. "
                    "Granger causality is about predictive precedence only."
                )
