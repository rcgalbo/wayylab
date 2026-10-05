"""F1 - Overview workspace."""
from __future__ import annotations

import streamlit as st

from lib import cached, charts, chrome, data, stats
from lib import theme as T

CFG = T.PLOTLY_CONFIG
TILE_SYMBOLS = ["SPY", "QQQ", "AAPL", "NVDA", "TLT"]
PANEL_SYMBOLS = ("SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TLT")


def render(cfg: dict) -> None:
    px = cached.ohlcv(cfg["symbol"], cfg["years"], cfg["interval"], source=cfg["source"])

    tiles = []
    for s in TILE_SYMBOLS:
        q = data.quote_from(cached.ohlcv(s, years=1.0, source=cfg["source"]))
        tiles.append({"label": s,
                      "value": f"{q['price']:,.2f}" if q["price"] else "—",
                      "change_pct": q["change_pct"]})
    chrome.stat_tiles(tiles)

    if px.empty:
        with chrome.panel("Price", key="err", meta=cfg["symbol"]):
            chrome.note(f"No data returned for {cfg['symbol']}. "
                        "Check the ticker, or switch Source to snapshot.")
        return

    rets = data.log_returns(px["close"])
    dd = stats.max_drawdown(px["close"])
    m = stats.moments(rets)

    left, right = st.columns([2.6, 1])
    with left:
        ma = {f"SMA{w}": px["close"].rolling(w).mean() for w in (20, 50, 200)}
        ma = {k: v for k, v in ma.items() if v.notna().any()}
        rev = f"{cfg['symbol']}|{cfg['interval']}|{cfg['years']}"
        with chrome.panel("Price Chart", key="chart",
                          meta=f"{cfg['symbol']} · {cfg['interval']} · {len(px)} bars",
                          footer="range buttons or drag a box to zoom · double-click to reset"):
            st.plotly_chart(
                charts.price_volume(px, ma, height=392, uirevision=rev),
                config=T.PLOTLY_CONFIG_PINNED, width="stretch", key="ov_pricevol")

    with right:
        q = data.quote_from(px)
        with chrome.panel("Session", key="session", meta=cfg["symbol"]):
            chrome.kv([
                ("Last", f"{q['price']:,.2f}"),
                ("Change", f"{q['change_pct']:+.2f}%",
                 "up" if (q["change_pct"] or 0) >= 0 else "dn"),
                ("As of", f"{q['asof']:%Y-%m-%d}" if q["asof"] is not None else "—", "dim"),
                ("Range", f"{px['low'].iloc[-1]:,.2f} – {px['high'].iloc[-1]:,.2f}"),
                ("Volume", f"{px['volume'].iloc[-1]:,.0f}"),
            ])
        with chrome.panel("Risk", key="risk", meta=f"{int(cfg['years'])}Y",
                          footer="annualised · 252d"):
            ar = stats.ann_return(rets)
            chrome.kv([
                ("Ann. return", f"{ar:+.2%}", "up" if ar >= 0 else "dn"),
                ("Ann. vol", f"{stats.ann_vol(rets):.2%}"),
                ("Sharpe", f"{stats.sharpe(rets):.2f}"),
                ("Sortino", f"{stats.sortino(rets):.2f}"),
                ("Max DD", f"{dd['depth']:.2%}", "dn"),
                ("VaR 95%", f"{stats.var_historical(rets):.2%}", "dn"),
                ("CVaR 95%", f"{stats.cvar_historical(rets):.2%}", "dn"),
            ])
        bt = stats.var_backtest(rets, 0.05, 250)
        with chrome.panel("VaR Backtest", key="varbt", meta="rolling 250d",
                          footer="a correct 95% model breaches ~5% of days"):
            chrome.kv([
                ("Expected rate", f"{bt['expected']:.2%}"),
                ("Realised rate", f"{bt['realised']:.2%}"
                 if bt["realised"] == bt["realised"] else "n/a",
                 "up" if abs((bt["realised"] or 0) - 0.05) < 0.015 else "dn"),
                ("Breaches", f"{bt['breaches']} / {bt['n']}"),
                ("Kupiec LR", f"{bt['kupiec_lr']:.2f}"
                 if bt["kupiec_lr"] == bt["kupiec_lr"] else "n/a"),
                ("Well calibrated",
                 "yes" if (bt["kupiec_lr"] == bt["kupiec_lr"] and bt["kupiec_lr"] < 3.84)
                 else "no",
                 "up" if (bt["kupiec_lr"] == bt["kupiec_lr"] and bt["kupiec_lr"] < 3.84)
                 else "dn"),
            ])

    a, b, c = st.columns([1, 1, 1.15])
    with a:
        with chrome.panel("Return Distribution", key="hist", meta="vs fitted normal",
                          footer=f"excess kurtosis {m['excess_kurtosis']:+.2f} — fat tails"):
            st.plotly_chart(charts.histogram(rets.values, height=232),
                            config=CFG, width="stretch")
    with b:
        with chrome.panel("Drawdown", key="dd", meta=f"max {dd['depth']:.1%}",
                          footer=f"trough {dd['trough']:%Y-%m-%d}" if dd["trough"] is not None else ""):
            st.plotly_chart(charts.area_single(dd["curve"], color=T.NEGATIVE, height=232),
                            config=CFG, width="stretch")
    with c:
        panel_px = cached.panel(PANEL_SYMBOLS, years=cfg["years"], source=cfg["source"])
        with chrome.panel("Correlation", key="corr",
                          meta=f"{len(PANEL_SYMBOLS)} assets · log returns",
                          footer="diverging scale · grey = uncorrelated"):
            if panel_px.shape[1] >= 2:
                st.plotly_chart(charts.heatmap(stats.correlation(panel_px), height=232),
                                config=CFG, width="stretch")
            else:
                chrome.pending("insufficient data")
