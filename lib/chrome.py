"""
Terminal chrome primitives: title bar, function-key nav, bordered panels,
stat tiles, dense key/value readouts, status bar.

Mirrors the React app's .app-titlebar / .function-bar / .panel / .status-bar
components so the two surfaces look like the same product.
"""
from __future__ import annotations

import html
from contextlib import contextmanager
from datetime import datetime, timezone

import streamlit as st

from . import theme

# (key, F-label, title) - order defines the function-key bar
PAGES: list[tuple[str, str, str]] = [
    ("overview", "F1", "Overview"),
    ("chart", "F2", "Chart"),
    ("returns", "F3", "Returns"),
    ("structure", "F4", "Structure"),
    ("vol", "F5", "Volatility"),
    ("regime", "F6", "Regime"),
    ("corr", "F7", "Correlation"),
    ("forecast", "F8", "Forecast"),
    ("analysis", "F9", "Analysis"),
]


def boot(title: str = "WAYY LAB") -> None:
    """set_page_config + inject the stylesheet. Call once, first thing."""
    st.set_page_config(
        page_title=title,
        page_icon="▲",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(theme.load_css(), unsafe_allow_html=True)


def titlebar(market_open: bool, source_label: str, source_kind: str = "live") -> None:
    now = datetime.now(timezone.utc).astimezone()
    dot = "wl-dot" if market_open else "wl-dot off"
    state = "MARKET OPEN" if market_open else "MARKET CLOSED"
    tag = "live" if source_kind == "live" else "snap"
    st.markdown(
        f"""<div class="wl-titlebar">
  <div class="wl-logo">WAYY<span class="accent">·</span>LAB
    <span class="sub">Financial Time Series Research</span></div>
  <div class="wl-tb-right">
    <span class="wl-tag {tag}">{html.escape(source_label)}</span>
    <span><span class="{dot}"></span>{state}</span>
    <span class="mono">{now:%Y-%m-%d %H:%M:%S %Z}</span>
  </div>
</div>""",
        unsafe_allow_html=True,
    )


def function_bar(active: str) -> str:
    """
    Render the F-key nav. Returns the (possibly updated) active page key.
    Uses button `type` to mark the active key, which CSS paints Wayy red.
    """
    if "page" not in st.session_state:
        st.session_state.page = active

    cols = st.columns(len(PAGES))
    for col, (key, fkey, label) in zip(cols, PAGES):
        with col:
            if st.button(
                f"{fkey} {label}",
                key=f"fkey_{key}",
                type="primary" if st.session_state.page == key else "secondary",
                width="stretch",
            ):
                st.session_state.page = key
                st.rerun()
    return st.session_state.page


@contextmanager
def panel(title: str, key: str, meta: str = "", icon: str = "▪", footer: str = ""):
    """
    A bordered panel with a 24px uppercase header, matching .panel in the
    React app. Content goes in the `with` block.

        with panel("Price Chart", key="chart", meta="AAPL · 1D"):
            st.plotly_chart(fig, ...)
    """
    box = st.container(key=f"panel_{key}")
    with box:
        st.markdown(
            f'<div class="wl-phead"><span><span class="ico">{icon}</span>'
            f'{html.escape(title)}</span>'
            f'<span class="meta">{html.escape(meta)}</span></div>',
            unsafe_allow_html=True,
        )
        yield
        if footer:
            st.markdown(
                f'<div class="wl-pfoot">{html.escape(footer)}</div>',
                unsafe_allow_html=True,
            )


def stat_tiles(items: list[dict]) -> None:
    """
    Dense quote tiles. Each item: {label, value, change_pct, (sub)}.
    Amber = up, red = down, per the terminal palette.
    """
    cells = []
    for it in items:
        chg = it.get("change_pct")
        rows = ""
        if chg is not None:
            cls = "up" if chg > 0 else "dn" if chg < 0 else "fl"
            arrow = "▲" if chg > 0 else "▼" if chg < 0 else "■"
            rows += f'<div class="chg {cls}">{arrow} {chg:+.2f}%</div>'
        # A tile with no percentage change shows only its caption - printing
        # "n/a" there reads as a failure rather than as "not applicable".
        if it.get("sub"):
            rows += f'<div class="chg fl">{html.escape(it["sub"])}</div>'
        if not rows:
            rows = '<div class="chg fl">&nbsp;</div>'
        cells.append(
            f'<div class="wl-tile"><div class="sym">{html.escape(it["label"])}</div>'
            f'<div class="px">{html.escape(it["value"])}</div>{rows}</div>'
        )
    st.markdown(f'<div class="wl-tilerow">{"".join(cells)}</div>', unsafe_allow_html=True)


def kv(rows: list[tuple[str, str] | tuple[str, str, str]]) -> None:
    """
    Dense label/value readout. Optional 3rd element is a value class:
    'up' | 'dn' | 'info' | 'dim'.
    """
    out = []
    for row in rows:
        k, v = row[0], row[1]
        cls = row[2] if len(row) > 2 else ""
        out.append(
            f'<tr><td class="k">{html.escape(k)}</td>'
            f'<td class="v {cls}">{html.escape(str(v))}</td></tr>'
        )
    st.markdown(f'<table class="wl-kv">{"".join(out)}</table>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="wl-note">{html.escape(text)}</div>', unsafe_allow_html=True)


def pending(label: str) -> None:
    st.markdown(f'<div class="wl-pending">{html.escape(label)}</div>', unsafe_allow_html=True)


def status_bar(left: list[tuple[str, str]], right: str = "") -> None:
    segs = " ".join(
        f"<span><b>{html.escape(k)}</b> <span class='mono'>{html.escape(v)}</span></span>"
        for k, v in left
    )
    st.markdown(
        f'<div class="wl-statusbar"><div class="seg">{segs}</div>'
        f'<div class="seg mono">{html.escape(right)}</div></div>',
        unsafe_allow_html=True,
    )
