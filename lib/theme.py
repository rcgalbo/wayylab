"""
Design tokens + Plotly template, mirrored from
frontend/src/styles/wayy-terminal.css so the dashboard and the React app
read as one product.

SERIES is a validated categorical palette (not the brand status colours).
Checked with the dataviz validator against surface #111111:
  lightness band PASS | chroma floor PASS | CVD separation PASS (worst
  adjacent dE 12.2 deutan) | normal-vision PASS | contrast PASS.
Order is fixed and must never be cycled or re-sorted by series rank.
"""
from __future__ import annotations

from pathlib import Path

# --- surfaces -------------------------------------------------------------
BG_PRIMARY = "#0a0a0a"
BG_SECONDARY = "#111111"
BG_TERTIARY = "#1a1a1a"
BG_HOVER = "#222222"

# --- ink ------------------------------------------------------------------
TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#e0e0e0"
TEXT_TERTIARY = "#a0a0a0"
TEXT_MUTED = "#666666"
TEXT_DIM = "#444444"

# --- lines ----------------------------------------------------------------
BORDER = "#333333"
BORDER_SUBTLE = "#222222"
BORDER_ACCENT = "#444444"

# --- brand / status (reserved: never used as a series colour) -------------
RED = "#E53935"
POSITIVE = "#FFB300"   # amber - classic terminal "up"
NEGATIVE = "#E53935"
NEUTRAL = "#9E9E9E"
INFO = "#64B5F6"
HIGHLIGHT = "#FFD54F"

# --- categorical series palette (validated, fixed order) ------------------
SERIES = [
    "#BE8700",  # amber
    "#2474CF",  # blue
    "#C0453D",  # red
    "#009FAC",  # teal
    "#8459C3",  # purple
    "#2B9F4A",  # green
]

# --- sequential ramp for heatmaps (single hue, light->dark reversed for dark mode)
SEQ_TEAL = ["#0a2a2d", "#0d454b", "#0f6069", "#0d7c88", "#009FAC", "#4FBCC5", "#9BD9DE"]
# diverging for correlation: two poles + neutral grey midpoint
DIVERGING = ["#2474CF", "#5A8FD6", "#9BB4DE", "#4a4a4a", "#D99A94", "#CC6B63", "#C0453D"]

FONT_MONO = "JetBrains Mono, SF Mono, Monaco, Consolas, monospace"
FONT_SANS = "Space Grotesk, -apple-system, BlinkMacSystemFont, sans-serif"


def load_css() -> str:
    """Return the stylesheet wrapped in a <style> tag, ready for st.markdown."""
    css = (Path(__file__).parent / "styles.css").read_text(encoding="utf-8")
    return f"<style>{css}</style>"


def plotly_layout(height: int = 260, showlegend: bool = False,
                  uirevision: str = "keep", **kw) -> dict:
    """
    Base layout for every chart. Recessive grid, mono tick labels, no title
    (the panel header names the chart), tight margins for dense packing.

    `uirevision` is what makes a zoom or pan survive a Streamlit rerun. Plotly
    resets the axes whenever the figure is re-sent unless this value is
    unchanged, so every widget interaction would otherwise snap the chart back
    to full extent. Pass a value derived from the dataset (symbol + interval)
    so the view persists while the data is the same and resets when it is not.
    """
    layout = dict(
        height=height,
        uirevision=uirevision,
        margin=dict(l=44, r=10, t=8, b=26),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_MONO, size=10, color=TEXT_TERTIARY),
        showlegend=showlegend,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(size=9, color=TEXT_TERTIARY),
        ),
        xaxis=dict(
            gridcolor=BORDER_SUBTLE, zerolinecolor=BORDER,
            linecolor=BORDER, showline=True, ticks="outside",
            tickcolor=BORDER, tickfont=dict(size=9),
        ),
        yaxis=dict(
            gridcolor=BORDER_SUBTLE, zerolinecolor=BORDER,
            linecolor=BORDER, showline=True, ticks="outside",
            tickcolor=BORDER, tickfont=dict(size=9), side="left",
        ),
        hoverlabel=dict(
            bgcolor=BG_TERTIARY, bordercolor=BORDER,
            font=dict(family=FONT_MONO, size=10, color=TEXT_PRIMARY),
        ),
        hovermode="x unified",
        # Box-zoom on drag, which is what a price chart implies. With "pan"
        # a drag throws the view outside the data entirely and there is no
        # way back without a reset control.
        dragmode="zoom",
    )
    for k, v in kw.items():
        if k in ("xaxis", "yaxis", "legend", "margin") and isinstance(v, dict):
            layout[k] = {**layout[k], **v}
        else:
            layout[k] = v
    return layout


_MODEBAR_HIDE = [
    "select2d", "lasso2d", "zoomIn2d", "zoomOut2d", "toggleSpikelines",
]

# Default: modebar appears on hover. Omitting `displayModeBar` entirely is
# what gives hover behaviour - pinning it on every panel covers the data in
# the small charts, which are read at a glance and never zoomed.
PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": _MODEBAR_HIDE,
    # scrollZoom is OFF deliberately. The charts are tall, so the cursor sits
    # over one most of the time; with it on, every page scroll zooms the chart
    # instead of scrolling the page, and scrolling back zooms it out again.
    # That reads as the chart glitching and resetting at random. Zooming is
    # drag-a-box, the range buttons, or the modebar - all deliberate acts.
    "scrollZoom": False,
    "doubleClick": "reset",
    "showTips": False,
    "toImageButtonOptions": {"format": "png", "scale": 2},
}

# Pinned: for the main price charts, where zoom/pan/reset is the whole point
# and the controls need to be visible without hunting for them.
PLOTLY_CONFIG_PINNED = {**PLOTLY_CONFIG, "displayModeBar": True}
