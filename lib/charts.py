"""
Plotly chart builders.

Conventions (from the dataviz method):
  * one y-axis per chart, never dual-axis
  * amber = up / red = down is a STATUS encoding, reserved; multi-series
    charts draw from theme.SERIES in fixed order, never cycled
  * 2px lines, >=8px markers, recessive grid, selective direct labels
  * hover layer always on
  * a single series needs no legend (the panel header names it)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import theme as T


def candles(df: pd.DataFrame, overlays: dict[str, pd.Series] | None = None,
            height: int = 300, uirevision: str = "keep") -> go.Figure:
    """Candlestick with optional overlay lines (e.g. moving averages)."""
    fig = go.Figure(
        go.Candlestick(
            x=df.index, open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            increasing=dict(line=dict(color=T.POSITIVE, width=1), fillcolor=T.POSITIVE),
            decreasing=dict(line=dict(color=T.NEGATIVE, width=1), fillcolor=T.NEGATIVE),
            name="price", showlegend=False,
            hoverlabel=dict(bgcolor=T.BG_TERTIARY),
        )
    )
    for i, (label, s) in enumerate((overlays or {}).items()):
        fig.add_trace(
            go.Scatter(
                x=s.index, y=s.values, mode="lines", name=label,
                line=dict(color=T.SERIES[i % len(T.SERIES)], width=2),
            )
        )
    fig.update_layout(**T.plotly_layout(
        height=height,
        showlegend=bool(overlays),
        uirevision=uirevision,
        xaxis=dict(rangeslider=dict(visible=False)),
    ))
    return fig


def price_volume(df: pd.DataFrame, overlays: dict[str, pd.Series] | None = None,
                 height: int = 380, uirevision: str = "keep",
                 volume_share: float = 0.22) -> go.Figure:
    """
    Candles and volume as ONE figure on a shared x-axis.

    Two stacked figures look the same but behave wrongly: zooming the price
    chart leaves volume at full extent, and the two drift out of alignment.
    A single figure with shared_xaxes keeps them locked together and gives one
    crosshair across both.
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.015,
        row_heights=[1.0 - volume_share, volume_share],
    )
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            increasing=dict(line=dict(color=T.POSITIVE, width=1), fillcolor=T.POSITIVE),
            decreasing=dict(line=dict(color=T.NEGATIVE, width=1), fillcolor=T.NEGATIVE),
            name="price", showlegend=False,
        ), row=1, col=1)
    for i, (label, s) in enumerate((overlays or {}).items()):
        fig.add_trace(
            go.Scatter(x=s.index, y=s.values, mode="lines", name=label,
                       line=dict(color=T.SERIES[i % len(T.SERIES)], width=2)),
            row=1, col=1)

    up = df["close"] >= df["open"]
    fig.add_trace(
        go.Bar(x=df.index, y=df["volume"],
               marker=dict(color=np.where(up, "rgba(255,179,0,0.45)",
                                          "rgba(229,57,53,0.45)"),
                           line=dict(width=0)),
               name="volume", showlegend=False,
               hovertemplate="%{y:,.0f}<extra></extra>"),
        row=2, col=1)

    base = T.plotly_layout(height=height, showlegend=bool(overlays),
                           uirevision=uirevision)
    axis = dict(gridcolor=T.BORDER_SUBTLE, zerolinecolor=T.BORDER,
                linecolor=T.BORDER, showline=True, ticks="outside",
                tickcolor=T.BORDER, tickfont=dict(size=9))
    # make_subplots(shared_xaxes=True) makes the TOP axis follow the bottom
    # one (xaxis.matches='x2'). That is backwards here: the user interacts
    # with the price panel, and a follower axis rejects range changes. Invert
    # it so the price axis drives and volume follows.
    # Range buttons: a deterministic way to change the window that does not
    # depend on dragging accurately, and cannot be triggered by accident.
    rangeselector = dict(
        buttons=[
            dict(count=1, label="1M", step="month", stepmode="backward"),
            dict(count=3, label="3M", step="month", stepmode="backward"),
            dict(count=6, label="6M", step="month", stepmode="backward"),
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(count=3, label="3Y", step="year", stepmode="backward"),
            dict(step="all", label="ALL"),
        ],
        bgcolor=T.BG_TERTIARY, activecolor=T.RED,
        bordercolor=T.BORDER, borderwidth=1,
        font=dict(family=T.FONT_SANS, size=9, color=T.TEXT_TERTIARY),
        x=0, y=1.14, xanchor="left", yanchor="top",
    )
    base.update(
        margin=dict(l=44, r=10, t=34, b=24),
        xaxis=dict(**axis, rangeslider=dict(visible=False),
                   showticklabels=False, matches=None,
                   rangeselector=rangeselector),
        xaxis2=dict(**axis, matches="x"),
        yaxis=dict(**axis, side="left", fixedrange=False),
        yaxis2=dict(**axis, side="left", showgrid=False, tickformat=".2s",
                    fixedrange=True),
    )
    fig.update_layout(**base)
    return fig


def volume(df: pd.DataFrame, height: int = 90) -> go.Figure:
    """Volume bars tinted by the day's direction, deliberately recessive."""
    up = df["close"] >= df["open"]
    colors = np.where(up, "rgba(255,179,0,0.45)", "rgba(229,57,53,0.45)")
    fig = go.Figure(
        go.Bar(x=df.index, y=df["volume"], marker=dict(color=colors, line=dict(width=0)),
               name="volume", showlegend=False, hovertemplate="%{y:,.0f}<extra></extra>")
    )
    # x tick labels are suppressed: the price chart directly above already
    # labels this time axis, and repeating them reads as a second chart.
    fig.update_layout(**T.plotly_layout(
        height=height, margin=dict(l=44, r=10, t=2, b=6),
        xaxis=dict(showticklabels=False, ticks=""),
        yaxis=dict(showgrid=False, tickformat=".2s"),
    ))
    return fig


def lines(frame: pd.DataFrame, height: int = 260, yfmt: str | None = None,
          hline: float | None = None) -> go.Figure:
    """Multi-series line chart. Colors assigned in fixed SERIES order."""
    fig = go.Figure()
    for i, col in enumerate(frame.columns):
        fig.add_trace(
            go.Scatter(
                x=frame.index, y=frame[col], mode="lines", name=str(col),
                line=dict(color=T.SERIES[i % len(T.SERIES)], width=2),
            )
        )
    if hline is not None:
        fig.add_hline(y=hline, line=dict(color=T.BORDER_ACCENT, width=1, dash="dot"))
    layout = T.plotly_layout(height=height, showlegend=len(frame.columns) > 1)
    if yfmt:
        layout["yaxis"]["tickformat"] = yfmt
    fig.update_layout(**layout)
    return fig


def area_single(s: pd.Series, color: str | None = None, height: int = 260,
                hline: float | None = None) -> go.Figure:
    """One series as a filled line - for equity curves, rolling stats."""
    c = color or T.SERIES[0]
    rgba = _rgba(c, 0.18)
    fig = go.Figure(
        go.Scatter(x=s.index, y=s.values, mode="lines", name=s.name or "value",
                   line=dict(color=c, width=2), fill="tozeroy", fillcolor=rgba,
                   showlegend=False)
    )
    if hline is not None:
        fig.add_hline(y=hline, line=dict(color=T.BORDER_ACCENT, width=1, dash="dot"))
    fig.update_layout(**T.plotly_layout(height=height))
    return fig


def histogram(values: np.ndarray, bins: int = 60, height: int = 240,
              overlay_normal: bool = True) -> go.Figure:
    """Return distribution with an optional fitted-normal reference curve."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    fig = go.Figure(
        go.Histogram(x=v, nbinsx=bins, name="observed",
                     marker=dict(color=_rgba(T.SERIES[1], 0.65), line=dict(width=0)),
                     histnorm="probability density", showlegend=overlay_normal)
    )
    if overlay_normal and v.size > 2:
        mu, sd = float(v.mean()), float(v.std(ddof=1))
        xs = np.linspace(v.min(), v.max(), 240)
        pdf = np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        fig.add_trace(go.Scatter(x=xs, y=pdf, mode="lines", name="normal",
                                 line=dict(color=T.POSITIVE, width=2, dash="dash")))
    fig.update_layout(**T.plotly_layout(height=height, showlegend=overlay_normal,
                                        hovermode="closest"))
    return fig


def heatmap(matrix: pd.DataFrame, height: int = 300, zmid: float | None = 0.0,
            zmin: float | None = -1.0, zmax: float | None = 1.0,
            diverging: bool = True) -> go.Figure:
    """
    Correlation-style heatmap. Diverging scale (two hues + neutral grey
    midpoint) because correlation has polarity around zero.
    """
    scale = T.DIVERGING if diverging else T.SEQ_TEAL
    colorscale = [[i / (len(scale) - 1), c] for i, c in enumerate(scale)]
    fig = go.Figure(
        go.Heatmap(
            z=matrix.values, x=[str(c) for c in matrix.columns],
            y=[str(i) for i in matrix.index],
            colorscale=colorscale, zmid=zmid, zmin=zmin, zmax=zmax,
            xgap=2, ygap=2,  # 2px surface gap between cells
            colorbar=dict(thickness=8, len=0.85, outlinewidth=0,
                          tickfont=dict(size=9, color=T.TEXT_TERTIARY)),
            hovertemplate="%{y} / %{x}<br>%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(**T.plotly_layout(
        height=height, margin=dict(l=60, r=10, t=8, b=40), hovermode="closest",
        xaxis=dict(showgrid=False, side="bottom"), yaxis=dict(showgrid=False, autorange="reversed"),
    ))
    return fig


def bars(labels: list[str], values: list[float], height: int = 240,
         color: str | None = None, signed: bool = False,
         yfmt: str | None = None, hline: float | None = None,
         yrange: tuple[float, float] | None = None,
         threshold: float | None = None) -> go.Figure:
    """
    Bar chart. `signed=True` paints by polarity (amber up / red down).
    `threshold` paints each bar by which side of that value it falls on --
    used for Hurst, where the only question is above or below 0.5.
    """
    if signed:
        cols = [T.POSITIVE if v >= 0 else T.NEGATIVE for v in values]
    elif threshold is not None:
        cols = [T.POSITIVE if v > threshold else T.SERIES[1] for v in values]
    else:
        cols = [color or T.SERIES[0]] * len(values)
    fig = go.Figure(
        go.Bar(x=labels, y=values, marker=dict(color=cols, line=dict(width=0)),
               width=0.45, showlegend=False,
               text=[f"{v:.4f}" for v in values], textposition="outside",
               textfont=dict(family=T.FONT_MONO, size=10, color=T.TEXT_SECONDARY),
               hovertemplate="%{x}<br>%{y:.4f}<extra></extra>")
    )
    if hline is not None:
        fig.add_hline(y=hline, line=dict(color=T.TEXT_TERTIARY, width=1, dash="dash"),
                      annotation_text=f"{hline:g}",
                      annotation_position="right",
                      annotation_font=dict(size=9, color=T.TEXT_TERTIARY))
    layout = T.plotly_layout(height=height, hovermode="closest")
    if yfmt:
        layout["yaxis"]["tickformat"] = yfmt
    if yrange:
        layout["yaxis"]["range"] = list(yrange)
    fig.update_layout(**layout)
    return fig


def stems(lags: np.ndarray, values: np.ndarray, conf: float | None = None,
          height: int = 220) -> go.Figure:
    """ACF/PACF style stem plot with an optional significance band."""
    fig = go.Figure()
    for lag, val in zip(lags, values):
        fig.add_trace(go.Scatter(x=[lag, lag], y=[0, val], mode="lines",
                                 line=dict(color=T.BORDER_ACCENT, width=1),
                                 showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=lags, y=values, mode="markers", name="coef",
        marker=dict(color=T.SERIES[1], size=8), showlegend=False,
        hovertemplate="lag %{x}<br>%{y:.3f}<extra></extra>"))
    if conf:
        for sign in (1, -1):
            fig.add_hline(y=sign * conf, line=dict(color=T.RED, width=1, dash="dot"))
    fig.add_hline(y=0, line=dict(color=T.BORDER, width=1))
    fig.update_layout(**T.plotly_layout(height=height, hovermode="closest"))
    return fig


def qq(values: np.ndarray, height: int = 240) -> go.Figure:
    """Normal Q-Q plot with a 45-degree reference line."""
    from scipy import stats

    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    (osm, osr), (slope, inter, _) = stats.probplot(v, dist="norm")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="sample",
                             marker=dict(color=T.SERIES[1], size=5, opacity=0.7),
                             showlegend=False,
                             hovertemplate="theo %{x:.2f}<br>obs %{y:.4f}<extra></extra>"))
    line_x = np.array([osm.min(), osm.max()])
    fig.add_trace(go.Scatter(x=line_x, y=slope * line_x + inter, mode="lines",
                             name="normal", line=dict(color=T.POSITIVE, width=2, dash="dash"),
                             showlegend=False, hoverinfo="skip"))
    fig.update_layout(**T.plotly_layout(height=height, hovermode="closest"))
    return fig


def band(center: pd.Series, lower: pd.Series, upper: pd.Series,
         color: str | None = None, height: int = 280,
         history: pd.Series | None = None) -> go.Figure:
    """Forecast fan: a central path inside a shaded interval."""
    c = color or T.SERIES[1]
    fig = go.Figure()
    if history is not None and len(history):
        fig.add_trace(go.Scatter(x=history.index, y=history.values, mode="lines",
                                 name="actual", line=dict(color=T.TEXT_TERTIARY, width=2)))
    fig.add_trace(go.Scatter(x=list(upper.index) + list(lower.index[::-1]),
                             y=list(upper.values) + list(lower.values[::-1]),
                             fill="toself", fillcolor=_rgba(c, 0.15),
                             line=dict(width=0), name="interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=center.index, y=center.values, mode="lines",
                             name="forecast", line=dict(color=c, width=2)))
    fig.update_layout(**T.plotly_layout(height=height, showlegend=True))
    return fig


def osc(series: pd.Series, bands: tuple[float, float] | None = (30.0, 70.0),
        height: int = 120, yrange: tuple[float, float] | None = (0, 100),
        extra: pd.Series | None = None) -> go.Figure:
    """Bounded oscillator (RSI, stochastic) with threshold bands."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines",
                             name=series.name or "osc",
                             line=dict(color=T.SERIES[1], width=2), showlegend=False))
    if extra is not None:
        fig.add_trace(go.Scatter(x=extra.index, y=extra.values, mode="lines",
                                 name=extra.name or "signal",
                                 line=dict(color=T.SERIES[0], width=1.5), showlegend=False))
    if bands:
        lo, hi = bands
        fig.add_hrect(y0=hi, y1=yrange[1] if yrange else hi,
                      fillcolor=_rgba(T.NEGATIVE, 0.08), line_width=0)
        fig.add_hrect(y0=yrange[0] if yrange else lo, y1=lo,
                      fillcolor=_rgba(T.POSITIVE, 0.08), line_width=0)
        for y in (lo, hi):
            fig.add_hline(y=y, line=dict(color=T.BORDER_ACCENT, width=1, dash="dot"))
    layout = T.plotly_layout(height=height, margin=dict(l=44, r=10, t=4, b=18))
    if yrange:
        layout["yaxis"]["range"] = list(yrange)
    fig.update_layout(**layout)
    return fig


def macd_panel(m: pd.DataFrame, height: int = 130) -> go.Figure:
    """MACD line + signal + histogram, histogram tinted by sign."""
    colors = np.where(m["hist"] >= 0, _rgba(T.POSITIVE, 0.55), _rgba(T.NEGATIVE, 0.55))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=m.index, y=m["hist"], marker=dict(color=colors, line=dict(width=0)),
                         name="hist", showlegend=False, hovertemplate="%{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=m.index, y=m["macd"], mode="lines", name="MACD",
                             line=dict(color=T.SERIES[1], width=2)))
    fig.add_trace(go.Scatter(x=m.index, y=m["signal"], mode="lines", name="signal",
                             line=dict(color=T.SERIES[0], width=1.5)))
    fig.add_hline(y=0, line=dict(color=T.BORDER, width=1))
    fig.update_layout(**T.plotly_layout(height=height, showlegend=True,
                                        margin=dict(l=44, r=10, t=4, b=18)))
    return fig


def scatter_fit(x: pd.Series, y: pd.Series, height: int = 260,
                xlabel: str = "", ylabel: str = "") -> go.Figure:
    """Scatter with an OLS line - used for hedge-ratio regressions."""
    xv, yv = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[m], yv[m]
    fig = go.Figure(go.Scatter(x=xv, y=yv, mode="markers", showlegend=False,
                               marker=dict(color=T.SERIES[1], size=4, opacity=0.5),
                               hovertemplate="%{x:.4f}, %{y:.4f}<extra></extra>"))
    if xv.size > 2:
        b, a = np.polyfit(xv, yv, 1)
        xs = np.array([xv.min(), xv.max()])
        fig.add_trace(go.Scatter(x=xs, y=a + b * xs, mode="lines", showlegend=False,
                                 line=dict(color=T.POSITIVE, width=2), hoverinfo="skip"))
    layout = T.plotly_layout(height=height, hovermode="closest")
    layout["xaxis"]["title"] = dict(text=xlabel, font=dict(size=9))
    layout["yaxis"]["title"] = dict(text=ylabel, font=dict(size=9))
    fig.update_layout(**layout)
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"
