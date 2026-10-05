"""Page renderers. Each module exposes render(cfg) -> None."""
from . import (analysis, chart, correlation, forecast, overview, regime,
               returns, structure, volatility)

RENDERERS = {
    "overview": overview.render,
    "chart": chart.render,
    "returns": returns.render,
    "structure": structure.render,
    "vol": volatility.render,
    "regime": regime.render,
    "corr": correlation.render,
    "forecast": forecast.render,
    "analysis": analysis.render,
}
