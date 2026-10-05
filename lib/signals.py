"""
Signal evidence: current indicator state plus the HISTORICAL track record of
each signal on this specific name. PURE - no streamlit.

The point of this module is `conditional_edge`. An indicator reading on its
own ("RSI is 34") says nothing about what to do. What matters is what has
followed that reading on this security in the past, and whether the sample is
large enough to mean anything. A rule that is supposed to be bullish but has
historically been followed by flat returns is not a signal, and the dashboard
should say so rather than recite the textbook.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from . import stats as st_
from . import vol as vol_


def snapshot(df: pd.DataFrame) -> dict:
    """Everything the analysis pane needs about one name, right now."""
    c = df["close"]
    r = np.log(c).diff().replace([np.inf, -np.inf], np.nan).dropna()
    last = float(c.iloc[-1])

    sma20, sma50, sma200 = (float(ind.sma(c, w).iloc[-1]) for w in (20, 50, 200))
    bb = ind.bollinger(c, 20)
    width = float(bb["upper"].iloc[-1] - bb["lower"].iloc[-1])
    bb_pos = float((last - bb["lower"].iloc[-1]) / width) if width > 0 else float("nan")
    macd = ind.macd(c)
    rv21 = float(vol_.realised_vol(r, 21).iloc[-1])
    rv252 = float(st_.ann_vol(r))

    return {
        "last": last,
        "rsi": float(ind.rsi(c, 14).iloc[-1]),
        "macd_hist": float(macd["hist"].iloc[-1]),
        "macd_rising": bool(macd["hist"].iloc[-1] > macd["hist"].iloc[-2]),
        "bb_pos": bb_pos,
        "z60": float(ind.zscore(c, 60).iloc[-1]),
        "sma20": sma20, "sma50": sma50, "sma200": sma200,
        "vs_sma50": last / sma50 - 1.0,
        "vs_sma200": last / sma200 - 1.0,
        "stacked": bool(sma20 > sma50 > sma200),
        "from_high": float(last / c.cummax().iloc[-1] - 1.0),
        "atr_pct": float(ind.atr(df, 14).iloc[-1] / last),
        "vol21": rv21, "vol_full": rv252,
        "vol_ratio": rv21 / rv252 if rv252 else float("nan"),
        "ann_return": st_.ann_return(r), "sharpe": st_.sharpe(r),
        "max_dd": st_.max_drawdown(c)["depth"],
        "var95": st_.var_historical(r), "cvar95": st_.cvar_historical(r),
        "skew": st_.moments(r)["skew"],
        "excess_kurtosis": st_.moments(r)["excess_kurtosis"],
        "n": int(len(c)),
    }


def conditional_edge(close: pd.Series, condition: pd.Series, horizon: int = 21,
                     label: str = "") -> dict:
    """
    Forward return distribution following every past occurrence of `condition`.

    `condition` is a boolean Series aligned to `close`. Returns the mean and
    median forward return over `horizon` bars, the hit rate, and the same
    statistics for all other days so the two can be compared - an edge is the
    DIFFERENCE between conditional and unconditional, not the conditional
    number on its own.

    Overlapping windows mean these observations are not independent, so the
    effective sample is far smaller than `n` suggests. Treated as descriptive
    evidence, never as a significance test.
    """
    c = close.astype(float)
    fwd = c.shift(-horizon) / c - 1.0
    cond = condition.reindex(c.index).fillna(False).astype(bool)

    hit = fwd[cond].dropna()
    miss = fwd[~cond].dropna()
    if hit.empty:
        return {"ok": False, "label": label, "n": 0,
                "reason": "this condition has never occurred in the window"}

    base = float(miss.mean()) if len(miss) else float("nan")
    return {
        "ok": True, "label": label, "horizon": horizon,
        "n": int(hit.size),
        "mean": float(hit.mean()), "median": float(hit.median()),
        "hit_rate": float((hit > 0).mean()),
        "base_mean": base,
        "base_hit": float((miss > 0).mean()) if len(miss) else float("nan"),
        "edge": float(hit.mean() - base) if np.isfinite(base) else float("nan"),
        "worst": float(hit.min()), "best": float(hit.max()),
        "occurrences_per_year": float(hit.size / (len(c) / 252.0)),
    }


def standard_conditions(df: pd.DataFrame) -> dict[str, pd.Series]:
    """The conditions the analysis pane tracks, as boolean Series."""
    c = df["close"]
    rsi = ind.rsi(c, 14)
    z = ind.zscore(c, 60)
    macd = ind.macd(c)
    bb = ind.bollinger(c, 20)
    width = (bb["upper"] - bb["lower"]).replace(0, np.nan)
    bb_pos = (c - bb["lower"]) / width
    above200 = c > ind.sma(c, 200)

    return {
        "RSI < 35 (oversold)": rsi < 35,
        "RSI > 70 (overbought)": rsi > 70,
        "z60 < -1.5 (stretched down)": z < -1.5,
        "z60 > +1.5 (stretched up)": z > 1.5,
        "Below lower Bollinger": bb_pos < 0,
        "Above upper Bollinger": bb_pos > 1,
        "MACD histogram > 0": macd["hist"] > 0,
        "Oversold AND above 200d": (rsi < 40) & above200,
        "Oversold AND below 200d": (rsi < 40) & (~above200),
    }


def edge_table(df: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """Every standard condition with its historical forward-return record."""
    c = df["close"]
    rows = []
    for label, cond in standard_conditions(df).items():
        e = conditional_edge(c, cond, horizon, label)
        if not e.get("ok"):
            rows.append({"condition": label, "n": 0, "mean_fwd": np.nan,
                         "hit_rate": np.nan, "base_mean": np.nan, "edge": np.nan,
                         "active": False})
            continue
        rows.append({
            "condition": label, "n": e["n"], "mean_fwd": e["mean"],
            "hit_rate": e["hit_rate"], "base_mean": e["base_mean"],
            "edge": e["edge"],
            "active": bool(cond.iloc[-1]) if len(cond) else False,
        })
    return pd.DataFrame(rows)


def active_conditions(df: pd.DataFrame) -> list[str]:
    """Which of the standard conditions are true as of the last bar."""
    return [k for k, v in standard_conditions(df).items()
            if len(v) and bool(v.iloc[-1])]


def risk_budget(snap: dict, account_risk_pct: float = 1.0,
                atr_stop_mult: float = 2.0) -> dict:
    """
    Position size implied by volatility rather than conviction.

    A name with 57% annualised volatility and a 2xATR stop needs a far
    smaller position than one at 20% to put the same money at risk. This is
    the concrete reason a "sell" can mean "hold less" rather than "exit".
    """
    stop_pct = atr_stop_mult * snap["atr_pct"]
    if stop_pct <= 0:
        return {"ok": False}
    weight = (account_risk_pct / 100.0) / stop_pct
    return {
        "ok": True,
        "stop_pct": stop_pct,
        "weight": weight,
        "weight_pct": weight * 100.0,
        "risk_pct": account_risk_pct,
        "atr_mult": atr_stop_mult,
    }
