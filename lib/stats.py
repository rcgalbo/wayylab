"""
Return and risk statistics. PURE - no streamlit, no I/O.

Everything takes a pandas Series/ndarray of *log returns* unless the name
says otherwise, and every annualised figure uses `periods` (252 for daily).
These are the functions the /api/quant endpoints will re-export.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def ann_vol(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = _clean(returns)
    return float(r.std(ddof=1) * np.sqrt(periods)) if r.size > 1 else float("nan")


def rolling_vol(returns: pd.Series, window: int = 21,
                periods: int = TRADING_DAYS) -> pd.Series:
    return returns.rolling(window).std(ddof=1) * np.sqrt(periods)


def ewma_vol(returns: pd.Series, lam: float = 0.94,
             periods: int = TRADING_DAYS) -> pd.Series:
    """RiskMetrics-style EWMA volatility (lambda=0.94 for daily)."""
    var = returns.pow(2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(var * periods)


def ann_return(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = _clean(returns)
    return float(r.mean() * periods) if r.size else float("nan")


def sharpe(returns: pd.Series, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    """Annualised Sharpe. `rf` is an annual rate."""
    r = _clean(returns)
    if r.size < 2:
        return float("nan")
    excess = r - rf / periods
    sd = excess.std(ddof=1)
    return float(excess.mean() / sd * np.sqrt(periods)) if sd > 0 else float("nan")


def sortino(returns: pd.Series, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    """
    Annualised Sortino: excess return over *downside* deviation only.
    Downside deviation is computed against the full sample size, not just
    the count of negative observations - the convention wrtrade uses.
    """
    r = _clean(returns)
    if r.size < 2:
        return float("nan")
    excess = r - rf / periods
    downside = excess.where(excess < 0, 0.0)
    dd = np.sqrt((downside ** 2).mean())
    if dd <= 0:
        return float("inf")
    return float(excess.mean() / dd * np.sqrt(periods))


def max_drawdown(close: pd.Series) -> dict:
    """Peak-to-trough on a *price* series. Returns depth plus the dates."""
    px = close.astype(float).dropna()
    if px.empty:
        return {"depth": float("nan"), "peak": None, "trough": None, "recovered": None}
    peak = px.cummax()
    dd = px / peak - 1.0
    trough_i = dd.idxmin()
    peak_i = px.loc[:trough_i].idxmax()
    after = px.loc[trough_i:]
    rec = after[after >= px.loc[peak_i]]
    return {
        "depth": float(dd.min()),
        "peak": peak_i,
        "trough": trough_i,
        "recovered": (rec.index[0] if len(rec) else None),
        "curve": dd,
    }


def var_historical(returns: pd.Series, alpha: float = 0.05) -> float:
    """Historical VaR as a positive loss fraction at the given tail."""
    r = _clean(returns)
    return float(-np.quantile(r, alpha)) if r.size else float("nan")


def cvar_historical(returns: pd.Series, alpha: float = 0.05) -> float:
    """Expected shortfall: mean loss conditional on breaching VaR."""
    r = _clean(returns)
    if not r.size:
        return float("nan")
    cut = np.quantile(r, alpha)
    tail = r[r <= cut]
    return float(-tail.mean()) if tail.size else float("nan")


def var_backtest(returns: pd.Series, alpha: float = 0.05, window: int = 250) -> dict:
    """
    Rolling out-of-sample VaR exceedance test. A correctly specified model
    breaches on about `alpha` of days; this reports the realised rate plus
    a Kupiec POF likelihood-ratio statistic.
    """
    r = _clean(returns)
    if r.size <= window + 10:
        return {"expected": alpha, "realised": float("nan"), "breaches": 0,
                "n": 0, "kupiec_lr": float("nan")}
    thresh = r.rolling(window).quantile(alpha).shift(1)
    tested = pd.concat([r, thresh], axis=1).dropna()
    tested.columns = ["ret", "var"]
    breach = tested["ret"] <= tested["var"]
    n, x = int(len(tested)), int(breach.sum())
    pi = x / n if n else float("nan")
    lr = float("nan")
    if n and 0 < x < n:
        lr = -2 * (
            (n - x) * np.log(1 - alpha) + x * np.log(alpha)
            - ((n - x) * np.log(1 - pi) + x * np.log(pi))
        )
    return {"expected": alpha, "realised": pi, "breaches": x, "n": n,
            "kupiec_lr": lr, "series": tested, "breach_mask": breach}


def moments(returns: pd.Series) -> dict:
    """Mean/sd/skew/excess-kurtosis plus a Jarque-Bera normality test."""
    from scipy import stats

    r = _clean(returns)
    if r.size < 8:
        return {k: float("nan") for k in
                ("mean", "sd", "skew", "excess_kurtosis", "jb", "jb_p", "n")}
    jb, jb_p = stats.jarque_bera(r)
    return {
        "mean": float(r.mean()), "sd": float(r.std(ddof=1)),
        "skew": float(stats.skew(r)), "excess_kurtosis": float(stats.kurtosis(r)),
        "jb": float(jb), "jb_p": float(jb_p), "n": int(r.size),
    }


def correlation(panel: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Correlation of log returns across a wide panel of closes."""
    rets = np.log(panel.astype(float)).diff().replace([np.inf, -np.inf], np.nan).dropna()
    return rets.corr(method=method)


def _clean(x) -> pd.Series:
    s = pd.Series(x).astype(float)
    return s.replace([np.inf, -np.inf], np.nan).dropna()
