"""
Time-series structure: stationarity tests, autocorrelation, decomposition.
PURE - no streamlit, no I/O.

Every test returns a dict with the statistic, the p-value, the critical
values where the test provides them, and a plain-language verdict, so the
UI never has to re-interpret a p-value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def adf(series: pd.Series, regression: str = "c", maxlag: int | None = None) -> dict:
    """
    Augmented Dickey-Fuller. H0 = a unit root is present (non-stationary).
    A small p-value is evidence FOR stationarity.
    """
    from statsmodels.tsa.stattools import adfuller

    x = _clean(series)
    if x.size < 20:
        return _insufficient("ADF")
    stat, p, used_lag, nobs, crit, _ = adfuller(x, maxlag=maxlag, regression=regression)
    return {
        "test": "ADF", "stat": float(stat), "p": float(p),
        "lags": int(used_lag), "nobs": int(nobs),
        "crit": {k: float(v) for k, v in crit.items()},
        "null": "unit root (non-stationary)",
        "stationary": bool(p < 0.05),
        "verdict": ("stationary - unit root rejected" if p < 0.05
                    else "non-stationary - cannot reject unit root"),
    }


def kpss(series: pd.Series, regression: str = "c", nlags: str = "auto") -> dict:
    """
    KPSS. H0 = the series IS stationary - the mirror image of ADF.
    A small p-value is evidence AGAINST stationarity.
    """
    from statsmodels.tsa.stattools import kpss as _kpss

    x = _clean(series)
    if x.size < 20:
        return _insufficient("KPSS")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # p-value is clipped at the table edges
        stat, p, lags, crit = _kpss(x, regression=regression, nlags=nlags)
    return {
        "test": "KPSS", "stat": float(stat), "p": float(p), "lags": int(lags),
        "crit": {k: float(v) for k, v in crit.items()},
        "null": "stationary",
        "stationary": bool(p >= 0.05),
        "verdict": ("stationary - cannot reject" if p >= 0.05
                    else "non-stationary - stationarity rejected"),
    }


def joint_stationarity(series: pd.Series) -> dict:
    """
    ADF and KPSS together. They test opposite nulls, so agreement is
    informative and disagreement is itself a finding (often fractional
    integration or a structural break).
    """
    a, k = adf(series), kpss(series)
    if a["stationary"] and k["stationary"]:
        concl, detail = "stationary", "both tests agree"
    elif not a["stationary"] and not k["stationary"]:
        concl, detail = "non-stationary", "both tests agree"
    elif a["stationary"] and not k["stationary"]:
        concl, detail = "inconclusive", "ADF says stationary, KPSS disagrees - possible trend-stationarity"
    else:
        concl, detail = "inconclusive", "KPSS says stationary, ADF disagrees - possible fractional integration"
    return {"adf": a, "kpss": k, "conclusion": concl, "detail": detail}


def acf_pacf(series: pd.Series, nlags: int = 40) -> dict:
    """ACF and PACF with the +-1.96/sqrt(n) significance band."""
    from statsmodels.tsa.stattools import acf as _acf, pacf as _pacf

    x = _clean(series)
    nlags = int(min(nlags, max(1, x.size // 2 - 1)))
    a = _acf(x, nlags=nlags, fft=True)
    try:
        p = _pacf(x, nlags=nlags)
    except Exception:
        p = np.full(nlags + 1, np.nan)
    return {
        "lags": np.arange(1, nlags + 1),
        "acf": np.asarray(a[1:]), "pacf": np.asarray(p[1:]),
        "conf": float(1.96 / np.sqrt(x.size)), "n": int(x.size),
    }


def ljung_box(series: pd.Series, lags: int = 20) -> dict:
    """
    Ljung-Box Q. H0 = no autocorrelation up to `lags`.
    Run it on squared returns to test for volatility clustering.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    x = _clean(series)
    if x.size < lags + 5:
        return _insufficient("Ljung-Box")
    res = acorr_ljungbox(x, lags=[lags], return_df=True)
    stat = float(res["lb_stat"].iloc[0])
    p = float(res["lb_pvalue"].iloc[0])
    return {
        "test": "Ljung-Box", "stat": stat, "p": p, "lags": lags,
        "null": "no autocorrelation",
        "autocorrelated": bool(p < 0.05),
        "verdict": ("autocorrelation present" if p < 0.05
                    else "no significant autocorrelation"),
    }


def difference(series: pd.Series, order: int = 1) -> pd.Series:
    out = series.astype(float).copy()
    for _ in range(max(0, order)):
        out = out.diff()
    return out.dropna()


def stl_decompose(series: pd.Series, period: int = 21, robust: bool = True) -> pd.DataFrame:
    """
    STL decomposition into trend / seasonal / residual.
    `period` is in observations (21 ~ one trading month on daily data).
    """
    from statsmodels.tsa.seasonal import STL

    x = series.astype(float).dropna()
    period = int(max(2, min(period, max(2, len(x) // 3))))
    res = STL(x, period=period, robust=robust).fit()
    return pd.DataFrame(
        {"observed": x, "trend": res.trend, "seasonal": res.seasonal, "resid": res.resid}
    )


def _clean(x) -> np.ndarray:
    s = pd.Series(x).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    return s.to_numpy()


def _insufficient(name: str) -> dict:
    return {"test": name, "stat": float("nan"), "p": float("nan"),
            "verdict": "insufficient data", "stationary": False,
            "autocorrelated": False, "crit": {}, "null": "", "lags": 0}
