"""
Volatility modelling: realised, EWMA, and GARCH. PURE - no streamlit.

Convention: returns go in as LOG returns in decimal form. `arch` fits far
more reliably on percentage returns, so the GARCH wrapper rescales by 100
internally and converts the output back to decimal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def realised_vol(returns: pd.Series, window: int = 21,
                 periods: int = TRADING_DAYS) -> pd.Series:
    return returns.rolling(window).std(ddof=1) * np.sqrt(periods)


def ewma_vol(returns: pd.Series, lam: float = 0.94,
             periods: int = TRADING_DAYS) -> pd.Series:
    """RiskMetrics EWMA. lambda=0.94 is the standard daily decay."""
    var = returns.pow(2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(var * periods)


def parkinson_vol(df: pd.DataFrame, window: int = 21,
                  periods: int = TRADING_DAYS) -> pd.Series:
    """
    Parkinson high-low range estimator. Uses intraday range, so it is far
    more efficient than close-to-close when the data allows it.
    """
    hl = np.log(df["high"] / df["low"]) ** 2
    var = hl.rolling(window).mean() / (4.0 * np.log(2.0))
    return np.sqrt(var * periods)


def fit_garch(returns: pd.Series, p: int = 1, q: int = 1,
              dist: str = "t", mean: str = "constant",
              periods: int = TRADING_DAYS) -> dict:
    """
    Fit GARCH(p,q). Returns conditional vol (annualised, decimal), the
    parameter table, persistence, and the implied long-run volatility.

    Persistence alpha+beta near 1 means shocks decay slowly - the formal
    version of "volatility clusters". At >= 1 the process is not stationary
    and the long-run variance is undefined.
    """
    from arch import arch_model

    r = returns.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if r.size < 100:
        return {"ok": False, "error": "need at least 100 observations"}

    scaled = r * 100.0
    try:
        res = arch_model(scaled, vol="GARCH", p=p, q=q, dist=dist, mean=mean).fit(
            disp="off", show_warning=False
        )
    except Exception as exc:  # noqa: BLE001 - surface the message to the UI
        return {"ok": False, "error": str(exc)}

    params = res.params
    alphas = [float(params.get(f"alpha[{i}]", 0.0)) for i in range(1, p + 1)]
    betas = [float(params.get(f"beta[{i}]", 0.0)) for i in range(1, q + 1)]
    omega = float(params.get("omega", np.nan))
    persistence = float(sum(alphas) + sum(betas))

    cond = pd.Series(res.conditional_volatility, index=r.index) / 100.0
    cond_ann = cond * np.sqrt(periods)

    if persistence < 1.0 and np.isfinite(omega):
        lr_var_pct = omega / (1.0 - persistence)          # in percent^2
        long_run = float(np.sqrt(lr_var_pct) / 100.0 * np.sqrt(periods))
    else:
        long_run = float("nan")

    half_life = (float(np.log(0.5) / np.log(persistence))
                 if 0 < persistence < 1 else float("inf"))

    return {
        "ok": True, "model": f"GARCH({p},{q})-{dist}",
        "params": params, "omega": omega,
        "alpha": alphas, "beta": betas,
        "persistence": persistence, "half_life": half_life,
        "long_run_vol": long_run,
        "cond_vol": cond_ann,
        "aic": float(res.aic), "bic": float(res.bic),
        "loglik": float(res.loglikelihood),
        "nu": float(params["nu"]) if "nu" in params else None,
        "res": res,
        "stationary": persistence < 1.0,
    }


def garch_forecast(fit: dict, horizon: int = 21,
                   periods: int = TRADING_DAYS) -> pd.Series:
    """Annualised conditional volatility forecast, `horizon` steps ahead."""
    if not fit.get("ok"):
        return pd.Series(dtype=float)
    f = fit["res"].forecast(horizon=horizon, reindex=False)
    var_pct = np.asarray(f.variance.iloc[-1])      # percent^2, per step
    vol = np.sqrt(var_pct) / 100.0 * np.sqrt(periods)
    return pd.Series(vol, index=np.arange(1, horizon + 1), name="forecast_vol")


def clustering_evidence(returns: pd.Series, lags: int = 20) -> dict:
    """
    The standard two-part check: returns themselves show little
    autocorrelation, but SQUARED returns show a lot. That gap is the
    signature of volatility clustering and the reason GARCH exists.
    """
    from .tsa import ljung_box

    lb_r = ljung_box(returns, lags=lags)
    lb_r2 = ljung_box(returns.pow(2), lags=lags)
    return {
        "returns": lb_r, "squared": lb_r2,
        "clustering": bool(lb_r2.get("autocorrelated") and not np.isnan(lb_r2.get("p", np.nan))),
        "summary": (
            "volatility clusters: squared returns are autocorrelated"
            if lb_r2.get("autocorrelated")
            else "no clustering detected in squared returns"
        ),
    }
