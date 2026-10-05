"""
Forecasting with honest evaluation. PURE - no streamlit.

The point of this module is the BASELINE. On daily equity prices a random
walk is a very strong forecaster, and any model that cannot beat it is not
adding information. Everything here is therefore reported relative to naive
benchmarks, out of sample, via walk-forward - never in-sample fit.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    a = np.asarray(actual, float)
    p = np.asarray(pred, float)
    m = np.isfinite(a) & np.isfinite(p)
    a, p = a[m], p[m]
    if a.size == 0:
        return {k: float("nan") for k in ("rmse", "mae", "mape", "n")}
    err = a - p
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(np.nanmean(np.abs(err / np.where(a == 0, np.nan, a))) * 100)
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "mape": mape,
        "n": int(a.size),
    }


def directional_accuracy(actual_now: np.ndarray, actual_prev: np.ndarray,
                         pred: np.ndarray) -> float:
    """
    Share of steps where the predicted direction matched the realised one.

    Steps where the forecast calls no change (p_dir == 0) are excluded, not
    scored as misses. The naive random-walk forecast predicts exactly the
    previous value every time, so scoring those as wrong would report 0%
    accuracy for a model that simply declines to make a directional call.
    Returns NaN when the forecast never takes a side.
    """
    a_dir = np.sign(np.asarray(actual_now, float) - np.asarray(actual_prev, float))
    p_dir = np.sign(np.asarray(pred, float) - np.asarray(actual_prev, float))
    m = np.isfinite(a_dir) & np.isfinite(p_dir) & (a_dir != 0) & (p_dir != 0)
    return float((a_dir[m] == p_dir[m]).mean()) if m.sum() else float("nan")


# ---------------------------------------------------------------------------
# walk-forward
# ---------------------------------------------------------------------------
def walk_forward(prices: pd.Series, order: tuple[int, int, int] = (1, 1, 1),
                 train: int = 500, step: int = 1, horizon: int = 1,
                 max_points: int = 250) -> dict:
    """
    Rolling-origin one-step-ahead evaluation of ARIMA against two baselines.

    At each origin the model is refitted on the trailing `train` window and
    asked for `horizon` steps. Nothing after the origin is ever visible to
    the fit, which is what makes this out of sample.

    Baselines:
      naive      - tomorrow equals today (the random walk)
      drift      - today plus the average per-step drift of the window
    """
    from statsmodels.tsa.arima.model import ARIMA

    px = prices.astype(float).dropna()
    n = px.size
    if n < train + 30:
        return {"ok": False, "error": f"need > {train + 30} observations, have {n}"}

    origins = list(range(train, n - horizon, step))
    if len(origins) > max_points:                     # keep refits bounded
        origins = origins[-max_points:]

    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for o in origins:
            hist = px.iloc[o - train:o]
            actual = float(px.iloc[o + horizon - 1])
            last = float(hist.iloc[-1])
            try:
                fit = ARIMA(hist.to_numpy(), order=order).fit()
                yhat = float(np.asarray(fit.forecast(steps=horizon))[-1])
            except Exception:
                yhat = np.nan
            drift = last + (last - float(hist.iloc[0])) / max(1, train - 1) * horizon
            rows.append({"when": px.index[o + horizon - 1], "actual": actual,
                         "arima": yhat, "naive": last, "drift": drift,
                         "prev": last})

    df = pd.DataFrame(rows).set_index("when")
    out = {"ok": True, "frame": df, "order": order, "train": train,
           "horizon": horizon, "n_origins": len(df)}
    for name in ("arima", "naive", "drift"):
        m = metrics(df["actual"].to_numpy(), df[name].to_numpy())
        m["dir_acc"] = directional_accuracy(
            df["actual"].to_numpy(), df["prev"].to_numpy(), df[name].to_numpy())
        out[name] = m

    naive_rmse = out["naive"]["rmse"]
    arima_rmse = out["arima"]["rmse"]
    out["theil_u"] = float(arima_rmse / naive_rmse) if naive_rmse else float("nan")
    out["beats_naive"] = bool(np.isfinite(out["theil_u"]) and out["theil_u"] < 1.0)
    out["verdict"] = (
        f"ARIMA{order} beats the random walk (Theil U = {out['theil_u']:.3f})"
        if out["beats_naive"] else
        f"ARIMA{order} does NOT beat the random walk (Theil U = {out['theil_u']:.3f})"
    )
    out["dm"] = diebold_mariano(
        df["actual"].to_numpy(), df["arima"].to_numpy(), df["naive"].to_numpy())
    return out


def diebold_mariano(actual: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                    h: int = 1) -> dict:
    """
    Diebold-Mariano test of equal predictive accuracy (squared-error loss).

    Without this, "model A had lower RMSE" is just a sample statement. DM
    asks whether the difference is larger than its own sampling noise.
    Negative statistic favours p1; a |stat| under ~1.96 means the two
    forecasts are statistically indistinguishable.
    """
    from scipy import stats

    a, x1, x2 = (np.asarray(v, float) for v in (actual, p1, p2))
    m = np.isfinite(a) & np.isfinite(x1) & np.isfinite(x2)
    a, x1, x2 = a[m], x1[m], x2[m]
    if a.size < 20:
        return {"ok": False}
    d = (a - x1) ** 2 - (a - x2) ** 2
    n = d.size
    dbar = float(d.mean())
    gamma0 = float(np.var(d, ddof=0))
    acc = sum((1 - k / h) * float(np.cov(d[k:], d[:-k], ddof=0)[0, 1])
              for k in range(1, h)) if h > 1 else 0.0
    var = (gamma0 + 2 * acc) / n
    if var <= 0:
        return {"ok": False}
    stat = dbar / np.sqrt(var)
    p = float(2 * (1 - stats.norm.cdf(abs(stat))))
    return {"ok": True, "stat": float(stat), "p": p,
            "favours": "model" if stat < 0 else "baseline",
            "significant": bool(p < 0.05)}


def fit_arima(prices: pd.Series, order: tuple[int, int, int] = (1, 1, 1),
              horizon: int = 21, alpha: float = 0.05) -> dict:
    """In-sample fit plus a forward forecast with a confidence band."""
    from statsmodels.tsa.arima.model import ARIMA

    px = prices.astype(float).dropna()
    if px.size < 100:
        return {"ok": False, "error": "need at least 100 observations"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = ARIMA(px.to_numpy(), order=order).fit()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        fc = res.get_forecast(steps=horizon)
        mean = np.asarray(fc.predicted_mean)
        ci = np.asarray(fc.conf_int(alpha=alpha))

    freq = pd.infer_freq(px.index) or "B"
    try:
        idx = pd.date_range(px.index[-1], periods=horizon + 1, freq=freq)[1:]
    except Exception:
        idx = pd.RangeIndex(len(px), len(px) + horizon)

    return {
        "ok": True, "order": order,
        "forecast": pd.Series(mean, index=idx, name="forecast"),
        "lower": pd.Series(ci[:, 0], index=idx, name="lower"),
        "upper": pd.Series(ci[:, 1], index=idx, name="upper"),
        "aic": float(res.aic), "bic": float(res.bic),
        "resid": pd.Series(np.asarray(res.resid), index=px.index),
        "res": res,
    }


def auto_order(prices: pd.Series, max_p: int = 3, max_q: int = 3,
               d: int = 1, ic: str = "aic") -> dict:
    """
    Small grid search over (p,d,q) by information criterion.

    Deliberately a grid and not pmdarima: one less dependency, and the full
    table is worth showing - it makes clear how flat the criterion usually
    is across orders, which is itself a result.
    """
    from statsmodels.tsa.arima.model import ARIMA

    px = prices.astype(float).dropna()
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    r = ARIMA(px.to_numpy(), order=(p, d, q)).fit()
                    rows.append({"p": p, "d": d, "q": q,
                                 "aic": float(r.aic), "bic": float(r.bic)})
                except Exception:
                    continue
    if not rows:
        return {"ok": False, "error": "no order converged"}
    tbl = pd.DataFrame(rows).sort_values(ic).reset_index(drop=True)
    best = tbl.iloc[0]
    return {"ok": True, "table": tbl,
            "best": (int(best["p"]), int(best["d"]), int(best["q"])),
            "ic": ic}
