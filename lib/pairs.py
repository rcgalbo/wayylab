"""
Cross-sectional structure: correlation/PCA, cointegration, spreads,
pair scanning, Granger causality. PURE - no streamlit.

These back the SpreadMonitor / PairsScanner / CorrelationMatrix /
GrangerCausality widgets, which currently fetch /api/quant/* and 404.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# correlation + PCA
# ---------------------------------------------------------------------------
def returns_panel(closes: pd.DataFrame) -> pd.DataFrame:
    r = np.log(closes.astype(float)).diff()
    return r.replace([np.inf, -np.inf], np.nan).dropna(how="any")


def corr_matrix(closes: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    return returns_panel(closes).corr(method=method)


def pca(closes: pd.DataFrame) -> dict:
    """
    Eigen-decomposition of the correlation matrix. PC1's share is the
    standard read on how much of the panel is just "the market".
    """
    rets = returns_panel(closes)
    if rets.shape[1] < 2 or rets.shape[0] < 10:
        return {"ok": False, "error": "need >=2 series and >=10 observations"}
    c = rets.corr().to_numpy()
    vals, vecs = np.linalg.eigh(c)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    share = vals / vals.sum()
    return {
        "ok": True,
        "eigenvalues": vals,
        "explained": share,
        "cumulative": np.cumsum(share),
        "loadings": pd.DataFrame(
            vecs, index=rets.columns,
            columns=[f"PC{i+1}" for i in range(vecs.shape[1])]),
        "pc1_share": float(share[0]),
        "n_assets": int(rets.shape[1]),
    }


# ---------------------------------------------------------------------------
# cointegration
# ---------------------------------------------------------------------------
def engle_granger(y: pd.Series, x: pd.Series) -> dict:
    """
    Two-step Engle-Granger on LOG prices.

    Step 1 regresses log(y) on log(x) by OLS; the slope is the hedge ratio.
    Step 2 runs ADF on the residual. Rejecting the unit root means the
    spread is stationary, i.e. the pair is cointegrated.

    Note this is not symmetric - eg(y,x) and eg(x,y) can differ. The
    scanner below tests both directions and keeps the stronger.
    """
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller

    df = pd.concat([y, x], axis=1).dropna()
    if df.shape[0] < 60:
        return {"ok": False, "error": "need at least 60 overlapping observations"}
    df.columns = ["y", "x"]
    ly, lx = np.log(df["y"]), np.log(df["x"])

    model = sm.OLS(ly, sm.add_constant(lx)).fit()
    beta = float(model.params.iloc[1])
    alpha = float(model.params.iloc[0])
    spread = ly - (alpha + beta * lx)

    stat, p, _, _, crit, _ = adfuller(spread.to_numpy(), regression="c")
    hl = half_life(spread)
    return {
        "ok": True, "beta": beta, "alpha": alpha,
        "spread": spread, "adf_stat": float(stat), "p": float(p),
        "crit": {k: float(v) for k, v in crit.items()},
        "cointegrated": bool(p < 0.05),
        "half_life": hl,
        "r2": float(model.rsquared),
        "zscore": float(((spread - spread.mean()) / spread.std(ddof=0)).iloc[-1])
        if spread.std(ddof=0) > 0 else float("nan"),
        "verdict": ("cointegrated - spread is stationary" if p < 0.05
                    else "not cointegrated - spread wanders"),
    }


def half_life(spread: pd.Series) -> float:
    """
    Ornstein-Uhlenbeck half-life of mean reversion, from the AR(1)
    coefficient of the spread. Short = reverts quickly = tradeable.
    """
    import statsmodels.api as sm

    s = spread.dropna()
    lag = s.shift(1).dropna()
    delta = (s - s.shift(1)).dropna()
    n = min(len(lag), len(delta))
    if n < 20:
        return float("nan")
    res = sm.OLS(delta.iloc[-n:], sm.add_constant(lag.iloc[-n:])).fit()
    k = float(res.params.iloc[1])
    return float(-np.log(2) / k) if k < 0 else float("inf")


def spread_series(y: pd.Series, x: pd.Series, beta: float, alpha: float = 0.0,
                  window: int = 60) -> pd.DataFrame:
    """Spread plus its rolling z-score, the usual entry/exit signal."""
    df = pd.concat([y, x], axis=1).dropna()
    df.columns = ["y", "x"]
    sp = np.log(df["y"]) - (alpha + beta * np.log(df["x"]))
    mu = sp.rolling(window).mean()
    sd = sp.rolling(window).std(ddof=0).replace(0, np.nan)
    return pd.DataFrame({"spread": sp, "mean": mu, "z": (sp - mu) / sd})


def scan_pairs(closes: pd.DataFrame, max_pairs: int | None = None,
               min_obs: int = 120) -> pd.DataFrame:
    """
    Test every ordered pair for cointegration, keep the better direction,
    and rank by p-value.

    Multiple-testing caveat: with k assets there are k(k-1) tests, so at
    alpha=0.05 a handful of "significant" pairs are expected by chance
    alone. The returned frame carries a Bonferroni-adjusted threshold so
    the UI can show which survive it.
    """
    cols = list(closes.columns)
    rows = []
    for a, b in itertools.combinations(cols, 2):
        sub = closes[[a, b]].dropna()
        if sub.shape[0] < min_obs:
            continue
        best = None
        for yy, xx in ((a, b), (b, a)):
            r = engle_granger(closes[yy], closes[xx])
            if not r.get("ok"):
                continue
            if best is None or r["p"] < best["p"]:
                best = {**r, "y": yy, "x": xx}
        if best is None:
            continue
        rows.append({
            "pair": f"{best['y']}/{best['x']}",
            "y": best["y"], "x": best["x"],
            "beta": best["beta"], "p": best["p"],
            "adf": best["adf_stat"], "half_life": best["half_life"],
            "z": best["zscore"], "r2": best["r2"],
            "cointegrated": best["cointegrated"],
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    n_tests = max(1, len(out))
    out.attrs["bonferroni"] = 0.05 / n_tests
    out.attrs["n_tests"] = n_tests
    if max_pairs:
        out = out.head(max_pairs)
    return out


# ---------------------------------------------------------------------------
# Granger causality
# ---------------------------------------------------------------------------
def granger(cause: pd.Series, effect: pd.Series, maxlag: int = 10) -> dict:
    """
    Does `cause` help predict `effect` beyond `effect`'s own history?

    Run on RETURNS, not prices - the test assumes stationary inputs, and
    running it on levels is the most common way to get a spurious result.
    "Granger causality" is predictive precedence, not causation.
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    df = pd.concat([effect, cause], axis=1).dropna()
    if df.shape[0] < maxlag * 5 + 20:
        return {"ok": False, "error": "not enough overlapping observations"}
    df.columns = ["effect", "cause"]
    data = df[["effect", "cause"]].to_numpy()

    # grangercausalitytests prints its whole table to stdout and the
    # `verbose` kwarg was removed in statsmodels 0.14, so swallow it.
    import contextlib
    import io
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                res = grangercausalitytests(data, maxlag=maxlag)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    per_lag = []
    for lag in range(1, maxlag + 1):
        f_stat, p, _, _ = res[lag][0]["ssr_ftest"]
        per_lag.append({"lag": lag, "F": float(f_stat), "p": float(p)})
    tbl = pd.DataFrame(per_lag)
    best = tbl.loc[tbl["p"].idxmin()]
    return {
        "ok": True, "table": tbl,
        "best_lag": int(best["lag"]), "best_p": float(best["p"]),
        "best_F": float(best["F"]),
        "significant": bool(best["p"] < 0.05),
        "bonferroni": 0.05 / maxlag,
        "survives_bonferroni": bool(best["p"] < 0.05 / maxlag),
    }
