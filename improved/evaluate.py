"""Accuracy metrics, forecast-comparison tests and sub-period breakdowns.

Two tests compare each model with the random walk (no-change) forecast:

- Diebold-Mariano (DM), two-sided, Harvey-Leybourne-Newbold corrected, on squared
  $/bbl errors. It asks whether the two forecasts are equally accurate.
- Clark-West (CW), one-sided, on h-month log changes. The random walk forecasts a
  log change of zero and is nested in every other model (all of them reduce to it
  when their coefficients are zero). Under that nesting DM is undersized, so CW is
  the appropriate test of "does the model beat no-change?" (Clark & West, 2006,
  2007). H0: the log price is a martingale (no-change is optimal); H1: the model's
  forecast has predictive content.

Neither test is corrected for multiple comparisons here; `add_holm` does that for
a whole family of tests (for example all models x targets x horizons of one run).
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Descriptive windows used to discuss RQ2 (behaviour under shocks). Chosen after the
# data were seen, so they are illustrations, not tests.
PERIODS = {
    "2016-19 (calm)": ("2016-01", "2019-12"),
    "2020-21 (COVID)": ("2020-01", "2021-12"),
    "2022-25 (energy crisis and after)": ("2022-01", "2025-12"),
    "2026 (spring spike)": ("2026-01", "2026-12"),
}


def metrics(actual: pd.Series, fc: pd.Series, rw: pd.Series) -> dict:
    """RMSE, MAE, MAPE, RMSE relative to the random walk, and directional accuracy.

    Direction: share of months where the forecast calls the sign of the change from
    the origin price correctly (a zero forecast change counts as wrong).
    """
    e = actual - fc
    rmse = np.sqrt(np.mean(e**2))
    moved, called = np.sign(actual - rw), np.sign(fc - rw)
    return {
        "RMSE": rmse,
        "MAE": np.mean(np.abs(e)),
        "MAPE %": np.mean(np.abs(e / actual)) * 100,
        "rel. RMSE vs RW": rmse / np.sqrt(np.mean((actual - rw) ** 2)),
        "direction %": np.mean(moved == called) * 100,
    }


def _lrv(x: np.ndarray, lags: int, bartlett: bool) -> float:
    """Long-run variance of the mean of x with `lags` autocovariances."""
    n, xc = len(x), x - x.mean()
    gamma = [np.sum(xc[k:] * xc[: n - k]) / n for k in range(lags + 1)]
    w = [1 - k / (lags + 1) if bartlett else 1.0 for k in range(lags + 1)]
    return (gamma[0] + 2 * sum(wk * g for wk, g in zip(w[1:], gamma[1:]))) / n


def diebold_mariano(actual, fc_a, fc_b, h):
    """DM test of equal squared-error accuracy with the Harvey-Leybourne-Newbold correction.

    Returns (statistic, two-sided p-value). Negative statistic = forecast A more accurate.
    """
    d = ((actual - fc_a) ** 2 - (actual - fc_b) ** 2).to_numpy()
    n = len(d)
    var = _lrv(d, h - 1, bartlett=False)
    if var <= 0:
        return np.nan, np.nan
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = hln * d.mean() / np.sqrt(var)
    return stat, 2 * stats.t.sf(abs(stat), df=n - 1)


def clark_west(y, yhat, h):
    """Clark-West test of a model forecast `yhat` against a zero (no-change) forecast.

    y and yhat are realised and forecast h-month log changes. The adjusted loss
    differential is f = y^2 - [(y - yhat)^2 - yhat^2]; the statistic is its mean over
    a Newey-West (Bartlett, h-1 lags) standard error, compared with N(0,1).
    Returns (statistic, one-sided p-value). Positive statistic = model adds information.
    """
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    f = y**2 - ((y - yhat) ** 2 - yhat**2)
    var = _lrv(f, h - 1, bartlett=True)
    if var <= 0:
        return np.nan, np.nan
    stat = f.mean() / np.sqrt(var)
    return stat, stats.norm.sf(stat)


def score(actual: pd.DataFrame, forecasts: dict, h: int) -> pd.DataFrame:
    """One row per (target, model): metrics plus DM and CW tests against the random walk."""
    rw = forecasts["Random walk"]
    rows = []
    for tgt in actual:
        y = np.log(actual[tgt] / rw[tgt])
        for name, fc in forecasts.items():
            row = {"target": tgt, "model": name, **metrics(actual[tgt], fc[tgt], rw[tgt])}
            if name == "Random walk":
                row["direction %"] = np.nan  # undefined: it never forecasts a change
            else:
                row["DM stat"], row["DM p (two-sided)"] = diebold_mariano(actual[tgt], fc[tgt], rw[tgt], h)
                row["CW stat"], row["CW p (one-sided)"] = clark_west(y, np.log(fc[tgt] / rw[tgt]), h)
            rows.append(row)
    return pd.DataFrame(rows)


def add_holm(scores: pd.DataFrame) -> pd.DataFrame:
    """Holm-adjust the DM and CW p-values across every row of one family of tests."""
    out = scores.copy()
    for col in ["DM p (two-sided)", "CW p (one-sided)"]:
        mask = out[col].notna()
        out.loc[mask, col.replace(" (", " Holm (")] = multipletests(out.loc[mask, col], method="holm")[1]
    return out


def score_by_period(actual: pd.DataFrame, forecasts: dict) -> pd.DataFrame:
    """Relative RMSE within each descriptive window (no tests; small samples)."""
    rows = []
    for period, (a, b) in PERIODS.items():
        if actual.loc[a:b].empty:  # window lies outside this evaluation sample
            continue
        rw = forecasts["Random walk"].loc[a:b]
        for tgt in actual:
            for name, fc in forecasts.items():
                m = metrics(actual.loc[a:b, tgt], fc.loc[a:b, tgt], rw[tgt])
                rows.append({"period": period, "months": len(actual.loc[a:b]), "target": tgt, "model": name,
                             "RMSE": m["RMSE"], "rel. RMSE vs RW": m["rel. RMSE vs RW"]})
    return pd.DataFrame(rows)
