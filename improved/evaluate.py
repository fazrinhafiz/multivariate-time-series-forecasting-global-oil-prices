"""Accuracy metrics, Diebold-Mariano tests and sub-period breakdowns."""
import numpy as np
import pandas as pd
from scipy import stats

# Periods an examiner will ask about (RQ2: volatility and structural shocks).
PERIODS = {
    "2016-19 (calm)": ("2016-01", "2019-12"),
    "2020-21 (COVID)": ("2020-01", "2021-12"),
    "2022-25 (energy crisis)": ("2022-01", "2025-12"),
    "2026 (spring spike)": ("2026-01", "2026-12"),
}


def metrics(actual: pd.Series, fc: pd.Series, rw: pd.Series) -> dict:
    """Standard errors plus two numbers that matter more for oil:

    - rel. RMSE: RMSE divided by the random walk's RMSE (<1 means the model beats no-change)
    - direction: share of months where the forecast gets the sign of the change right
      (ties with the random walk's zero change count as wrong)
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


def diebold_mariano(actual, fc_a, fc_b, h):
    """DM test of equal squared-error accuracy with the Harvey-Leybourne-Newbold correction.

    Returns (statistic, two-sided p-value). Negative statistic = model A more accurate.
    """
    d = (actual - fc_a) ** 2 - (actual - fc_b) ** 2
    d = d.to_numpy()
    n, mean = len(d), d.mean()
    gamma = [np.sum((d[k:] - mean) * (d[: n - k] - mean)) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return np.nan, np.nan
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = hln * mean / np.sqrt(var)
    return stat, 2 * stats.t.sf(abs(stat), df=n - 1)


def score(actual: pd.DataFrame, forecasts: dict, h: int) -> pd.DataFrame:
    """One row per (target, model) with metrics and a DM test against the random walk."""
    rw = forecasts["Random walk"]
    rows = []
    for tgt in actual:
        for name, fc in forecasts.items():
            row = {"target": tgt, "model": name, **metrics(actual[tgt], fc[tgt], rw[tgt])}
            if name != "Random walk":
                row["DM stat vs RW"], row["DM p-value"] = diebold_mariano(actual[tgt], fc[tgt], rw[tgt], h)
            rows.append(row)
    return pd.DataFrame(rows)


def score_by_period(actual: pd.DataFrame, forecasts: dict) -> pd.DataFrame:
    rows = []
    for period, (a, b) in PERIODS.items():
        if actual.loc[a:b].empty:  # period lies beyond the data
            continue
        rw = forecasts["Random walk"].loc[a:b]
        for tgt in actual:
            for name, fc in forecasts.items():
                m = metrics(actual.loc[a:b, tgt], fc.loc[a:b, tgt], rw[tgt])
                rows.append({"period": period, "target": tgt, "model": name,
                             "RMSE": m["RMSE"], "rel. RMSE vs RW": m["rel. RMSE vs RW"]})
    return pd.DataFrame(rows)
