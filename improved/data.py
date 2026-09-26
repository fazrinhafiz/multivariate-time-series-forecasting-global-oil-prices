"""Load the World Bank Pink Sheet and build a clean, gap-free monthly panel.

Differences from the original notebook:
- The sample starts at the first month where *every* series is observed
  (1984-01), so no backward-filling invents values that were never recorded.
- Prices are modelled as log-returns, which are stationary (see diagnostics),
  and converted back to $/bbl only for scoring.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_XLSX = ROOT / "CMO-Historical-Data-Monthly.xlsx"

# (Pink Sheet header, unit) -> short name used everywhere else
SERIES = {
    ("Crude oil, Brent", "($/bbl)"): "brent",
    ("Crude oil, WTI", "($/bbl)"): "wti",
    ("Coal, Australian", "($/mt)"): "coal_au",
    ("Coal, South African **", "($/mt)"): "coal_za",
    ("Natural gas, US", "($/mmbtu)"): "gas_us",
    ("Natural gas, Europe", "($/mmbtu)"): "gas_eu",
    ("Liquefied natural gas, Japan", "($/mmbtu)"): "lng_jp",
    ("Palm oil", "($/mt)"): "palm_oil",
    ("Copper", "($/mt)"): "copper",
    ("Gold", "($/troy oz)"): "gold",
}
TARGETS = ["brent", "wti"]
LABELS = {"brent": "Brent", "wti": "WTI"}


def load_raw(path=DEFAULT_XLSX) -> pd.DataFrame:
    """All selected series from the 'Monthly Prices' sheet, NaN where unobserved."""
    raw = pd.read_excel(path, sheet_name="Monthly Prices", skiprows=4, header=[0, 1], engine="openpyxl")
    dates = pd.to_datetime(raw.iloc[:, 0], format="%YM%m")
    df = raw[list(SERIES)].apply(pd.to_numeric, errors="coerce")
    df.columns = list(SERIES.values())
    df.index = pd.DatetimeIndex(dates, name="date", freq="MS")
    return df


def load_prices(path=DEFAULT_XLSX) -> pd.DataFrame:
    """Balanced panel: from the first month all series exist to the last month all exist."""
    df = load_raw(path)
    start = max(df[c].first_valid_index() for c in df)
    end = min(df[c].last_valid_index() for c in df)
    df = df.loc[start:end]
    if df.isna().any().any():
        raise ValueError(f"Gaps inside the balanced sample:\n{df.isna().sum()[lambda s: s > 0]}")
    return df.asfreq("MS")


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices).diff().dropna()
