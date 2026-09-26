"""Choose XGBoost and LSTM hyperparameters on a validation period that ends before the test.

Validation design (no test data involved):
- The price panel is truncated at VAL_END (2015-12) before anything else happens.
- Validation targets are the months VAL_START (2010-01) to VAL_END, forecast with the
  same rolling-origin protocol as the test: origins = target - h, models re-trained
  every January on data up to the origin.
- Each candidate is scored by RMSE relative to the random walk, averaged over Brent
  and WTI, separately for each horizon. The lowest wins; ties go to the earlier
  (simpler or previous-default) candidate.

The grids are deliberately small: the aim is to show the ML models were given a fair,
pre-registered chance, not to search exhaustively.
"""
from functools import partial

import numpy as np
import pandas as pd

from data import TARGETS
from models import LSTM_DEFAULT, XGB_DEFAULT, run_lstm, run_random_walk, run_xgboost

VAL_START, VAL_END = "2010-01", "2015-12"

XGB_GRID = [dict(XGB_DEFAULT)] + [
    {**XGB_DEFAULT, "max_depth": d, "learning_rate": lr}
    for d in (2, 3, 4) for lr in (0.03, 0.1) if not (d == 3 and lr == 0.03)
]
LSTM_GRID = [dict(LSTM_DEFAULT)] + [
    {"window": w, "units": u} for w in (6, 12) for u in (16, 32) if (w, u) != (12, 32)
]
TUNING_SEEDS = (0, 1)


def validation_origins(prices, h):
    targets = prices.loc[VAL_START:VAL_END].index
    return list(targets - pd.DateOffset(months=h))


def _rel_rmse(actual, fc, rw):
    return float(np.mean([np.sqrt(((actual[t] - fc[t]) ** 2).mean()) / np.sqrt(((actual[t] - rw[t]) ** 2).mean())
                          for t in TARGETS]))


def tune(prices, horizons, xgb_grid=None, lstm_grid=None, lstm_seeds=TUNING_SEEDS, lstm_max_epochs=300):
    """Return (table of validation scores, {model: {h: chosen params}})."""
    prices = prices.loc[:VAL_END]  # hard stop: nothing after the validation period is visible
    xgb_grid = XGB_GRID if xgb_grid is None else xgb_grid
    lstm_grid = LSTM_GRID if lstm_grid is None else lstm_grid
    actual = prices.loc[VAL_START:VAL_END, TARGETS]
    rows, chosen = [], {"XGBoost": {}, "LSTM": {}}
    for h in horizons:
        origins = validation_origins(prices, h)
        rw = run_random_walk(prices, origins, h)
        candidates = [("XGBoost", i, p, partial(run_xgboost, params=p)) for i, p in enumerate(xgb_grid)] + \
                     [("LSTM", i, p, partial(run_lstm, seeds=lstm_seeds, max_epochs=lstm_max_epochs, **p))
                      for i, p in enumerate(lstm_grid)]
        for model, i, params, fn in candidates:
            fc = fn(prices, origins, h)
            rows.append({"horizon": h, "model": model, "candidate": i, "params": params,
                         "val rel. RMSE": _rel_rmse(actual, fc, rw)})
        table = pd.DataFrame(rows)
        for model in chosen:
            t = table[(table.horizon == h) & (table.model == model)]
            chosen[model][h] = t.loc[t["val rel. RMSE"].idxmin(), "params"]
    table = pd.DataFrame(rows)
    table["selected"] = [chosen[r.model][r.horizon] == r.params for r in table.itertuples()]
    return table, chosen
