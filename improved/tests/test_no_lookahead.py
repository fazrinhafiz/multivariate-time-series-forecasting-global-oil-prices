"""Guard against look-ahead bias, the most common flaw in forecasting theses.

The idea: corrupt every observation *after* the last forecast origin. If any
model's forecast changes, it was reading data it could not have had.
"""
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import load_prices  # noqa: E402
from models import (make_features, run_ar, run_lstm, run_random_walk, run_var,  # noqa: E402
                    run_vecm, run_xgboost)

H = 3
ORIGINS = list(pd.date_range("2015-10-01", "2016-09-01", freq="MS"))  # spans a yearly re-train


@pytest.fixture(scope="module")
def prices():
    return load_prices()


@pytest.fixture(scope="module")
def poisoned(prices):
    p = prices.copy()
    future = p.index > ORIGINS[-1]
    rng = np.random.default_rng(0)
    p.loc[future] = p.loc[future] * rng.uniform(0.2, 5.0, size=p.loc[future].shape)
    return p


@pytest.mark.parametrize("model", [
    run_random_walk, run_ar, run_var, run_vecm, run_xgboost,
    partial(run_lstm, seeds=(0,), max_epochs=3),
], ids=["random_walk", "ar", "var", "vecm", "xgboost", "lstm"])
def test_forecasts_ignore_future_data(model, prices, poisoned):
    clean = model(prices, ORIGINS, H)
    dirty = model(poisoned, ORIGINS, H)
    pd.testing.assert_frame_equal(clean, dirty, rtol=1e-6)


def test_features_at_t_use_only_data_up_to_t(prices, poisoned):
    cut = ORIGINS[-1]
    pd.testing.assert_frame_equal(make_features(prices).loc[:cut], make_features(poisoned).loc[:cut])


def test_forecast_index_is_origin_plus_horizon(prices):
    fc = run_random_walk(prices, ORIGINS, H)
    assert fc.index[0] == ORIGINS[0] + pd.DateOffset(months=H)


def test_balanced_sample_has_no_imputed_values(prices):
    assert prices.index[0] == pd.Timestamp("1984-01-01")  # first month South African coal exists
    assert not prices.isna().any().any()
