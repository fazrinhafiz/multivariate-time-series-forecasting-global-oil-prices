"""Guard against look-ahead bias, the most common flaw in forecasting studies.

The idea: corrupt every observation *after* the last forecast origin. If any
forecast changes, the pipeline read data it could not have had.

Coverage: every forecasting pipeline (random walk, AR, VAR, VECM, XGBoost, LSTM at
two window lengths) x horizons 1, 3, 6 x three blocks of 12 origins (one spanning
the first yearly re-train, one around COVID, one late in the test window), plus the
hyperparameter-tuning step.

What this proves: forecasts at the tested origins do not depend on the *values* of
any later observation. It cannot detect leakage that works through something other
than data values (for example the number of rows), and it covers these origins and
horizons, not every possible one. The LSTM runs with one seed and 3 epochs here.
"""
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import load_prices  # noqa: E402
from models import (_to_prices, make_features, run_ar, run_lstm,  # noqa: E402
                    run_random_walk, run_var, run_vecm, run_xgboost)
from tuning import VAL_END, tune  # noqa: E402

HORIZONS = [1, 3, 6]
BLOCKS = {
    "2015-16": ("2015-07-01", "2016-06-01"),   # spans the first yearly re-train
    "2019-20": ("2019-10-01", "2020-09-01"),   # COVID collapse
    "2023-24": ("2023-10-01", "2024-09-01"),   # late test window
}
PIPELINES = {
    "random_walk": run_random_walk,
    "ar": run_ar,
    "var": run_var,
    "vecm": run_vecm,
    "xgboost": run_xgboost,
    "xgboost_alt_params": partial(run_xgboost, params={"max_depth": 2, "learning_rate": 0.1}),
    "lstm_window6": partial(run_lstm, window=6, units=16, seeds=(0,), max_epochs=3),
    "lstm_window12": partial(run_lstm, window=12, units=32, seeds=(0,), max_epochs=3),
}


@pytest.fixture(scope="module")
def prices():
    return load_prices()


def poison_after(prices, cutoff, seed=0):
    p = prices.copy()
    future = p.index > pd.Timestamp(cutoff)
    rng = np.random.default_rng(seed)
    p.loc[future] = p.loc[future] * rng.uniform(0.2, 5.0, size=p.loc[future].shape)
    return p


@pytest.fixture(scope="module")
def poisoned(prices):
    return {name: poison_after(prices, end) for name, (_, end) in BLOCKS.items()}


def origins(block):
    return list(pd.date_range(*BLOCKS[block], freq="MS"))


@pytest.mark.parametrize("block", list(BLOCKS))
@pytest.mark.parametrize("h", HORIZONS)
@pytest.mark.parametrize("pipeline", list(PIPELINES))
def test_forecasts_ignore_future_data(pipeline, h, block, prices, poisoned):
    fn = PIPELINES[pipeline]
    clean = fn(prices, origins(block), h)
    dirty = fn(poisoned[block], origins(block), h)
    pd.testing.assert_frame_equal(clean, dirty, rtol=1e-6)


@pytest.mark.parametrize("h", HORIZONS)
def test_a_leaky_model_is_caught(h, prices, poisoned):
    """Sanity check on the test itself: a model that peeks at the target month must fail it."""
    def leaky(p, orig, h):
        tgt = [o + pd.DateOffset(months=h) for o in orig]
        return _to_prices(p, orig, h, np.log(p.loc[tgt, ["brent", "wti"]].values / p.loc[orig, ["brent", "wti"]].values))
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(leaky(prices, origins("2019-20"), h),
                                      leaky(poisoned["2019-20"], origins("2019-20"), h))


def test_features_at_t_use_only_data_up_to_t(prices, poisoned):
    cut = BLOCKS["2019-20"][1]
    pd.testing.assert_frame_equal(make_features(prices).loc[:cut], make_features(poisoned["2019-20"]).loc[:cut])


def test_tuning_ignores_everything_after_the_validation_window(prices):
    kwargs = dict(horizons=[1, 3], xgb_grid=[{"max_depth": 2}, {"max_depth": 3}],
                  lstm_grid=[{"window": 6, "units": 16}], lstm_seeds=(0,), lstm_max_epochs=3)
    clean_table, clean_choice = tune(prices, **kwargs)
    dirty_table, dirty_choice = tune(poison_after(prices, VAL_END), **kwargs)
    pd.testing.assert_series_equal(clean_table["val rel. RMSE"], dirty_table["val rel. RMSE"])
    assert clean_choice == dirty_choice


@pytest.mark.parametrize("h", HORIZONS)
def test_forecast_index_is_origin_plus_horizon(h, prices):
    fc = run_random_walk(prices, origins("2015-16"), h)
    assert fc.index[0] == pd.Timestamp(BLOCKS["2015-16"][0]) + pd.DateOffset(months=h)


def test_balanced_sample_has_no_imputed_values(prices):
    assert prices.index[0] == pd.Timestamp("1984-01-01")  # first month South African coal exists
    assert not prices.isna().any().any()
