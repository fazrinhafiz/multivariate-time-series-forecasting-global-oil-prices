"""Known-answer checks for the forecast-comparison tests and the VECM specification."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import load_prices  # noqa: E402
from diagnostics import johansen_trace  # noqa: E402
from evaluate import add_holm, clark_west, diebold_mariano  # noqa: E402
from models import run_vecm  # noqa: E402
from statsmodels.tsa.vector_ar.vecm import coint_johansen  # noqa: E402

rng = np.random.default_rng(7)


def test_clark_west_is_undefined_for_a_zero_forecast():
    y = rng.normal(size=200)
    stat, p = clark_west(y, np.zeros_like(y), h=1)
    assert np.isnan(stat) and np.isnan(p)  # adjusted loss differential is exactly zero


def test_clark_west_detects_real_predictive_content():
    x = rng.normal(size=300)
    y = 0.5 * x + rng.normal(size=300)
    assert clark_west(y, 0.5 * x, h=1)[1] < 0.01


def test_clark_west_does_not_reward_noise_forecasts():
    y = rng.normal(size=300)
    noise = 0.3 * rng.normal(size=300)
    assert clark_west(y, noise, h=1)[1] > 0.05


def test_diebold_mariano_sign_convention():
    idx = pd.RangeIndex(200)
    actual = pd.Series(rng.normal(size=200), idx)
    good = actual + 0.1 * rng.normal(size=200)
    bad = actual + 1.0 * rng.normal(size=200)
    stat, p = diebold_mariano(actual, good, bad, h=1)
    assert stat < 0 and p < 0.01


def test_holm_adjustment_never_lowers_p_values():
    df = pd.DataFrame({"DM p (two-sided)": [0.01, 0.04, 0.3, np.nan], "CW p (one-sided)": [0.002, 0.2, 0.5, np.nan]})
    out = add_holm(df)
    for raw, adj in [("DM p (two-sided)", "DM p Holm (two-sided)"), ("CW p (one-sided)", "CW p Holm (one-sided)")]:
        m = out[raw].notna()
        assert (out.loc[m, adj] >= out.loc[m, raw]).all()


def test_own_johansen_matches_statsmodels_for_unrestricted_constant():
    y = np.log(load_prices().loc[:"2015-12", ["brent", "wti"]]).to_numpy()
    for k in (1, 2, 4):
        assert np.allclose(johansen_trace(y, k, restricted_constant=False), coint_johansen(y, 0, k).lr1)


def test_vecm_records_a_rank_per_origin():
    prices = load_prices()
    origins = list(pd.date_range("2015-12-01", "2016-05-01", freq="MS"))
    ranks = run_vecm(prices, origins, 1).attrs["vecm"]["ranks"]
    assert len(ranks) == len(origins) and set(ranks) <= {0, 1, 2}
