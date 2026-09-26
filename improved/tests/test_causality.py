"""Sanity checks for the causality module on synthetic data with a known answer."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causality import gfevd, girf  # noqa: E402
from statsmodels.tsa.api import VAR  # noqa: E402

rng = np.random.default_rng(1)


def test_girf_own_shock_on_impact_is_one_standard_deviation():
    sigma = np.array([[4.0, 1.2], [1.2, 9.0]])
    ma = np.stack([np.eye(2), rng.normal(size=(2, 2))])
    g = girf(ma, sigma)
    assert np.allclose(np.diag(g[0]), np.sqrt(np.diag(sigma)))


def test_girf_equals_cholesky_irf_when_shocks_are_uncorrelated():
    sigma = np.diag([4.0, 9.0, 1.0])
    ma = np.concatenate([np.eye(3)[None], rng.normal(size=(5, 3, 3))])
    assert np.allclose(girf(ma, sigma), ma @ np.linalg.cholesky(sigma))


def test_gfevd_rows_are_shares():
    a = rng.normal(size=(3, 3))
    sigma = a @ a.T + np.eye(3)
    ma = np.concatenate([np.eye(3)[None], 0.3 * rng.normal(size=(11, 3, 3))])
    theta = gfevd(ma, sigma)
    assert np.allclose(theta.sum(axis=1), 1) and (theta >= 0).all()


def test_granger_finds_a_planted_lead_and_not_the_reverse():
    n = 500
    x = rng.normal(size=n)
    y = np.r_[0, 0.6 * x[:-1]] + rng.normal(size=n)
    res = VAR(pd.DataFrame({"y": y, "x": x})).fit(1)
    assert res.test_causality("y", "x", kind="f").pvalue < 0.001
    assert res.test_causality("x", "y", kind="f").pvalue > 0.05
