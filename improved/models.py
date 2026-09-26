"""Forecasting models under one protocol.

Every model answers the same question: standing at the end of month t, using
only data observed up to and including t, what will the Brent and WTI prices
be in month t+h?

Each ``run_*`` function takes the full price panel, a list of forecast origins
and a horizon, and returns a DataFrame of *price* forecasts indexed by target
month (origin + h), with columns ``brent`` and ``wti``. Internally every model
works on log-returns and forecasts the h-month log change, which is mapped back
to $/bbl as P_t * exp(forecast).

Cheap models (AR, VAR, VECM) are re-estimated at every origin. XGBoost and the
LSTM are re-trained once per calendar year and then roll forward month by month
with fresh inputs - the usual compromise for ML models in the forecasting
literature, and still strictly out-of-sample.
"""
import os
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank, select_order

from data import TARGETS, log_returns

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*frequency.*")

MAX_LAGS = 12


def _to_prices(prices, origins, h, logret_fc):
    """Map h-step log-change forecasts (rows aligned with origins) to price forecasts."""
    base = prices.loc[origins, TARGETS].to_numpy()
    idx = pd.DatetimeIndex(origins) + pd.DateOffset(months=h)
    return pd.DataFrame(base * np.exp(np.asarray(logret_fc, dtype=float)), index=idx, columns=TARGETS)


def _yearly_groups(origins):
    """Split origins into runs that share a calendar year (one re-train per run)."""
    s = pd.Series(pd.DatetimeIndex(origins))
    return [list(g) for _, g in s.groupby(s.dt.year)]


# --------------------------------------------------------------------------- #
# Benchmarks and econometric models
# --------------------------------------------------------------------------- #

def run_random_walk(prices, origins, h):
    """No-change forecast: next price = today's price. The standard benchmark for oil."""
    return _to_prices(prices, origins, h, np.zeros((len(origins), 2)))


def run_ar(prices, origins, h):
    """Univariate AR(p) on each oil series' log-returns; p chosen by BIC on the first window."""
    rets = log_returns(prices)
    out = np.zeros((len(origins), 2))
    for j, tgt in enumerate(TARGETS):
        first = rets.loc[: origins[0], tgt]
        lags = ar_select_order(first, maxlag=MAX_LAGS, ic="bic", trend="c").ar_lags or [1]
        for i, t in enumerate(origins):
            y = rets.loc[:t, tgt].reset_index(drop=True)
            res = AutoReg(y, lags=lags, trend="c").fit()
            out[i, j] = res.predict(start=len(y), end=len(y) + h - 1).sum()
    return _to_prices(prices, origins, h, out)


def run_var(prices, origins, h):
    """VAR on log-returns of all ten commodities; lag order chosen by BIC on the first window.

    Unlike the original VARMAX, the other commodities are endogenous here, so the
    model has to forecast them too instead of being handed their future values.
    """
    rets = log_returns(prices)
    p = max(VAR(rets.loc[: origins[0]].to_numpy()).select_order(MAX_LAGS).bic, 1)
    cols = [rets.columns.get_loc(c) for c in TARGETS]
    out = np.zeros((len(origins), 2))
    for i, t in enumerate(origins):
        y = rets.loc[:t].to_numpy()
        res = VAR(y).fit(p)
        out[i] = res.forecast(y[-p:], steps=h)[:, cols].sum(axis=0)
    return _to_prices(prices, origins, h, out)


VECM_DETERMINISTIC = "co"  # unrestricted constant: the case statsmodels' Johansen test (det_order=0) assumes


def run_vecm(prices, origins, h):
    """Vector error-correction model on log Brent and log WTI.

    Specification kept internally consistent:
    - deterministic terms: unrestricted constant ("co"), the same case as the
      Johansen critical values used to pick the rank (det_order=0). A restricted
      constant is checked as a robustness case in diagnostics.py.
    - lag: k_ar_diff chosen once by BIC on data up to the first origin.
    - rank: re-tested with the Johansen trace test (5%) at every origin, on data up
      to that origin. Rank 0 is allowed and gives a VAR in differences.
    """
    logp = np.log(prices[TARGETS])
    first = logp.loc[: origins[0]].to_numpy()
    k = select_order(first, maxlags=MAX_LAGS, deterministic=VECM_DETERMINISTIC).bic
    out, ranks = np.zeros((len(origins), 2)), []
    for i, t in enumerate(origins):
        y = logp.loc[:t].to_numpy()
        rank = select_coint_rank(y, det_order=0, k_ar_diff=k, method="trace", signif=0.05).rank
        res = VECM(y, k_ar_diff=k, coint_rank=rank, deterministic=VECM_DETERMINISTIC).fit()
        out[i] = res.predict(steps=h)[-1] - y[-1]
        ranks.append(rank)
    fc = _to_prices(prices, origins, h, out)
    fc.attrs["vecm"] = {"k_ar_diff": int(k), "origins": [f"{t:%Y-%m}" for t in origins], "ranks": [int(r) for r in ranks]}
    return fc


# --------------------------------------------------------------------------- #
# Machine-learning models
# --------------------------------------------------------------------------- #

N_RET_LAGS = 6  # XGBoost: months of lagged returns per series

# Hyperparameters used before validation-based tuning was added (kept as the
# default and as one candidate in tuning.py).
XGB_DEFAULT = dict(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8,
                   colsample_bytree=0.8, min_child_weight=5)
LSTM_DEFAULT = dict(window=12, units=32)


def make_features(prices):
    """Features known at the end of month t (every column uses data <= t only)."""
    rets = log_returns(prices)
    feats = {f"{c}_ret_l{k}": rets[c].shift(k) for c in rets for k in range(N_RET_LAGS)}
    feats["brent_wti_spread"] = np.log(prices["brent"] / prices["wti"])
    feats["brent_vol_12m"] = rets["brent"].rolling(12).std()
    return pd.DataFrame(feats).dropna()


def make_targets(prices, h):
    """h-month-ahead log change for each oil series, indexed by origin month."""
    logp = np.log(prices[TARGETS])
    return (logp.shift(-h) - logp).dropna()


def _training_rows(index, cutoff, h):
    """Origins s whose target month s+h has already been observed at the cutoff."""
    return index[index + pd.DateOffset(months=h) <= cutoff]


def run_xgboost(prices, origins, h, params=None, seed=42):
    from xgboost import XGBRegressor

    X_all, Y_all = make_features(prices), make_targets(prices, h)
    params = {**XGB_DEFAULT, **(params or {}), "random_state": seed, "n_jobs": 4}
    out, models = np.zeros((len(origins), 2)), {}
    pos = {t: i for i, t in enumerate(origins)}
    for group in _yearly_groups(origins):
        rows = _training_rows(X_all.index.intersection(Y_all.index), group[0], h)
        for j, tgt in enumerate(TARGETS):
            m = XGBRegressor(**params).fit(X_all.loc[rows], Y_all.loc[rows, tgt])
            models[tgt] = m
            out[[pos[t] for t in group], j] = m.predict(X_all.loc[group])
    fc = _to_prices(prices, origins, h, out)
    fc.attrs["models"] = models          # last-year models, used for SHAP
    fc.attrs["X"] = X_all.loc[origins]
    return fc


def _sequences(feats, ends, window):
    """Stack the `window` rows ending at each date in ``ends`` -> (n, window, n_features)."""
    loc = feats.index.get_indexer(ends)
    return np.stack([feats.iloc[i - window + 1 : i + 1].to_numpy() for i in loc])


def run_lstm(prices, origins, h, window=LSTM_DEFAULT["window"], units=LSTM_DEFAULT["units"],
             seeds=(0, 1, 2), max_epochs=300, verbose=0):
    """Two-output LSTM on `window`-month sequences of returns; averaged over several seeds."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    from tensorflow import keras

    tf.config.experimental.enable_op_determinism()
    rets = log_returns(prices)
    feats = rets.join(np.log(prices["brent"] / prices["wti"]).rename("spread")).dropna()
    Y_all = make_targets(prices, h)
    valid = feats.index[window - 1 :].intersection(Y_all.index)

    out = np.zeros((len(origins), 2))
    pos = {t: i for i, t in enumerate(origins)}
    histories = []
    for group in _yearly_groups(origins):
        rows = _training_rows(valid, group[0], h)
        X, Y = _sequences(feats, rows, window), Y_all.loc[rows].to_numpy()
        # Scale with training statistics only.
        x_mu, x_sd = X.reshape(-1, X.shape[-1]).mean(0), X.reshape(-1, X.shape[-1]).std(0)
        y_mu, y_sd = Y.mean(0), Y.std(0)
        Xs, Ys = (X - x_mu) / x_sd, (Y - y_mu) / y_sd
        X_new = (_sequences(feats, group, window) - x_mu) / x_sd

        preds = []
        for seed in seeds:
            keras.utils.set_random_seed(seed)
            model = keras.Sequential([
                keras.Input(shape=Xs.shape[1:]),
                keras.layers.LSTM(units),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(16, activation="relu"),
                keras.layers.Dense(2),
            ])
            model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
            stop = keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
            hist = model.fit(Xs, Ys, validation_split=0.15, shuffle=False, epochs=max_epochs,
                             batch_size=32, callbacks=[stop], verbose=verbose)
            histories.append({"year": group[0].year, "seed": seed, **{k: v for k, v in hist.history.items()}})
            preds.append(model(X_new.astype("float32"), training=False).numpy())
        out[[pos[t] for t in group]] = np.mean(preds, axis=0) * y_sd + y_mu
        keras.backend.clear_session()
    fc = _to_prices(prices, origins, h, out)
    fc.attrs["histories"] = histories
    return fc


MODELS = {
    "Random walk": run_random_walk,
    "AR": run_ar,
    "VAR": run_var,
    "VECM": run_vecm,
    "XGBoost": run_xgboost,
    "LSTM": run_lstm,
}
