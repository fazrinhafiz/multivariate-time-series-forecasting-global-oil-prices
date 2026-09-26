"""Pre-modelling and residual diagnostics, run on the training window only.

These replace three tests in the original script that did not test what the
report says they test:
- Durbin-Watson was computed on raw price *levels*, not model residuals.
- Breusch-Pagan regressed forecast errors on random numbers (a different,
  meaningless result on every run).
- ADF was run on levels only, so it never showed that differencing works.
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from data import TARGETS, log_returns

warnings.filterwarnings("ignore")


def stationarity(prices: pd.DataFrame) -> pd.DataFrame:
    """ADF (H0: unit root) and KPSS (H0: stationary) on log levels and log-returns."""
    rows = []
    for name, df in {"log level": np.log(prices), "log-return": log_returns(prices)}.items():
        for c in df:
            rows.append({"series": c, "transform": name,
                         "ADF p-value": adfuller(df[c], autolag="AIC")[1],
                         "KPSS p-value": kpss(df[c], regression="c", nlags="auto")[1]})
    out = pd.DataFrame(rows)
    out["verdict"] = np.where((out["ADF p-value"] < 0.05) & (out["KPSS p-value"] > 0.05),
                              "stationary", "non-stationary")
    return out


def johansen(prices: pd.DataFrame) -> pd.DataFrame:
    """Johansen trace test for cointegration between log Brent and log WTI."""
    res = coint_johansen(np.log(prices[TARGETS]), det_order=0, k_ar_diff=2)
    return pd.DataFrame({"H0: rank <=": [0, 1], "trace stat": res.lr1,
                         "5% critical value": res.cvt[:, 1],
                         "reject H0": res.lr1 > res.cvt[:, 1]})


def var_lag_selection(prices: pd.DataFrame) -> pd.DataFrame:
    sel = VAR(log_returns(prices)).select_order(12)
    return pd.DataFrame([sel.selected_orders]).rename(index={0: "selected lag"})


def var_residuals(prices: pd.DataFrame) -> pd.DataFrame:
    """Ljung-Box (autocorrelation) and ARCH-LM (heteroskedasticity) on in-sample VAR residuals."""
    rets = log_returns(prices)
    p = max(VAR(rets).select_order(12).bic, 1)
    resid = VAR(rets).fit(p).resid
    rows = []
    for c in TARGETS:
        lb = acorr_ljungbox(resid[c], lags=[12])
        arch = het_arch(resid[c], nlags=12)
        rows.append({"series": c, "VAR lag": p,
                     "Ljung-Box(12) p-value": lb["lb_pvalue"].iloc[0],
                     "ARCH-LM(12) p-value": arch[1]})
    return pd.DataFrame(rows)
