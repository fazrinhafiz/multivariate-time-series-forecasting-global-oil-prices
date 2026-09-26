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
from statsmodels.tsa.vector_ar.vecm import coint_johansen, select_order

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


# MacKinnon-Haug-Michelis (1999) 5% trace critical values for two variables,
# restricted constant (Case 2, constant only inside the cointegrating relation).
CASE2_TRACE_CV5 = {0: 20.2618, 1: 9.1645}


def johansen_trace(y: np.ndarray, k: int, restricted_constant: bool) -> np.ndarray:
    """Johansen trace statistics for rank 0..n-1.

    restricted_constant=False: unrestricted constant (Case 3; should match statsmodels
    coint_johansen with det_order=0). True: constant restricted to the cointegrating
    relation (Case 2), which statsmodels does not provide.
    """
    dy = np.diff(y, axis=0)
    n, T = y.shape[1], len(dy) - k
    z0 = dy[k:]
    z1 = y[k:-1]
    lagged = [dy[k - i: len(dy) - i] for i in range(1, k + 1)]
    z2 = np.hstack(lagged) if lagged else np.empty((T, 0))
    if restricted_constant:
        z1 = np.hstack([z1, np.ones((T, 1))])
    else:
        z2 = np.hstack([z2, np.ones((T, 1))])

    def resid(z):
        if z2.shape[1] == 0:
            return z
        return z - z2 @ np.linalg.lstsq(z2, z, rcond=None)[0]

    r0, r1 = resid(z0), resid(z1)
    s00, s11, s01 = r0.T @ r0 / T, r1.T @ r1 / T, r0.T @ r1 / T
    m = np.linalg.solve(s11, s01.T) @ np.linalg.solve(s00, s01)
    lam = np.sort(np.linalg.eigvals(m).real)[::-1][:n]
    return np.array([-T * np.log(1 - lam[r:]).sum() for r in range(n)])


def johansen(prices: pd.DataFrame) -> pd.DataFrame:
    """Johansen tests for log Brent and log WTI, using the VECM's own lag (BIC).

    Main case (used by the VECM): unrestricted constant, trace and max-eigenvalue.
    Robustness case: restricted constant (trace only).
    """
    logp = np.log(prices[TARGETS])
    k = select_order(logp.to_numpy(), maxlags=12, deterministic="co").bic
    res = coint_johansen(logp, det_order=0, k_ar_diff=k)
    case2 = johansen_trace(logp.to_numpy(), k, restricted_constant=True)
    rows = []
    for r in range(2):
        rows.append({"case": "unrestricted constant (used by VECM)", "k_ar_diff": k, "H0: rank <=": r,
                     "trace stat": res.lr1[r], "trace 5% cv": res.cvt[r, 1],
                     "max-eig stat": res.lr2[r], "max-eig 5% cv": res.cvm[r, 1],
                     "reject H0 (trace)": res.lr1[r] > res.cvt[r, 1]})
    for r in range(2):
        rows.append({"case": "restricted constant (robustness)", "k_ar_diff": k, "H0: rank <=": r,
                     "trace stat": case2[r], "trace 5% cv": CASE2_TRACE_CV5[r],
                     "reject H0 (trace)": case2[r] > CASE2_TRACE_CV5[r]})
    return pd.DataFrame(rows)


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
