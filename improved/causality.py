"""Granger causality, impulse responses and variance decomposition for Brent and WTI.

These answer the thesis's first research question directly, rather than
through forecast accuracy: do related commodity markets carry information
about oil prices, and how does oil respond when they move?

    python improved/causality.py                          # thesis data
    python improved/causality.py --data data/CMO-Historical-Data-Monthly_latest.xlsx --out improved/results_latest

All analysis uses monthly log-returns, which are stationary (see diagnostics.py).

- Granger causality: pairwise two-variable VARs, lag chosen by BIC (at least 1),
  F-tests in both directions, Holm-corrected for the number of tests. A joint
  test asks whether the eight non-oil commodities together help predict oil.
- Impulse responses: generalized IRFs (Pesaran & Shin, 1998) from the
  ten-variable VAR, so results do not depend on the order of the variables.
  Responses are cumulated, so they read as the % change in the oil *price*
  after a typical (one standard deviation) monthly shock. 90% bands come from
  a residual bootstrap.
- Variance decomposition: generalized FEVD (Lanne & Nyberg, 2016 normalisation)
  at 12 months: the share of each oil price's forecast surprises attributable
  to each market.

Caveat: the World Bank series are monthly averages. Averaging can create
spurious lead-lag relations, so Granger results here are an upper bound on
genuine predictability.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.api import VAR

from data import DEFAULT_XLSX, LABELS, ROOT, TARGETS, load_prices, log_returns

TRAIN_END = "2015-12"
MAX_LAGS = 12
NAMES = {
    "brent": "Brent", "wti": "WTI", "coal_au": "Coal (Australia)", "coal_za": "Coal (South Africa)",
    "gas_us": "Natural gas (US)", "gas_eu": "Natural gas (Europe)", "lng_jp": "LNG (Japan)",
    "palm_oil": "Palm oil", "copper": "Copper", "gold": "Gold",
}
COLORS = {"brent": "#2a78d6", "wti": "#eb6834"}
SEQ = ["#f4f7fb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
    "figure.constrained_layout.use": True,
})


def _lag(rets: pd.DataFrame) -> int:
    return max(VAR(rets).select_order(MAX_LAGS).bic, 1)


# --------------------------------------------------------------------------- #
# Granger causality
# --------------------------------------------------------------------------- #

def granger(rets: pd.DataFrame) -> pd.DataFrame:
    """Pairwise F-tests in both directions between each oil series and every other series."""
    rows = []
    for tgt in TARGETS:
        for x in rets.columns:
            if x == tgt:
                continue
            pair = rets[[tgt, x]]
            p = _lag(pair)
            res = VAR(pair).fit(p)
            for caused, causing in [(tgt, x), (x, tgt)]:
                t = res.test_causality(caused, causing, kind="f")
                rows.append({"caused": caused, "causing": causing, "lag": p,
                             "F": t.test_statistic, "p": t.pvalue})
    out = pd.DataFrame(rows).drop_duplicates(["caused", "causing"])  # brent<->wti appears twice
    out["p_holm"] = multipletests(out["p"], method="holm")[1]
    out["significant"] = out["p_holm"] < 0.05
    return out.reset_index(drop=True)


def granger_joint(rets: pd.DataFrame) -> pd.DataFrame:
    """Do the eight non-oil commodities, jointly, help predict each oil price in the full VAR?"""
    others = [c for c in rets.columns if c not in TARGETS]
    p = _lag(rets)
    res = VAR(rets).fit(p)
    rows = []
    for tgt in TARGETS:
        t = res.test_causality(tgt, others, kind="f")
        rows.append({"caused": tgt, "causing": "all 8 non-oil commodities", "lag": p,
                     "F": t.test_statistic, "p": t.pvalue})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Generalized impulse responses and variance decomposition
# --------------------------------------------------------------------------- #

def girf(ma: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Generalized IRF. ma: (H+1, K, K) MA coefficients; returns (H+1, K responses, K shocks).

    Response to a one-standard-deviation shock in variable j: Psi_h @ Sigma[:, j] / sqrt(Sigma[j, j]).
    """
    return ma @ sigma / np.sqrt(np.diag(sigma))


def gfevd(ma: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Generalized FEVD over the horizons in ma, rows normalised to sum to 1. Returns (K, K)."""
    num = ((ma @ sigma) ** 2).sum(axis=0) / np.diag(sigma)        # (K, K): [i, j]
    den = np.einsum("hik,kl,hil->i", ma, sigma, ma)                # (K,)
    theta = num / den[:, None]
    return theta / theta.sum(axis=1, keepdims=True)


def _simulate(res, rng) -> np.ndarray:
    """One residual-bootstrap sample from a fitted VAR, same length and start values."""
    y, u = res.endog, res.resid
    u = np.asarray(u)
    p, (n, k) = res.k_ar, y.shape
    sim = np.empty_like(y)
    sim[:p] = y[:p]
    draws = u[rng.integers(0, len(u), size=n)]
    for t in range(p, n):
        sim[t] = res.intercept + sum(res.coefs[i] @ sim[t - 1 - i] for i in range(p)) + draws[t]
    return sim


def impulse_responses(rets: pd.DataFrame, horizon=12, n_boot=500, seed=0):
    """Cumulative generalized IRFs of Brent and WTI (in %), with 90% bootstrap bands."""
    p = _lag(rets)
    res = VAR(rets.to_numpy()).fit(p)
    sigma = np.asarray(res.sigma_u)
    point = np.cumsum(girf(res.ma_rep(horizon), sigma), axis=0) * 100

    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot,) + point.shape)
    for b in range(n_boot):
        rb = VAR(_simulate(res, rng)).fit(p)
        boots[b] = np.cumsum(girf(rb.ma_rep(horizon), np.asarray(rb.sigma_u)), axis=0) * 100
    lo, hi = np.percentile(boots, [5, 95], axis=0)

    cols = list(rets.columns)
    tidx = [cols.index(t) for t in TARGETS]
    rows = []
    for j, shock in enumerate(cols):
        for i, tgt in zip(tidx, TARGETS):
            for h in range(horizon + 1):
                rows.append({"shock": shock, "response": tgt, "month": h, "pct": point[h, i, j],
                             "lo90": lo[h, i, j], "hi90": hi[h, i, j]})
    shock_size = pd.Series(np.sqrt(np.diag(sigma)) * 100, index=cols, name="shock_sd_pct")
    fevd = pd.DataFrame(gfevd(res.ma_rep(horizon - 1), sigma)[tidx], index=TARGETS, columns=cols)
    return pd.DataFrame(rows), shock_size, fevd, p


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #

def _seq_color(v, vmax=4.0):
    return SEQ[min(int(np.clip(v, 0, vmax) / vmax * (len(SEQ) - 1) + 0.5), len(SEQ) - 1)]


def plot_granger(tables: dict, path):
    """Matrix of Holm-corrected p-values; darker = stronger evidence of predictive content."""
    order = [c for c in NAMES if c not in TARGETS] + TARGETS
    cols = [("brent", "to"), ("wti", "to"), ("brent", "from"), ("wti", "from")]
    headers = ["→ Brent", "→ WTI", "Brent →", "WTI →"]
    fig, axes = plt.subplots(1, len(tables), figsize=(5.2 * len(tables) + 0.6, 5.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (title, g) in zip(axes, tables.items()):
        ax.set_xlim(0, len(cols)); ax.set_ylim(len(order), 0); ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for r, var in enumerate(order):
            for c, (oil, direction) in enumerate(cols):
                if var == oil:
                    ax.text(c + 0.5, r + 0.5, "—", ha="center", va="center", color=INK_2)
                    continue
                q = g.query("caused == @oil and causing == @var") if direction == "to" else \
                    g.query("caused == @var and causing == @oil")
                if q.empty:
                    continue
                ph = q["p_holm"].iloc[0]
                v = -np.log10(max(ph, 1e-12))
                ax.add_patch(plt.Rectangle((c + 0.04, r + 0.06), 0.92, 0.88, color=_seq_color(v), lw=0))
                ax.text(c + 0.5, r + 0.5, f"{ph:.3f}" if ph >= 0.001 else "<0.001", ha="center", va="center",
                        fontsize=9, fontweight="bold" if ph < 0.05 else "normal",
                        color="white" if v >= 2 else INK)
        ax.set_xticks(np.arange(len(cols)) + 0.5, headers)
        ax.xaxis.tick_top()
        ax.set_yticks(np.arange(len(order)) + 0.5, [NAMES[v] for v in order])
        ax.tick_params(length=0)
        ax.set_title(title, pad=28)
    fig.suptitle("Granger causality: Holm-corrected p-values (bold = significant at 5%)", x=0.02, ha="left")
    sm = plt.cm.ScalarMappable(cmap=matplotlib.colors.ListedColormap(SEQ), norm=plt.Normalize(0, 4))
    cb = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
    cb.set_ticks([0, 1.3, 2, 3, 4], labels=["1", "0.05", "0.01", "0.001", "≤0.0001"])
    cb.set_label("Holm p-value (darker = stronger)")
    cb.outline.set_visible(False)
    fig.savefig(path); plt.close(fig)


def plot_irf(irf: pd.DataFrame, shock_size: pd.Series, title_suffix, path):
    shocks = [c for c in NAMES if c not in TARGETS]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2), sharex=True, sharey=True)
    for ax, s in zip(axes.flat, shocks):
        ax.axhline(0, color=INK_2, lw=0.8)
        b = irf.query("shock == @s and response == 'brent'")
        w = irf.query("shock == @s and response == 'wti'")
        ax.fill_between(b.month, b.lo90, b.hi90, color=COLORS["brent"], alpha=0.18, lw=0, label="Brent 90% band")
        ax.plot(b.month, b.pct, color=COLORS["brent"], lw=2, label="Brent")
        ax.plot(w.month, w.pct, color=COLORS["wti"], lw=2, ls="--", label="WTI")
        ax.set_title(f"{NAMES[s]} shock (+{shock_size[s]:.1f}%)", fontsize=10)
        ax.set_xticks([0, 3, 6, 9, 12])
    for ax in axes[1]:
        ax.set_xlabel("Months after shock")
    for ax in axes[:, 0]:
        ax.set_ylabel("Change in oil price (%)")
    axes[0, -1].legend(loc="upper right")
    fig.suptitle(f"How oil prices respond to a typical one-month shock in each market {title_suffix}\n"
                 "Cumulative generalized impulse responses; WTI's band is similar to Brent's", x=0.02, ha="left")
    fig.savefig(path); plt.close(fig)


def plot_fevd(fevd: pd.DataFrame, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    for ax, tgt in zip(axes, TARGETS):
        s = (fevd.loc[tgt] * 100).sort_values()
        colors = [COLORS[c] if c in TARGETS else "#9ec5f4" for c in s.index]
        bars = ax.barh([NAMES[c] for c in s.index], s.values, color=colors, height=0.65)
        ax.bar_label(bars, fmt="%.1f%%", padding=3, color=INK, fontsize=9)
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
        ax.set_title(f"{LABELS[tgt]}")
        ax.set_xlabel("Share of 12-month forecast error variance (%)")
        ax.set_xlim(0, s.max() * 1.18)
    fig.suptitle("Where oil price surprises come from (generalized variance decomposition, 12 months)",
                 x=0.02, ha="left")
    fig.savefig(path); plt.close(fig)


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DEFAULT_XLSX))
    ap.add_argument("--out", default=str(ROOT / "improved" / "results"))
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--horizon", type=int, default=12)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)

    prices = load_prices(args.data)
    rets = log_returns(prices)
    full_label = f"Full sample {rets.index[0]:%Y-%m} to {rets.index[-1]:%Y-%m}"
    train_label = f"Training window {rets.index[0]:%Y-%m} to {TRAIN_END}"

    g_full, g_train = granger(rets), granger(rets.loc[:TRAIN_END])
    pd.concat([g_full.assign(sample="full"), g_train.assign(sample="train")]).to_csv(
        out / "tables/granger_pairwise.csv", index=False)
    joint = pd.concat([granger_joint(rets).assign(sample="full"),
                       granger_joint(rets.loc[:TRAIN_END]).assign(sample="train")])
    joint.to_csv(out / "tables/granger_joint.csv", index=False)
    plot_granger({full_label: g_full, train_label: g_train}, out / "plots/granger_causality.png")

    irf, shock_size, fevd, p = impulse_responses(rets, args.horizon, args.boot)
    irf.to_csv(out / "tables/irf_cumulative.csv", index=False)
    shock_size.to_csv(out / "tables/irf_shock_sizes.csv")
    fevd.to_csv(out / "tables/fevd_12m.csv")
    plot_irf(irf, shock_size, f"({full_label.lower()}, VAR({p}))", out / "plots/impulse_responses.png")
    plot_fevd(fevd, out / "plots/variance_decomposition.png")

    with pd.option_context("display.width", 160, "display.float_format", "{:.4f}".format):
        print(full_label, f"| VAR lag {p}")
        print("\nSignificant pairwise Granger links (Holm 5%):")
        print(pd.concat([g_full.assign(sample="full"), g_train.assign(sample="train")])
              .query("significant")[["sample", "causing", "caused", "lag", "F", "p_holm"]].to_string(index=False))
        print("\nJoint test (8 non-oil commodities → oil):\n", joint.to_string(index=False))
        m12 = irf.query("month == 12").pivot(index="shock", columns="response", values=["pct", "lo90", "hi90"])
        print("\nCumulative response after 12 months (%):\n", m12.round(2))
        print("\n12-month FEVD (%):\n", (fevd * 100).round(1).T)


if __name__ == "__main__":
    main()
