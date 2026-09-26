"""Run the full improved pipeline: diagnostics, all models, all horizons, tables and plots.

    python improved/run_all.py              # full run (about 15-25 min on a laptop CPU)
    python improved/run_all.py --quick      # smoke test: h=1, one LSTM seed, few epochs
    python improved/run_all.py --replot     # redraw figures from saved forecasts
    python improved/run_all.py --data path/to/newer/CMO-Historical-Data-Monthly.xlsx
"""
import argparse
import os
import sys
import time
from functools import partial
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import diagnostics
from data import DEFAULT_XLSX, LABELS, ROOT, TARGETS, load_prices, load_raw
from evaluate import score, score_by_period
from models import MODELS, run_lstm

TRAIN_END = "2015-12"
TEST_START = "2016-01"

# Categorical slots in fixed order (validated palette, see dataviz reference).
COLORS = {"AR": "#2a78d6", "VAR": "#eb6834", "VECM": "#1baf7a", "XGBoost": "#eda100", "LSTM": "#e87ba4"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
    "figure.constrained_layout.use": True,
})


def origins_for(prices, h):
    """Origins whose target month falls inside the test period."""
    targets = prices.loc[TEST_START:].index
    return list(targets - pd.DateOffset(months=h))


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def plot_rel_rmse_by_horizon(scores, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, tgt in zip(axes, TARGETS):
        s = scores[(scores.target == tgt) & (scores.model != "Random walk")]
        ax.axhline(1, color=INK_2, lw=1.2, ls="--")
        ax.text(s.horizon.max() + 0.15, 1, "Random walk", va="center", color=INK_2, fontsize=9)
        for model, g in s.groupby("model", sort=False):
            ax.plot(g.horizon, g["rel. RMSE vs RW"], marker="o", ms=5, lw=2, color=COLORS[model], label=model)
        ax.set_title(LABELS[tgt])
        ax.set_xlabel("Forecast horizon (months)")
        ax.set_xticks(sorted(s.horizon.unique()))
    axes[0].set_ylabel("RMSE relative to random walk\n(below 1 = beats no-change)")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 0.9))
    fig.suptitle("Out-of-sample accuracy vs the no-change forecast, 2016-01 to test end", x=0.02, ha="left")
    fig.savefig(path); plt.close(fig)


def plot_rel_rmse_by_period(periods, h, path):
    """Dot plot: bars would have to start at 0 and hide differences of a few percent."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    models = [m for m in COLORS if m in set(periods.model)]
    offsets = np.linspace(-0.3, 0.3, len(models))
    for ax, tgt in zip(axes, TARGETS):
        s = periods[(periods.target == tgt) & (periods.model != "Random walk")]
        names = list(dict.fromkeys(s.period))
        x = np.arange(len(names))
        ax.axhline(1, color=INK_2, lw=1.2, ls="--")
        for off, m in zip(offsets, models):
            vals = s[s.model == m].set_index("period").loc[names, "rel. RMSE vs RW"]
            ax.vlines(x + off, 1, vals, color=COLORS[m], lw=2)
            ax.plot(x + off, vals, "o", ms=8, color=COLORS[m], mec="white", mew=1.5, label=m)
        ax.set_xticks(x, [n.replace(" (", "\n(") for n in names])
        ax.grid(axis="x", visible=False)
        ax.set_title(LABELS[tgt])
    axes[0].set_ylabel("RMSE relative to random walk\n(below 1 = beats no-change)")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 0.95))
    fig.suptitle(f"Accuracy by period, {h}-month horizon", x=0.02, ha="left")
    fig.savefig(path); plt.close(fig)


def plot_forecasts(actual, forecasts, h, tgt, path):
    models = [m for m in COLORS if m in forecasts]
    fig, axes = plt.subplots(len(models), 1, figsize=(10, 1.9 * len(models)), sharex=True, sharey=True)
    for ax, m in zip(np.atleast_1d(axes), models):
        ax.plot(actual.index, actual[tgt], color=INK, lw=1.4, label="Actual")
        ax.plot(forecasts[m].index, forecasts[m][tgt], color=COLORS[m], lw=2, label=m)
        ax.text(0.01, 0.85, m, transform=ax.transAxes, fontweight="bold", color=INK)
        ax.set_ylabel("$/bbl")
    np.atleast_1d(axes)[0].legend(["Actual", "Forecast"], loc="upper right", ncol=2)
    fig.suptitle(f"{LABELS[tgt]}: {h}-month-ahead forecasts vs actual", x=0.02, ha="left")
    fig.savefig(path); plt.close(fig)


def plot_original_vs_naive(table, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    orig_colors = {"VARMAX": "#2a78d6", "XGBoost": "#eda100", "LSTM": "#e87ba4"}
    for ax, tgt in zip(axes, TARGETS):
        s = table[table.target == tgt]
        models = [m for m in orig_colors if m in set(s.model)]
        vals = s.set_index("model").loc[models, "RMSE"]
        bars = ax.bar(models, vals, color=[orig_colors[m] for m in models], width=0.6)
        ax.bar_label(bars, fmt="%.2f", color=INK, fontsize=9, padding=2)
        rw = s.set_index("model").loc["Random walk (h=1)", "RMSE"]
        ax.axhline(rw, color=INK, lw=1.4, ls="--", label=f"No-change forecast ({rw:.2f})")
        ax.legend(loc="upper right")
        ax.grid(axis="x", visible=False)
        ax.set_title(LABELS[tgt])
    axes[0].set_ylabel("Test RMSE ($/bbl)")
    fig.suptitle("Original results vs the no-change forecast on the same test months", x=0.02, ha="left")
    fig.savefig(path); plt.close(fig)


def plot_shap(fc_xgb, tgt, path):
    import shap
    model, X = fc_xgb.attrs["models"][tgt], fc_xgb.attrs["X"]
    values = shap.TreeExplainer(model)(X)
    shap.plots.beeswarm(values, max_display=12, show=False)
    plt.title(f"XGBoost ({LABELS[tgt]}, h=1): top features by SHAP value", loc="left")
    plt.savefig(path); plt.close("all")


# --------------------------------------------------------------------------- #

def original_vs_naive(raw):
    """Compare the submitted results with the random walk on the identical test months."""
    p = raw[TARGETS]
    actual, rw = p.loc[TEST_START:], p.shift(1).loc[TEST_START:]
    rows = []
    for tgt, fname in [("brent", "evaluation_results_brent.csv"), ("wti", "evaluation_results_wti.csv")]:
        shipped = pd.read_csv(ROOT / "tables" / fname, index_col=0)
        for m, r in shipped.iterrows():
            rows.append({"target": tgt, "model": m, "RMSE": r.RMSE, "MAE": r.MAE, "MAPE %": r.MAPE})
        e = actual[tgt] - rw[tgt]
        rows.append({"target": tgt, "model": "Random walk (h=1)", "RMSE": np.sqrt((e**2).mean()),
                     "MAE": e.abs().mean(), "MAPE %": (e.abs() / actual[tgt]).mean() * 100})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DEFAULT_XLSX))
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 6])
    ap.add_argument("--out", default=str(ROOT / "improved" / "results"))
    ap.add_argument("--quick", action="store_true", help="h=1, one LSTM seed, 30 epochs")
    ap.add_argument("--replot", action="store_true", help="redraw figures from saved forecasts (no refitting)")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    models = dict(MODELS)
    if args.quick:
        args.horizons = [1]
        models["LSTM"] = partial(run_lstm, seeds=(0,), max_epochs=30)

    prices = load_prices(args.data)
    train = prices.loc[:TRAIN_END]
    print(f"Sample {prices.index[0]:%Y-%m} to {prices.index[-1]:%Y-%m}; "
          f"train to {TRAIN_END}, test {TEST_START} to {prices.index[-1]:%Y-%m} "
          f"({len(prices.loc[TEST_START:])} months)")

    # Diagnostics on the training window only.
    diagnostics.stationarity(train).to_csv(out / "tables/stationarity_adf_kpss.csv", index=False)
    diagnostics.johansen(train).to_csv(out / "tables/johansen_brent_wti.csv", index=False)
    diagnostics.var_lag_selection(train).to_csv(out / "tables/var_lag_selection.csv")
    diagnostics.var_residuals(train).to_csv(out / "tables/var_residual_tests.csv", index=False)

    ovn = original_vs_naive(load_raw(args.data))
    ovn.to_csv(out / "tables/original_results_vs_random_walk.csv", index=False)
    plot_original_vs_naive(ovn, out / "plots/original_vs_random_walk.png")

    actual = prices.loc[TEST_START:, TARGETS]
    if args.replot:
        long = pd.read_csv(out / "tables/forecasts_all_horizons.csv", parse_dates=["date"])
        by_h = {h: {m: g.pivot(index="date", columns="target", values="forecast")[TARGETS].asfreq("MS")
                    for m, g in gh.groupby("model", sort=False)}
                for h, gh in long.groupby("horizon")}
        make_plots(actual, by_h, pd.read_csv(out / "tables/scores_all_horizons.csv"), out)
        return

    all_scores, by_h = [], {}
    for h in args.horizons:
        origins = origins_for(prices, h)
        forecasts = {}
        for name, fn in models.items():
            t0 = time.time()
            forecasts[name] = fn(prices, origins, h)
            print(f"  h={h:<2} {name:<12} {time.time() - t0:6.1f}s", flush=True)
        by_h[h] = forecasts
        all_scores.append(score(actual, forecasts, h).assign(horizon=h))
        score_by_period(actual, forecasts).to_csv(out / f"tables/period_scores_h{h}.csv", index=False)
        if h == 1:
            for tgt in TARGETS:
                plot_shap(forecasts["XGBoost"], tgt, out / f"plots/shap_xgboost_h1_{tgt}.png")
            pd.DataFrame(forecasts["LSTM"].attrs["histories"]).to_json(out / "tables/lstm_training_histories.json")

    scores = pd.concat(all_scores)
    scores.to_csv(out / "tables/scores_all_horizons.csv", index=False)
    long = pd.concat(
        f.rename_axis("date").reset_index().melt("date", var_name="target", value_name="forecast")
         .assign(model=m, horizon=h)
        for h, fcs in by_h.items() for m, f in fcs.items())
    long[["date", "horizon", "model", "target", "forecast"]].to_csv(out / "tables/forecasts_all_horizons.csv", index=False)
    make_plots(actual, by_h, scores, out)

    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(scores.set_index(["horizon", "target", "model"]))


def make_plots(actual, by_h, scores, out):
    """Every figure except SHAP, which needs the fitted models."""
    for h, forecasts in by_h.items():
        for tgt in TARGETS:
            plot_forecasts(actual, forecasts, h, tgt, out / f"plots/forecasts_h{h}_{tgt}.png")
    if 1 in by_h:
        plot_rel_rmse_by_period(score_by_period(actual, by_h[1]), 1, out / "plots/rel_rmse_by_period_h1.png")
    plot_rel_rmse_by_horizon(scores, out / "plots/rel_rmse_by_horizon.png")


if __name__ == "__main__":
    main()
