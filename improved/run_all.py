"""Run the 2026 re-audit pipeline: diagnostics, tuning, all models and horizons, tables, plots.

    python improved/run_all.py              # re-audit on the thesis data (test 2016-01 to 2025-04)
    python improved/run_all.py --data data/CMO-Historical-Data-Monthly_latest.xlsx --out improved/results_latest
                                            # same design on newer data; months after 2025-04 are
                                            # scored separately as the untouched holdout
    python improved/run_all.py --quick      # smoke test: h=1, default hyperparameters, short LSTM
    python improved/run_all.py --replot     # redraw figures from saved forecasts

Evaluation windows (never mixed in one test):
- validation 2010-01 to 2015-12: hyperparameter choice only (tuning.py)
- test       2016-01 to 2025-04: the months the thesis was evaluated on
- holdout    2025-05 onwards:    data released after the re-audit design was fixed
"""
import argparse
import json
import os
import sys
import textwrap
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
from evaluate import PERIODS, add_holm, score, score_by_period
from models import LSTM_DEFAULT, MODELS, XGB_DEFAULT, run_lstm, run_xgboost
from tuning import VAL_END, VAL_START, tune

TRAIN_END = "2015-12"
TEST_START, TEST_END = "2016-01", "2025-04"
HOLDOUT_START = "2025-05"

# Categorical slots in fixed order (validated palette, see dataviz reference).
COLORS = {"AR": "#2a78d6", "VAR": "#eb6834", "VECM": "#1baf7a", "XGBoost": "#eda100", "LSTM": "#e87ba4"}
INK, INK_2, GRID, NAIVE = "#0b0b0b", "#52514e", "#e4e3df", "#b8b6ad"

plt.rcParams.update({
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
    "figure.constrained_layout.use": True,
})


def origins_for(prices, h):
    """Origins whose target month falls in the test or holdout window."""
    targets = prices.loc[TEST_START:].index
    return list(targets - pd.DateOffset(months=h))


def windows(prices):
    """The evaluation windows present in this data file."""
    out = {"test": (TEST_START, TEST_END)}
    if prices.index[-1] >= pd.Timestamp(HOLDOUT_START):
        out["holdout"] = (HOLDOUT_START, f"{prices.index[-1]:%Y-%m}")
    return out


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def plot_rel_rmse_by_horizon(scores, title, path):
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
    fig.suptitle(title, x=0.02, ha="left")
    fig.savefig(path); plt.close(fig)


def plot_rel_rmse_by_period(periods, h, path):
    """Dot plot: bars would have to start at 0 and hide differences of a few percent."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    models = [m for m in COLORS if m in set(periods.model)]
    offsets = np.linspace(-0.3, 0.3, len(models))
    for ax, tgt in zip(axes, TARGETS):
        s = periods[(periods.target == tgt) & (periods.model != "Random walk")]
        names = list(dict.fromkeys(s.period))
        months = s.drop_duplicates("period").set_index("period")["months"]
        x = np.arange(len(names))
        ax.axhline(1, color=INK_2, lw=1.2, ls="--")
        for off, m in zip(offsets, models):
            vals = s[s.model == m].set_index("period").loc[names, "rel. RMSE vs RW"]
            ax.vlines(x + off, 1, vals, color=COLORS[m], lw=2)
            ax.plot(x + off, vals, "o", ms=8, color=COLORS[m], mec="white", mew=1.5, label=m)
        ax.set_xticks(x, [textwrap.fill(n, 13) + f"\n{months[n]} months" for n in names])
        ax.grid(axis="x", visible=False)
        ax.set_title(LABELS[tgt])
    axes[0].set_ylabel("RMSE relative to random walk\n(below 1 = beats no-change)")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 0.95))
    fig.suptitle(f"Accuracy by period, {h}-month horizon (descriptive; windows chosen after seeing the data)",
                 x=0.02, ha="left")
    fig.savefig(path); plt.close(fig)


def plot_forecasts(actual, forecasts, h, tgt, path):
    models = [m for m in COLORS if m in forecasts]
    fig, axes = plt.subplots(len(models), 1, figsize=(10, 1.9 * len(models)), sharex=True, sharey=True)
    has_holdout = actual.index[-1] >= pd.Timestamp(HOLDOUT_START)
    for ax, m in zip(np.atleast_1d(axes), models):
        if has_holdout:
            ax.axvspan(pd.Timestamp(HOLDOUT_START), actual.index[-1], color=GRID, alpha=0.6, lw=0)
        ax.plot(actual.index, actual[tgt], color=INK, lw=1.4, label="Actual")
        ax.plot(forecasts[m].index, forecasts[m][tgt], color=COLORS[m], lw=2, label=m)
        ax.text(0.01, 0.85, m, transform=ax.transAxes, fontweight="bold", color=INK)
        ax.set_ylabel("$/bbl")
    np.atleast_1d(axes)[0].legend(["Actual", "Forecast"], loc="upper right", ncol=2)
    note = " (shaded: holdout)" if has_holdout else ""
    fig.suptitle(f"{LABELS[tgt]}: {h}-month-ahead forecasts vs actual{note}", x=0.02, ha="left")
    fig.savefig(path); plt.close(fig)


def plot_original_vs_naive(table, path):
    """Each submitted model next to the naive forecast that matches its forecasting task."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    orig_colors = {"VARMAX": "#2a78d6", "XGBoost": "#eda100", "LSTM": "#e87ba4"}
    for ax, tgt in zip(axes, TARGETS):
        s = table[table.target == tgt].set_index("model").loc[list(orig_colors)]
        x = np.arange(len(s))
        b1 = ax.bar(x - 0.2, s["RMSE"], 0.38, color=[orig_colors[m] for m in s.index], label="Submitted model")
        b2 = ax.bar(x + 0.2, s["naive RMSE"], 0.38, color=NAIVE, label="Matching naive forecast")
        ax.bar_label(b1, fmt="%.2f", color=INK, fontsize=9, padding=2)
        ax.bar_label(b2, fmt="%.2f", color=INK, fontsize=9, padding=2)
        ax.set_xticks(x, [f"{m}\nvs {'static' if s.loc[m, 'naive'].startswith('static') else 'one-step'} naive"
                          for m in s.index])
        ax.grid(axis="x", visible=False)
        ax.set_title(LABELS[tgt])
    axes[0].set_ylabel("Test RMSE ($/bbl), 2016-01 to 2025-04")
    axes[1].legend(loc="upper right")
    fig.suptitle("Submitted thesis results vs the naive forecast for the same task", x=0.02, ha="left")
    fig.savefig(path); plt.close(fig)


def plot_shap(fc_xgb, tgt, path):
    import shap
    model, X = fc_xgb.attrs["models"][tgt], fc_xgb.attrs["X"]
    values = shap.TreeExplainer(model)(X)
    shap.plots.beeswarm(values, max_display=12, show=False)
    plt.title(f"XGBoost ({LABELS[tgt]}, h=1), final-year model: top features by SHAP value", loc="left")
    plt.savefig(path); plt.close("all")


# --------------------------------------------------------------------------- #

def original_vs_naive(raw):
    """Submitted thesis metrics next to the naive forecast that matches each model's task.

    - XGBoost and LSTM forecast one month ahead, so the matching benchmark is the
      one-step naive forecast (last month's price).
    - VARMAX forecast 1 to 112 months ahead from Dec 2015 (a conditional scenario path),
      so the matching benchmark is the static naive forecast: the Dec-2015 price held flat.
    Scored on exactly the months the submitted models were evaluated on.
    """
    months = pd.read_csv(ROOT / "tables" / "forecast_varmax.csv", index_col=0, parse_dates=True).index
    p = raw[TARGETS]
    actual = p.loc[months]
    naive = {"one-step naive": p.shift(1).loc[months],
             "static naive (Dec-2015 price held flat)": pd.DataFrame(
                 {t: p.loc["2015-12-01", t] for t in TARGETS}, index=months)}
    match = {"VARMAX": "static naive (Dec-2015 price held flat)", "XGBoost": "one-step naive", "LSTM": "one-step naive"}
    rows = []
    for tgt, fname in [("brent", "evaluation_results_brent.csv"), ("wti", "evaluation_results_wti.csv")]:
        shipped = pd.read_csv(ROOT / "tables" / fname, index_col=0)
        for m, r in shipped.iterrows():
            e = actual[tgt] - naive[match[m]][tgt]
            rows.append({"target": tgt, "model": m, "RMSE": r.RMSE, "MAE": r.MAE, "MAPE %": r.MAPE,
                         "naive": match[m], "naive RMSE": np.sqrt((e**2).mean()), "naive MAE": e.abs().mean(),
                         "naive MAPE %": (e.abs() / actual[tgt]).mean() * 100})
    out = pd.DataFrame(rows)
    out["model beats matching naive (RMSE)"] = out["RMSE"] < out["naive RMSE"]
    return out


def score_window(name, bounds, prices, by_h, out):
    a, b = bounds
    actual = prices.loc[a:b, TARGETS]
    parts = []
    for h, fcs in by_h.items():
        sliced = {m: f.loc[a:b] for m, f in fcs.items()}
        parts.append(score(actual, sliced, h).assign(horizon=h))
    scores = add_holm(pd.concat(parts, ignore_index=True)).assign(window=name, start=a, end=b)
    scores.to_csv(out / f"tables/scores_{name}.csv", index=False)
    return scores


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DEFAULT_XLSX))
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 6])
    ap.add_argument("--out", default=str(ROOT / "improved" / "results"))
    ap.add_argument("--quick", action="store_true", help="h=1, default hyperparameters, 1 LSTM seed, 30 epochs")
    ap.add_argument("--replot", action="store_true", help="redraw figures from saved forecasts (no refitting)")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)

    prices = load_prices(args.data)
    wins = windows(prices)
    print(f"Sample {prices.index[0]:%Y-%m} to {prices.index[-1]:%Y-%m}; train to {TRAIN_END}; "
          + "; ".join(f"{k} {a} to {b}" for k, (a, b) in wins.items()))

    # Diagnostics on the training window only.
    train = prices.loc[:TRAIN_END]
    diagnostics.stationarity(train).to_csv(out / "tables/stationarity_adf_kpss.csv", index=False)
    diagnostics.johansen(train).to_csv(out / "tables/johansen_brent_wti.csv", index=False)
    diagnostics.var_lag_selection(train).to_csv(out / "tables/var_lag_selection.csv")
    diagnostics.var_residuals(train).to_csv(out / "tables/var_residual_tests.csv", index=False)

    ovn = original_vs_naive(load_raw(args.data))
    ovn.to_csv(out / "tables/original_results_vs_matching_naive.csv", index=False)
    plot_original_vs_naive(ovn, out / "plots/original_vs_matching_naive.png")

    actual_all = prices.loc[TEST_START:, TARGETS]
    if args.replot:
        long = pd.read_csv(out / "tables/forecasts_all_horizons.csv", parse_dates=["date"])
        by_h = {h: {m: g.pivot(index="date", columns="target", values="forecast")[TARGETS].asfreq("MS")
                    for m, g in gh.groupby("model", sort=False)}
                for h, gh in long.groupby("horizon")}
        scores = {w: pd.read_csv(out / f"tables/scores_{w}.csv") for w in wins}
        make_plots(actual_all, by_h, scores, out)
        return

    # Hyperparameters: chosen on the validation window (data up to 2015-12 only).
    if args.quick:
        args.horizons = [1]
        chosen = {"XGBoost": {1: XGB_DEFAULT}, "LSTM": {1: LSTM_DEFAULT}}
        lstm_kwargs = dict(seeds=(0,), max_epochs=30)
    else:
        t0 = time.time()
        table, chosen = tune(prices, args.horizons)
        table.assign(params=table.params.astype(str)).to_csv(out / "tables/tuning_validation.csv", index=False)
        (out / "tables/tuning_selected.json").write_text(json.dumps(
            {"validation": [VAL_START, VAL_END], **{m: {str(h): p for h, p in c.items()} for m, c in chosen.items()}},
            indent=1))
        print(f"  tuning done in {time.time() - t0:.0f}s: {chosen}")
        lstm_kwargs = {}

    by_h, vecm_info = {}, []
    for h in args.horizons:
        origins = origins_for(prices, h)
        models = dict(MODELS)
        models["XGBoost"] = partial(run_xgboost, params=chosen["XGBoost"][h])
        models["LSTM"] = partial(run_lstm, **chosen["LSTM"][h], **lstm_kwargs)
        forecasts = {}
        for name, fn in models.items():
            t0 = time.time()
            forecasts[name] = fn(prices, origins, h)
            print(f"  h={h:<2} {name:<12} {time.time() - t0:6.1f}s", flush=True)
        by_h[h] = forecasts
        v = forecasts["VECM"].attrs["vecm"]
        vecm_info.append(pd.DataFrame({"origin": v["origins"], "rank": v["ranks"],
                                       "horizon": h, "k_ar_diff": v["k_ar_diff"]}))
        if h == 1:
            for tgt in TARGETS:
                plot_shap(forecasts["XGBoost"], tgt, out / f"plots/shap_xgboost_h1_{tgt}.png")
            pd.DataFrame(forecasts["LSTM"].attrs["histories"]).to_json(out / "tables/lstm_training_histories.json")

    pd.concat(vecm_info).to_csv(out / "tables/vecm_rank_path.csv", index=False)
    long = pd.concat(
        pd.DataFrame(f.to_numpy(), index=f.index.rename("date"), columns=f.columns).reset_index()  # drop attrs
          .melt("date", var_name="target", value_name="forecast")
         .assign(model=m, horizon=h)
        for h, fcs in by_h.items() for m, f in fcs.items())
    long[["date", "horizon", "model", "target", "forecast"]].to_csv(out / "tables/forecasts_all_horizons.csv", index=False)

    scores = {w: score_window(w, b, prices, by_h, out) for w, b in wins.items()}
    make_plots(actual_all, by_h, scores, out)

    cols = ["rel. RMSE vs RW", "DM p (two-sided)", "DM p Holm (two-sided)", "CW stat", "CW p (one-sided)",
            "CW p Holm (one-sided)"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        for w, s in scores.items():
            print(f"\n{w} window {wins[w][0]} to {wins[w][1]}")
            print(s.set_index(["horizon", "target", "model"])[cols])


def period_table(actual_all, forecasts):
    """Descriptive period scores, labelled by which evaluation window(s) the months fall in."""
    periods = score_by_period(actual_all, forecasts)
    bounds = {name: (a, b) for name, (a, b) in PERIODS.items()}
    last = actual_all.index[-1]

    def which(name):
        a, b = pd.Timestamp(bounds[name][0]), min(pd.Timestamp(bounds[name][1]), last)
        if b < pd.Timestamp(HOLDOUT_START):
            return "test"
        return "holdout" if a >= pd.Timestamp(HOLDOUT_START) else "test + holdout"
    periods["window"] = periods.period.map(which)
    return periods


def make_plots(actual_all, by_h, scores, out):
    """Every figure except SHAP (which needs the fitted models), plus the period tables."""
    for h, forecasts in by_h.items():
        for tgt in TARGETS:
            plot_forecasts(actual_all, forecasts, h, tgt, out / f"plots/forecasts_h{h}_{tgt}.png")
        period_table(actual_all, forecasts).to_csv(out / f"tables/period_scores_h{h}.csv", index=False)
    if 1 in by_h:
        plot_rel_rmse_by_period(score_by_period(actual_all, by_h[1]), 1, out / "plots/rel_rmse_by_period_h1.png")
    for w, s in scores.items():
        a, b = s["start"].iloc[0], s["end"].iloc[0]
        n = len(actual_all.loc[a:b])
        plot_rel_rmse_by_horizon(s, f"{w.capitalize()} window {a} to {b} ({n} target months): accuracy vs no-change",
                                 out / f"plots/rel_rmse_by_horizon_{w}.png")


if __name__ == "__main__":
    main()
