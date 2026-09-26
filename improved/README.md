# 2026 re-audit pipeline

This is the re-audit of the MSc thesis's forecasting evaluation. It is **not** the thesis:
the thesis code (`code_full.py`) and its outputs (`plots/`, `tables/`) are kept
unchanged at the repo root as historical work. Why the re-audit exists is set out in
[`docs/METHODOLOGY_REVIEW.md`](../docs/METHODOLOGY_REVIEW.md).

## Run it

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python improved/run_all.py --quick      # smoke test, ~1 min: h=1, default hyperparameters
python improved/run_all.py              # re-audit on the thesis data (tuning + all models, ~15 min on CPU)
python improved/run_all.py --data data/CMO-Historical-Data-Monthly_latest.xlsx --out improved/results_latest
python improved/causality.py            # Granger causality, impulse responses, variance decomposition
python -m pytest improved/tests         # leakage tests and known-answer checks (~5 min)
```

## Design

**Question.** At the end of month *t*, using only data published by then, what will
the Brent and WTI monthly-average prices be in month *t+h*, for h = 1, 3 and 6?

**Sample.** From 1984-01, the first month in which all ten series exist, so nothing
is imputed. Models work on log-returns (the VECM on log levels).

**Three separate windows, never pooled in one test:**

| Window | Months | Used for |
|---|---|---|
| Training history | 1984-01 onwards, expanding | Estimation at every origin |
| **Validation** | 2010-01 to 2015-12 | Choosing XGBoost and LSTM hyperparameters only (`tuning.py`) |
| **Test** | 2016-01 to 2025-04 | The months the thesis was scored on; main re-audit results |
| **Holdout** | 2025-05 onwards | Data released after the thesis data ended; scored separately |

**Forecasting.** Rolling origins with an expanding window. AR, VAR and VECM are
re-estimated at every origin and forecast h steps iteratively. XGBoost and the LSTM
forecast the h-month change directly and are re-trained every January.

| Model | Specification |
|---|---|
| Random walk | Forecast = price at the origin (the benchmark) |
| AR | Univariate AR(p) on log-returns; p chosen by BIC on data up to the first origin |
| VAR | VAR(p) on log-returns of all 10 commodities, all endogenous; p chosen by BIC (floored at 1, because BIC picks 0) |
| VECM | Log Brent and log WTI. Unrestricted constant, the same deterministic case as the Johansen test used to pick the rank. Lag by BIC. **Rank re-tested at every origin** (trace test, 5%); rank 0 allowed. The restricted-constant case is checked in `diagnostics.py` |
| XGBoost | 6 monthly lags of all 10 returns, Brent-WTI spread, 12-month Brent volatility. Hyperparameters chosen per horizon on the validation window |
| LSTM | Sequences of the 10 returns plus the spread; two outputs; early stopping; 3-seed average. Window length and units chosen per horizon on the validation window |

**Tests against the random walk** (`evaluate.py`):
- **Clark-West** (one-sided): the correct test when the benchmark is nested in the
  model (every model here reduces to no-change when its coefficients are zero). It
  asks whether the forecast carries information about the direction and size of the
  change, not whether its RMSE is lower.
- **Diebold-Mariano** (two-sided, HLN-corrected): equal squared-error accuracy.
  Undersized against a nested benchmark, so it's reported as a secondary check.
- Both are **Holm-corrected** across the 30 comparisons (5 models x 2 targets x 3
  horizons) in each window.

**Period breakdown.** Relative RMSE in 2016-19, 2020-21, 2022-25 and 2026. These windows
were chosen after seeing the data, so they are descriptive only; nothing is tested.

## Files

| File | Purpose |
|---|---|
| `data.py` | Loads the Pink Sheet; balanced sample; log-returns |
| `models.py` | The six forecasting pipelines behind one interface |
| `tuning.py` | Validation-window hyperparameter selection for XGBoost and the LSTM |
| `evaluate.py` | Metrics, Clark-West, Diebold-Mariano, Holm correction, period breakdown |
| `diagnostics.py` | ADF/KPSS, Johansen (both deterministic cases), VAR lag selection, Ljung-Box, ARCH-LM |
| `run_all.py` | Runs everything; writes `results*/tables` and `results*/plots` |
| `causality.py` | Granger causality (Holm-corrected), generalized impulse responses and variance decomposition |
| `tests/test_no_lookahead.py` | Leakage tests: every pipeline x h = 1, 3, 6 x three origin blocks, plus the tuning step |
| `tests/test_evaluate.py` | Known-answer checks for Clark-West, Diebold-Mariano, Holm, the Johansen implementation and the VECM rank path |
| `tests/test_causality.py` | Impulse-response and variance-decomposition identities; detection of a planted lead |

## Outputs (`results/` = thesis data, `results_latest/` = newer data)

| File | Contents |
|---|---|
| `tables/scores_test.csv` | Test window: metrics, DM and CW with raw and Holm p-values |
| `tables/scores_holdout.csv` | Holdout window (only in `results_latest/`) |
| `tables/tuning_validation.csv`, `tuning_selected.json` | Every candidate's validation score and the chosen settings |
| `tables/vecm_rank_path.csv` | Johansen rank chosen at every origin |
| `tables/original_results_vs_matching_naive.csv` | Thesis metrics next to the naive forecast that matches each model's task |
| `tables/period_scores_h*.csv` | Descriptive period breakdown |
| `tables/forecasts_all_horizons.csv` | Every forecast (long format) |
| diagnostics, causality tables | See file names |
