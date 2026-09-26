# Execution log: every section and what it runs

This file covers three things, in order:

1. **Part A**: every section of the original `code_full.py`: what it executes,
   its inputs, its outputs, and what the re-run produced.
2. **Part B**: every stage of the improved pipeline in `improved/`.
3. **Part C**: every run carried out while reviewing the project (commands,
   runtimes, outcomes).

Line numbers refer to `code_full.py`. The notebook `code_full.ipynb` holds the
same code (only two plot titles differ).

---

## Part A: original pipeline (`code_full.py`)

Run command: `python code_full.py`. Re-run environment: Python 3.12.x, macOS
arm64, CPU only, package versions pinned in `requirements.txt`. The original
run used Python 3.9 on Linux with an NVIDIA RTX 3060.

### A1. Imports and setup (l.16-53)

| Executes | Output |
|---|---|
| Imports pandas, numpy, matplotlib, seaborn, statsmodels (STL, ADF, Johansen, Durbin-Watson, VAR, VARMAX, Breusch-Pagan, ACF/PACF), scikit-learn, scipy, xgboost, tensorflow/keras, shap | none |
| `os.makedirs("plots")`, `os.makedirs("tables")` | the two output folders |

`coint_johansen` is imported but never used.

### A2. Data loading (l.59-99)

| Step | Executes | Result |
|---|---|---|
| Read workbook | `pd.read_excel(..., sheet_name='Monthly Prices', skiprows=4, header=[0,1])` | 784 monthly rows, 1960-01 to 2025-04 |
| Select columns | 10 series: Brent, WTI, Australian coal, South African coal, US gas, European gas, Japan LNG, palm oil, copper, gold | `df_selected` |
| Flatten headers | Joins name and unit, e.g. `Crude oil, Brent ($/bbl)` | |
| Parse dates | `format='%YM%m'` (e.g. `1960M01`), set as index | |
| Coerce numerics | `pd.to_numeric(errors='coerce')` on object columns | ".." placeholders become NaN |

### A3. Descriptive analysis (l.112-215)

| Step | Executes | Output file |
|---|---|---|
| Summary statistics | `describe()` plus skewness and kurtosis, on the full 1960-2025 data | `tables/summary_stats.csv` |
| Long-term trends | Line plot of all 10 series with 1973, 1979, 2008 and 2020 markers | `plots/long_term_trends.png` |
| STL, Brent | `STL(period=12)` on Brent 1960-2025 | `plots/stl_decomposition_brent.png` |
| STL, WTI | `STL(period=12)` on WTI 1982-2025 | `plots/stl_decomposition_wti.png` |
| Pearson heatmap | `corr('pearson')` on price levels | `plots/correlation_heatmap_pearson.png` |
| Spearman heatmap | `corr('spearman')` on price levels | `plots/correlation_heatmap_Spearman.png` |
| Volatility | 12-month rolling std of Brent and WTI | `plots/rolling_volatility.png` |
| Boxplots (grid) | One boxplot per series, 5x2 layout | `plots/outlier_boxplots.png` |
| Boxplot (combined) | seaborn boxplot of all series | `plots/boxplots.png` |

`plots/correlation_heatmap.png` is left over from a 27 May run. No current
code writes it.

### A4. Cleaning and alignment (l.221-229)

| Step | Executes | Result |
|---|---|---|
| Trim | Keep rows from the first WTI value (1982-01) | 520 rows, 1982-01 to 2025-04 |
| Forward fill | `.ffill()` | No effect (no gaps inside series) |
| Backward fill | `.bfill()` | **Copies South African coal's Jan-1984 price into 24 earlier months** (1982-83) |

### A5. Pre-modelling tests (l.245-292)

| Step | Executes | Output file |
|---|---|---|
| Winsorize | `winsorize(limits=[0.01, 0.01])` per column into `df_winsor` | Used only by the tests below and lag selection; **no model uses it** |
| ADF | `adfuller()` p-value per series, on levels | `tables/adf_results.csv` |
| Shapiro-Wilk | `shapiro()` p-value per series, on levels | `tables/shapiro_results.csv` |
| Durbin-Watson | `durbin_watson(df_winsor)`, on levels, not residuals | `tables/durbin_watson_stats.csv` |
| VAR lag selection | `VAR(df_winsor).select_order(12)`, on levels | Printed. AIC picks 12, BIC 1, HQIC 3 |
| Min-Max scaling | `MinMaxScaler` fitted on all data into `df_scaled` | **Never used** |

Printed lag-selection table from the re-run:

```
       AIC         BIC         FPE         HQIC
0        58.36       58.44   2.219e+25       58.39
1        34.35      35.26*   8.262e+14       34.71
3        33.15       35.73   2.488e+14      34.16*
12      30.86*       40.94  2.781e+13*       34.81
Optimal lag (AIC): 12
```

### A6. VARMAX (l.303-386)

| Step | Executes | Result |
|---|---|---|
| Difference | `df_final.diff().dropna()` | 519 rows |
| Endogenous | Brent and WTI first differences | |
| Exogenous | The other 8 commodities' first differences | |
| Split | Train to 2015-12 (407 months), test from 2016-01 (112 months) | |
| Fit | `VARMAX(endog_train, exog=exog_train, order=(12, 0)).fit()` with L-BFGS | 69 parameters; **about 13 min on CPU** |
| Forecast | `results.forecast(steps=112, exog=exog_test)`, a 1- to 112-step forecast from Dec 2015 fed the **actual** future exogenous values | |
| Rebuild prices | Dec-2015 price plus cumulative sum of forecast differences | `tables/forecast_varmax.csv` |
| Plots | Actual vs forecast | `plots/varmax_forecast_brent.png`, `plots/varmax_forecast_wti.png` |
| Comparison table | Actual and forecast side by side | `tables/comparison_varmax.csv` |

### A7. XGBoost (l.398-473)

| Step | Executes | Result |
|---|---|---|
| Lag features | The 8 non-oil series shifted by 1 month; the oil targets are not lagged | 519 rows |
| Split | Train to 2015-12 (407), test from 2016-01 (112) | |
| Fit | `XGBRegressor(objective='reg:squarederror')` with library defaults, one model per target | |
| Predict | Price level in month t from other commodities at t-1 | |
| Plots | Actual vs predicted | `plots/xgboost_forecast_brent.png`, `plots/xgboost_forecast_wti.png` |

### A8. LSTM (l.484-636)

| Step | Executes | Result |
|---|---|---|
| Scale inputs | `MinMaxScaler` on Brent and WTI, **fitted on the full sample including the test period** | |
| Sequences | 15-month windows of scaled Brent and WTI; target is the raw price next month | 505 sequences |
| Split | Train 393 (fit 334, validate 59) / test 112 | |
| Model | `LSTM(64) → Dense(32, relu) → Dense(1)`, Adam, MSE, 50 epochs, batch 32, **no seed, no early stopping** | one model per target |
| Predict | One step ahead on the test sequences | |
| Plots | Loss curves; actual vs predicted | `plots/lstm_train_val_loss_{brent,wti}.png`, `plots/lstm_forecast_{brent,wti}.png` |

### A9. Forecast comparison (l.645-695)

| Executes | Output |
|---|---|
| Overlays the VARMAX, XGBoost and LSTM forecasts on actual prices, with a COVID marker | `plots/forecast_comparison_brent_all_models.png`, `plots/forecast_comparison_wti_all_models.png` |

### A10. Evaluation (l.708-751)

| Executes | Output |
|---|---|
| RMSE, MSE, MAE and MAPE for each model and target | `tables/evaluation_results_brent.csv`, `tables/evaluation_results_wti.csv` |

### A11. Diagnostics (l.764-942)

| Step | Executes | Output file |
|---|---|---|
| Metrics again | `evaluate_forecast` is defined a second time; the Brent table is recomputed with VARMAX renamed "VAR" | `tables/eval_metrics_brent.csv` |
| Breusch-Pagan | VARMAX forecast errors regressed on **two columns of random normal noise** | `tables/breusch_pagan_combined.csv` |
| SHAP | `shap.Explainer` on each XGBoost model, test features | `plots/shap_xgboost (brent).png`, `plots/shap_xgboost (wti).png` |
| Year-wise RMSE | XGBoost, Brent only, 2016-2024 | `tables/yearly_rmse_xgb_brent.csv`, `plots/yearly_rmse_xgb_brent.png` |
| ACF/PACF | On the 1- to 112-step VARMAX forecast errors | `plots/acf_pacf_var_residuals.png` |
| Combined metrics chart | Bar charts of RMSE, MAE, MAPE and MSE | `plots/combined_error_metrics_brent_wti.png` |

### A12. Re-run results vs the submitted outputs

| Output | Submitted | Re-run | Match |
|---|---|---|---|
| Summary stats, ADF, Shapiro, Durbin-Watson | | | Exact (max difference 2e-15) |
| XGBoost RMSE, Brent / WTI | 16.15 / 11.29 | 16.15 / 11.29 | Exact |
| XGBoost yearly RMSE, Brent 2016→2024 | 6.50 … 25.56 | 6.50 … 25.56 | Exact |
| VARMAX RMSE, Brent / WTI | 18.54 / 15.67 | 18.54 / 15.67 | Forecasts within $0.008 |
| **LSTM RMSE, Brent / WTI** | **9.14 / 7.07** | **7.67 / 6.69** | **No** (no seed) |
| **Breusch-Pagan p, Brent / WTI** | **0.639 / 0.076** | **0.498 / 0.536** | **No** (random regressors) |
| Notebook's stored VARMAX log | 25 parameters, i.e. order (1,0) | Code uses order (12,0), 69 parameters | Notebook is stale |

---

## Part B: improved pipeline (`improved/`)

Run command: `python improved/run_all.py [--data FILE] [--out DIR] [--horizons 1 3 6] [--quick | --replot]`.

### B1. Data (`data.py`)

| Step | Executes |
|---|---|
| `load_raw` | Reads the same sheet, selects the 10 series, renames them (`brent`, `wti`, `coal_au`, …), sets a monthly index |
| `load_prices` | Keeps only the months where all 10 series exist (1984-01 to the file's end). Raises an error if any gap remains, so nothing is imputed |
| `log_returns` | `log(P).diff()` |

### B2. Diagnostics (`diagnostics.py`, training window only, to 2015-12)

| Test | Executes | Output file |
|---|---|---|
| Stationarity | ADF and KPSS on log levels and log-returns of all 10 series | `tables/stationarity_adf_kpss.csv` |
| Cointegration | Johansen trace test, log Brent and log WTI | `tables/johansen_brent_wti.csv` |
| Lag selection | `VAR(returns).select_order(12)` with AIC, BIC, HQIC and FPE | `tables/var_lag_selection.csv` |
| Residual checks | Ljung-Box(12) and ARCH-LM(12) on in-sample VAR residuals | `tables/var_residual_tests.csv` |

### B3. Original vs naive (`run_all.original_vs_naive`)

This scores the no-change forecast on exactly the 112 months your submitted models
were evaluated on, and puts it beside your submitted metrics. Outputs:
`tables/original_results_vs_random_walk.csv` and
`plots/original_vs_random_walk.png`.

### B4. Models (`models.py`), run for h = 1, 3 and 6

For each horizon h, the forecast origins are every month t such that t + h
falls in the test period.

| Model | Executes at each origin |
|---|---|
| Random walk | Forecast = price at t |
| AR | `AutoReg` on each oil series' returns, lags chosen once by BIC, re-fitted every month, iterated h steps |
| VAR | `VAR` on all 10 return series, lag chosen once by BIC (min 1), re-fitted every month, iterated h steps |
| VECM | Johansen rank and lag chosen once, `VECM(deterministic='ci')` re-fitted every month, `predict(h)` |
| XGBoost | Features: 6 lags of all 10 returns, the Brent-WTI spread and 12-month Brent volatility. The target is the h-month log change. Re-trained every January on rows whose target is already observed, one model per target |
| LSTM | 12-month windows of 10 returns plus the spread, scaled on training data only. `LSTM(32) → Dropout(0.2) → Dense(16) → Dense(2)` with early stopping (patience 20, 15% chronological validation), 3 seeds averaged, deterministic ops, re-trained every January |

### B5. Evaluation (`evaluate.py`)

| Step | Executes | Output file |
|---|---|---|
| Metrics | RMSE, MAE, MAPE, RMSE relative to the random walk, directional accuracy | `tables/scores_all_horizons.csv` |
| Significance | Diebold-Mariano test against the random walk, with HLN correction and h−1 autocovariances | same file |
| By period | Relative RMSE for 2016-19, 2020-21, 2022-25 and 2026 (only periods with data) | `tables/period_scores_h{1,3,6}.csv` |
| Forecasts | Every forecast in long format (date, horizon, model, target, forecast) | `tables/forecasts_all_horizons.csv` |
| LSTM training | Loss curves for each year and seed | `tables/lstm_training_histories.json` |

### B6. Plots (`run_all.py`)

| Plot | File |
|---|---|
| Relative RMSE by horizon | `plots/rel_rmse_by_horizon.png` |
| Relative RMSE by period (h=1) | `plots/rel_rmse_by_period_h1.png` |
| Forecasts vs actual, one panel per model | `plots/forecasts_h{1,3,6}_{brent,wti}.png` |
| SHAP beeswarm, XGBoost h=1 | `plots/shap_xgboost_h1_{brent,wti}.png` |
| Original results vs naive | `plots/original_vs_random_walk.png` |

### B7. Tests (`tests/test_no_lookahead.py`)

| Test | Checks |
|---|---|
| `test_forecasts_ignore_future_data` ×6 (one per model) | Multiplies every observation after the last origin by random factors (0.2-5×); forecasts must be identical |
| `test_features_at_t_use_only_data_up_to_t` | XGBoost features up to the cutoff are unchanged by the scrambled future data |
| `test_forecast_index_is_origin_plus_horizon` | Forecasts are dated origin + h |
| `test_balanced_sample_has_no_imputed_values` | The sample starts in 1984-01 and has no NaNs |

---

## Part C: runs carried out during the review

| # | Run | Command | Runtime | Outcome |
|---|---|---|---|---|
| 1 | Environment | `uv venv --python 3.12 .venv` + pinned install | ~2 min | OK. The original `requirements.txt` had a local conda path for `matplotlib-inline`, since fixed |
| 2 | Original pipeline re-run | `MPLBACKEND=Agg python code_full.py` (in a scratch copy) | ~15 min (VARMAX ~13 min) | Exit 0. Results in A12 |
| 3 | VARMAX order check | `VARMAX(order=(1,0))` on the same data | ~20 s | Doesn't match the submitted forecasts (max difference $23), so the tables came from order (12,0) |
| 4 | Naive benchmark | Random walk on the 112 test months | <1 s | Brent 5.93, WTI 5.77 |
| 5 | STL seasonal strength | `STL(log price, period=12)`, F_s = 1 − Var(R)/Var(S+R) | <1 s | Brent 0.31, WTI 0.30 (weak) |
| 6 | Improved smoke test | `python improved/run_all.py --quick` | ~1 min | Exit 0 |
| 7 | Look-ahead tests | `python -m pytest improved/tests` | 19-80 s | 9 passed |
| 8 | Mutation check | A deliberately leaky model against the look-ahead test | <1 s | Caught |
| 9 | Improved full run, thesis data | `python improved/run_all.py` | ~7 min wall (LSTM ~2 min per horizon) | Exit 0. Results in `improved/results/`, summary in METHODOLOGY_REVIEW §6 |
| 10 | Directional accuracy test | Binomial test against 50% | <1 s | LSTM h=1: Brent p = 0.07, WTI p = 0.03 |
| 11 | Redraw | `python improved/run_all.py --replot` | ~10 s | Figures redrawn from saved forecasts |
| 12 | Newer data download | World Bank Commodity Markets page → `data/CMO-Historical-Data-Monthly_latest.xlsx` (587 KB, published 2026-09-02) | ~2 s | Data to 2026-08. Past values changed only by rounding (≤ $0.50) |
| 13 | Improved full run, latest data | `python improved/run_all.py --data data/CMO-Historical-Data-Monthly_latest.xlsx --out improved/results_latest` | ~5 min of model fitting (longer wall time while other jobs shared the CPU) | Exit 0. Test period to 2026-08 (128 months). Results in `improved/results_latest/`, summary in METHODOLOGY_REVIEW §7 |
| 14 | Run ledger page | Built from the result tables and published as an artifact | <1 min | Interactive version of this file |
| 15 | Causality analysis, thesis data | `python improved/causality.py` | ~5 s (500 bootstrap replications) | No commodity leads oil after correction; oil leads LNG and gas |
| 16 | Causality analysis, latest data | `python improved/causality.py --data data/CMO-Historical-Data-Monthly_latest.xlsx --out improved/results_latest` | ~5 s | Same conclusions |
| 17 | Tests | `python -m pytest improved/tests` | ~8 s | 13 passed (9 leak tests + 4 causality checks) |
