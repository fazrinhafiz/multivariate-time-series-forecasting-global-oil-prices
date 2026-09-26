# Methodology review of the submitted pipeline

This review covers `code_full.py` / `code_full.ipynb` and `Report.pdf` as submitted
(the first commit in this repo). It is written the way an external examiner
would read the work. Line numbers refer to `code_full.py`.

Every claim below was checked by running the code, not by reading alone. The
re-run used Python 3.12 with the pinned package versions in `requirements.txt`.

---

## 1. Reproduction result

| Output | Re-run matches submission? | Note |
|---|---|---|
| Summary stats, ADF, Shapiro, Durbin-Watson | Exactly | |
| XGBoost forecasts and metrics | Exactly | XGBoost is deterministic by default |
| VARMAX forecasts and metrics | Within $0.01 | Optimiser noise only |
| **LSTM metrics** | **No** | Brent RMSE 7.67 on re-run vs 9.14 reported (WTI 6.69 vs 7.07). No random seed is set anywhere |
| **Breusch-Pagan table** | **No** | WTI p-value 0.536 on re-run vs 0.076 reported. The test uses random numbers (see 2.4) |
| Notebook saved outputs | **Inconsistent** | The notebook's stored VARMAX log shows 25 parameters, i.e. order (1,0). The code and the tables use order (12,0). The notebook was edited after it was last run |

**Consequence:** the headline LSTM numbers in Tables 6-7 cannot be reproduced
by anyone, including you.

---

## 2. Critical issues (these change the conclusions)

### 2.1 No naive benchmark, and the naive benchmark beats every model

The standard benchmark in oil price forecasting is the **random walk
(no-change) forecast**: next month's price equals this month's (Alquist, Kilian
& Vigfusson, 2013). On the same 112 test months it scores:

| RMSE ($/bbl) | VARMAX | XGBoost | LSTM | **Random walk** |
|---|---|---|---|---|
| Brent | 18.54 | 16.15 | 9.14 | **5.93** |
| WTI | 15.67 | 11.29 | 7.07 | **5.77** |

All three submitted models are worse than doing nothing. An examiner who knows
the oil forecasting literature will check this first.
Plot: `improved/results/plots/original_vs_random_walk.png`.

### 2.2 The three models answer three different questions

The comparison in Section 10 is not like-for-like:

| Model | What it actually forecasts | Information it gets |
|---|---|---|
| VARMAX (l.321-325) | Price 1 to 112 months ahead, all from Dec 2015 | The **actual future values** of coal, gas, copper, gold and palm oil (`exog=exog_test`). These are unknown at forecast time, so it's an oracle |
| XGBoost (l.404-414) | This month's oil price | Last month's *other* commodity prices only. **No oil price history at all** |
| LSTM (l.484-530) | Next month's price | The actual Brent and WTI prices for the previous 15 months, refreshed every month |

The LSTM is the only model that sees last month's oil price, so it can land near
the random walk just by copying it. Its "win" comes from getting the easiest
task, not from better modelling. VARMAX has to forecast up to nine years ahead
with no feedback, which is a much harder task.

### 2.3 Results are presented as significant without any significance test

With 112 test months and very noisy errors, differences of $1-2 in RMSE are
usually not statistically distinguishable. The standard tool is the
Diebold-Mariano test (Diebold & Mariano, 1995), with the Harvey, Leybourne &
Newbold (1997) small-sample correction.

### 2.4 The Breusch-Pagan test regresses on random numbers

```python
dummy_X = sm.add_constant(np.random.normal(size=(len(residuals), 2)))   # l.777
```

The test asks whether the error variance depends on two columns of random
noise, so it can never detect heteroskedasticity. It also gives a different
answer on every run. Section 10 builds an argument on it ("marginal
heteroskedasticity in WTI (p = 0.077) ... partially the reason why ... VARMAX
underperform[s]"). Both the result and the argument need to go.

A proper ARCH-LM test on the VAR residuals shows strong heteroskedasticity for
both series (p ≈ 0.001). This is the textbook volatility clustering of oil
prices, and the opposite of the report's conclusion for Brent.

### 2.5 The report describes things the code does not do

| Report says | Code does |
|---|---|
| Sources include IMF macro variables and the Caldara & Iacoviello GPR index (5.2) | Only World Bank commodity prices are loaded |
| Objective 2: "integrate macroeconomic indicators" | No macroeconomic indicators are used, only commodity prices |
| Period 1960-2024; training Jan 1960 - Dec 2015 (5.2, 5.3) | Data is trimmed to the first WTI observation (l.222-225), so the sample starts **1982-01**. It ends **2025-04** |
| Testing Jan 2016 - Dec 2024 | The test set runs to **Apr 2025** (112 months) |
| Winsorization at 1% (7.1) | `df_winsor` (l.245) is **never used by any model**. All models use the un-winsorized `df_final` |
| Min-Max scaling for LSTM and XGBoost (7.1) | `df_scaled` (l.274) is never used. XGBoost gets raw prices; the LSTM re-scales its own inputs |
| "Optimal lag = 12" chosen by AIC | Lag selection ran on *levels* of the winsorized data (l.265), but the model uses differences of the non-winsorized data. The comment on l.320 says "p=1, q=0" while the code says `order=(12, 0)` |

---

## 3. Major issues

1. **Back-filled data** (l.229). South African coal starts in 1984-01, so `.bfill()`
   copies its Jan 1984 price into all 24 months of 1982-83. That's invented
   data. The fix is to start the sample in 1984-01, when every series exists.
2. **Durbin-Watson on price levels** (l.257). DW is a test on regression
   *residuals*. On raw trending prices it's always about 0 (Table 4 shows
   0.001-0.024) and only says that prices are persistent. Replace it with
   Ljung-Box on model residuals.
3. **Stationarity was only tested in levels.** The report never shows that
   differencing made the series stationary. Use ADF and KPSS together, on both
   levels and returns.
4. **Cointegration was ignored.** `coint_johansen` is imported (l.24) but never
   used. Brent and WTI are strongly cointegrated (trace statistic 41.1 vs 5%
   critical value 15.5). Differencing both throws away that long-run link. A
   VECM is the model the econometrics literature would expect here.
5. **Look-ahead in the LSTM scaler** (l.490-491). The `MinMaxScaler` is fitted on
   the full sample, test period included, so the 2022 price spike is known during
   training. The effect is small, but examiners look for it.
6. **Levels instead of returns.** XGBoost and the LSTM predict price *levels*.
   Tree models can't predict outside the range they saw in training, and a
   levels LSTM mostly learns to copy the last value. Forecasting log-returns
   and converting them back is the standard approach.
7. **No hyperparameter validation.** XGBoost uses library defaults. The LSTM
   runs a fixed 50 epochs with no early stopping, even though it has a
   validation split.
8. **ACF/PACF of forecast errors** (l.878-893). These are errors from 1- to
   112-step-ahead forecasts, which are autocorrelated by construction. They
   say nothing about whether the model is well specified. Plot the in-sample
   residuals instead.
9. **Correlation heatmaps on price levels.** Trending series always look
   correlated (spurious correlation). Correlations of *returns* measure the
   co-movement that matters for forecasting.

## 4. Minor issues

- `evaluate_forecast` is defined twice (l.708, l.764), and `eval_metrics_brent.csv`
  duplicates `evaluation_results_brent.csv` with VARMAX renamed "VAR".
- Plot titles say "2016-2024", but the data runs to 2025-04. `compute_yearly_rmse`
  ignores 2025.
- `plots/correlation_heatmap.png` is left over from an earlier run (27 May)
  and duplicates the Pearson heatmap.
- The year-wise stability analysis covers only XGBoost on Brent. Research
  Question 2 asks which *model* does best under shocks, so every model needs it.
- STL "confirmed seasonality" (Section 6). The measured seasonal strength of
  log Brent is F_s ≈ 0.31 (WTI 0.30), which is weak. Values above about 0.64
  count as strong (Wang, Smith & Hyndman, 2006). The report should say that
  seasonality is weak.

---

## 5. What the improved pipeline changes

Code: `improved/`. How to run it: `improved/README.md`.

| Issue | Fix |
|---|---|
| No benchmark | Random walk and univariate AR are included. Every metric is also reported relative to the random walk |
| Different tasks per model | **One protocol.** Every model forecasts h = 1, 3 and 6 months ahead, from the same origins, using only data available at the origin |
| Oracle exogenous variables | The VAR treats all 10 commodities as endogenous, so it forecasts them instead of being given them |
| Static single split | Expanding window. AR, VAR and VECM are re-estimated every month; XGBoost and the LSTM are re-trained every January |
| No significance test | Diebold-Mariano with the HLN correction, against the random walk |
| Unreproducible LSTM | Fixed seeds, deterministic ops, and a 3-seed ensemble average |
| Invented data | The sample starts in 1984-01, so no imputation is needed |
| Levels | All models work on log-returns |
| Scaler leakage | Scalers are fitted on each training window only |
| Wrong diagnostics | ADF + KPSS, Johansen, VAR lag selection, Ljung-Box and ARCH-LM on residuals, all on the training window |
| Cointegration ignored | VECM added |
| RQ2 not answered | Scores are broken down by period: 2016-19 (calm), 2020-21 (COVID), 2022-25 (energy crisis) |
| Leakage risk | `improved/tests/test_no_lookahead.py` changes all data after the last forecast origin and checks that no forecast moves. A deliberately leaky model is caught |

Results: see `improved/results/` and section 6 below.

---

## 6. Results of the improved pipeline

Test period 2016-01 to 2025-04 (112 target months). DM p = two-sided
Diebold-Mariano p-value against the random walk (HLN-corrected).

**Brent** (RMSE in $/bbl; rel. = RMSE ÷ random-walk RMSE)

| Model | h=1 RMSE | h=1 rel. | h=1 DM p | h=3 rel. | h=3 DM p | h=6 rel. | h=6 DM p |
|---|---|---|---|---|---|---|---|
| Random walk | 5.93 | 1.000 | — | 1.000 | — | 1.000 | — |
| AR | 5.81 | 0.980 | 0.58 | 1.019 | 0.15 | 1.019 | 0.40 |
| VAR | 5.86 | 0.989 | 0.83 | 1.017 | 0.59 | 1.014 | 0.61 |
| VECM | 5.83 | 0.984 | 0.66 | 1.020 | 0.09 | 1.009 | 0.64 |
| XGBoost | 6.19 | 1.044 | 0.36 | 1.179 | 0.14 | 1.380 | 0.11 |
| LSTM | 5.68 | 0.958 | 0.13 | 1.014 | 0.76 | 0.970 | 0.56 |

**WTI** (RMSE in $/bbl; rel. = RMSE ÷ random-walk RMSE)

| Model | h=1 RMSE | h=1 rel. | h=1 DM p | h=3 rel. | h=3 DM p | h=6 rel. | h=6 DM p |
|---|---|---|---|---|---|---|---|
| Random walk | 5.77 | 1.000 | — | 1.000 | — | 1.000 | — |
| AR | 5.66 | 0.981 | 0.59 | 1.022 | 0.08 | 1.025 | 0.29 |
| VAR | 5.74 | 0.995 | 0.91 | 1.015 | 0.63 | 1.017 | 0.37 |
| VECM | 5.69 | 0.986 | 0.69 | 1.013 | 0.60 | 1.006 | 0.89 |
| XGBoost | 6.06 | 1.050 | 0.32 | 1.197 | 0.19 | 1.445 | 0.12 |
| LSTM | 5.62 | 0.973 | 0.25 | 1.032 | 0.50 | 1.020 | 0.59 |

**By period, h = 1** (relative RMSE vs random walk)

| Target | Model | 2016-19 (calm) | 2020-21 (COVID) | 2022-25 (energy crisis) |
|---|---|---|---|---|
| Brent | AR | 0.982 | 0.912 | 1.020 |
| Brent | VAR | 1.046 | 0.968 | 0.972 |
| Brent | VECM | 1.013 | 0.924 | 1.005 |
| Brent | XGBoost | 1.018 | 0.939 | 1.118 |
| Brent | LSTM | 1.001 | 0.931 | 0.953 |
| WTI | AR | 1.009 | 0.917 | 1.014 |
| WTI | VAR | 1.037 | 0.986 | 0.981 |
| WTI | VECM | 1.021 | 0.915 | 1.021 |
| WTI | XGBoost | 0.997 | 0.971 | 1.133 |
| WTI | LSTM | 0.989 | 0.950 | 0.984 |

### What the results say

1. **Nothing significantly beats no-change.** None of the 30 model-horizon
   comparisons rejects equal accuracy at 5%. The lowest DM p-value is 0.08.
2. **h = 1:** every model except XGBoost is 1-4% better than no-change, which
   is within noise. The LSTM gets closest (Brent 0.958, p = 0.13).
3. **h = 3 and h = 6:** the econometric models and the LSTM sit at 0.97-1.03.
   **XGBoost is 18-45% worse than no-change.** The direct multi-step tree model
   overfits pre-2016 patterns and misses the 2021-22 run-up (see
   `plots/forecasts_h6_brent.png`).
4. **By period (RQ2):** during COVID (2020-21), every model beats no-change by
   1-9%. The collapse and rebound had momentum, and univariate AR and the VECM
   do best (≈0.91-0.92). In the 2022-25 energy crisis, the LSTM and VAR are
   slightly better (0.95-0.98) and XGBoost is 12-13% worse. In the calm
   period (2016-19), nothing helps.
5. **Multivariate information adds little.** The univariate AR is as good as
   the 10-variable VAR at every horizon. That's the direct answer to RQ1.
6. **Direction:** the LSTM calls the direction of the next month's move right
   59% (Brent) and 61% (WTI) of the time. A binomial test against 50% gives
   p = 0.07 and 0.03. After allowing for the number of comparisons, that's
   suggestive at best.

These results are consistent with the oil forecasting literature: at short
horizons the no-change forecast is very hard to beat (Alquist, Kilian &
Vigfusson, 2013).
