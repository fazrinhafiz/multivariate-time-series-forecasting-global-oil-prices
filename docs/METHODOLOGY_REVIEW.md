# Methodology review of the submitted pipeline

> **Revision note.** Sections 1-5 and 8 review the original thesis code. Sections
> 6, 7 and 9 describe the 2026 re-audit; they were revised after a follow-up audit,
> and section 10 lists what changed and why. Where an earlier statement here was
> wrong or too strong, it has been corrected in place and noted in section 10.

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

### 2.1 No naive benchmark

The standard benchmark in oil price forecasting is the **no-change (naive)
forecast** (Alquist, Kilian & Vigfusson, 2013). The fair naive benchmark depends on
the task each model was given, so each submitted model is compared with the naive
forecast for *its* task, on the same 112 test months:

| RMSE ($/bbl) | Submitted model | Matching naive forecast | Model better? |
|---|---|---|---|
| Brent, XGBoost (one month ahead) | 16.15 | 5.93 (one-step naive) | No |
| Brent, LSTM (one month ahead) | 9.14 | 5.93 (one-step naive) | No |
| Brent, VARMAX (1-112 months ahead from Dec 2015) | 18.54 | 35.65 (static naive: Dec-2015 price held flat) | Yes, but it was given the actual future prices of the other commodities |
| WTI, XGBoost | 11.29 | 5.77 | No |
| WTI, LSTM | 7.07 | 5.77 | No |
| WTI, VARMAX | 15.67 | 31.97 | Yes, with the same caveat |

The two one-step models are worse than doing nothing. VARMAX beats its matching
naive forecast, but as a *scenario* model: with real-time inputs it would not have
had those commodity prices. (An earlier version of this section compared VARMAX
with the one-step naive forecast and said all three models lost to naive; that
was the wrong comparison for VARMAX's task.)
Table and plot: `improved/results/tables/original_results_vs_matching_naive.csv`,
`improved/results/plots/original_vs_matching_naive.png`.

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

## 5. What the re-audit pipeline changes

Code: `improved/`. How to run it: `improved/README.md`.

| Issue in the thesis pipeline | Re-audit design |
|---|---|
| No benchmark | No-change (naive) and univariate AR baselines. Every metric is also reported relative to no-change |
| Different tasks per model | **One protocol.** Every model forecasts h = 1, 3 and 6 months ahead from the same origins, using only data available at the origin |
| Oracle exogenous variables | The VAR treats all 10 commodities as endogenous, so it forecasts them instead of being given them |
| Static single split | Rolling origins, expanding window. AR, VAR and VECM re-estimated every month; XGBoost and the LSTM re-trained every January |
| No significance test | Clark-West (primary; the benchmark is nested) and Diebold-Mariano (secondary), both Holm-corrected across the 30 comparisons per window |
| Untuned ML models | Hyperparameters chosen on a validation window (2010-01 to 2015-12) that ends before the test |
| Unreproducible LSTM | Fixed seeds, deterministic ops, 3-seed average |
| Invented data | Sample starts in 1984-01, so nothing is imputed |
| Levels | Log-returns (VECM on log levels) |
| Scaler fitted on the full sample | Scalers fitted on each training window only |
| Invalid diagnostics | ADF + KPSS, Johansen (both deterministic cases), VAR lag selection, Ljung-Box and ARCH-LM on residuals, all on the training window |
| Cointegration ignored | VECM with the rank re-tested at every origin |
| Test and later data mixed | Test (2016-01 to 2025-04) and holdout (2025-05 onwards) scored separately |
| Leakage risk | Leakage tests for every pipeline, h = 1, 3, 6, three origin blocks, and the tuning step |

---

## 6. Re-audit results: test window 2016-01 to 2025-04 (thesis data)

Outputs: `improved/results/tables/scores_test.csv`. RMSE ÷ naive RMSE (below 1 =
better than no-change); naive RMSE at h = 1 is 5.93 (Brent) and 5.77 (WTI).

| Model | Brent h=1 | h=3 | h=6 | WTI h=1 | h=3 | h=6 |
|---|---|---|---|---|---|---|
| AR | 0.980 | 1.019 | 1.019 | 0.981 | 1.022 | 1.025 |
| VAR | 0.989 | 1.017 | 1.014 | 0.995 | 1.015 | 1.017 |
| VECM | 0.985 | 1.025 | 1.019 | 0.988 | 1.018 | 1.016 |
| XGBoost | 1.045 | 1.174 | 1.354 | 1.086 | 1.223 | 1.410 |
| LSTM | 1.003 | 1.036 | 1.045 | 0.981 | 1.009 | 1.025 |

**Clark-West p-values** (one-sided; H1: the forecast has predictive content), raw:

| Model | Brent h=1 | h=3 | h=6 | WTI h=1 | h=3 | h=6 |
|---|---|---|---|---|---|---|
| AR | 0.067 | 0.734 | 0.642 | 0.320 | 0.801 | 0.802 |
| VAR | 0.175 | 0.747 | 0.520 | 0.376 | 0.821 | 0.712 |
| VECM | 0.244 | 0.812 | 0.749 | 0.328 | 0.754 | 0.567 |
| XGBoost | 0.118 | 0.879 | 0.924 | 0.184 | 0.848 | 0.892 |
| LSTM | 0.097 | 0.580 | 0.274 | **0.006** | 0.295 | 0.182 |

After Holm correction across these 30 tests, the smallest p-value is 0.178 (LSTM,
WTI, h = 1). The only raw Diebold-Mariano p below 0.05 is the VECM for Brent at
h = 3 (p = 0.037), a model that is *worse* than no-change; its Holm p-value is 1.0.

**Tuning** (`tables/tuning_validation.csv`): at every horizon the validation window
chose the simplest candidates. For XGBoost that was depth 2 and learning rate 0.03;
for the LSTM, a 6-month window and 16 units. Every candidate was worse than
no-change on the validation window (XGBoost 1.09-1.21, LSTM 1.00-1.19). The fixed
settings used before tuning was added (LSTM window 12, 32 units) had scored 0.958 for
Brent at h = 1 on the *test* set. Reporting that setting now would be choosing by
test-set performance, so the validation choice is reported instead.

**VECM** (`tables/vecm_rank_path.csv`): lag 1 (BIC); Johansen rank 1 at all 112
origins for every horizon. The restricted-constant check also gives rank 1 (trace
43.98 vs 20.26 for rank 0; 2.66 vs 9.16 for rank 1).

**By period, h = 1** (descriptive; windows chosen after seeing the data):

| Target | Model | 2016-19 (48 months) | 2020-21 COVID (24) | 2022-25 (40) |
|---|---|---|---|---|
| Brent | AR | 0.982 | 0.912 | 1.020 |
| Brent | VAR | 1.046 | 0.968 | 0.972 |
| Brent | VECM | 1.013 | 0.925 | 1.008 |
| Brent | XGBoost | 1.026 | 0.936 | 1.118 |
| Brent | LSTM | 1.012 | 1.008 | 0.995 |
| WTI | AR | 1.009 | 0.917 | 1.014 |
| WTI | VAR | 1.037 | 0.986 | 0.981 |
| WTI | VECM | 1.023 | 0.916 | 1.023 |
| WTI | XGBoost | 0.995 | 0.989 | 1.196 |
| WTI | LSTM | 0.988 | 0.993 | 0.967 |

### What the results support

1. **No model shows significant predictive content over no-change** once the
   nested-benchmark test is used and the 30 comparisons are corrected for. This is
   consistent with the oil forecasting literature (Alquist, Kilian & Vigfusson, 2013).
2. **Point estimates are within a few percent of no-change** for AR, VAR, VECM and
   the LSTM; XGBoost is clearly worse at every horizon.
3. **Related commodities add little** (RQ1): the 10-commodity VAR is no better than
   the univariate AR.
4. **On shocks (RQ2), only a description is possible.** AR and VECM were 8-9% better
   than no-change in 2020-21 and roughly level elsewhere. Twenty-four months, chosen
   after the fact, cannot establish which model handles shocks best.

What the results do **not** support: any claim that a particular model is superior
to no-change, or to the others, at these horizons.

---

## 7. Holdout: 2025-05 to 2026-08 (newer data)

Data: `data/CMO-Historical-Data-Monthly_latest.xlsx`, published by the World Bank on
2026-09-02. Past values differ from the thesis data only by rounding (at most
$0.50). Outputs: `improved/results_latest/tables/scores_holdout.csv`.

**Status.**
- No design decision in the current pipeline uses these months. Tuning uses data
  to 2015-12; lags and ranks use data up to each origin.
- An earlier version of the re-audit (September 2026) reported a pooled 2016-2026
  window that included them, before this design was fixed. So they have been
  seen, though not used.

| Model | Brent h=1 | h=3 | h=6 | WTI h=1 | h=3 | h=6 |
|---|---|---|---|---|---|---|
| AR | 0.935 | 1.061 | 0.999 | 0.956 | 1.044 | 1.002 |
| VAR | 1.039 | 1.084 | 0.978 | 0.973 | 1.059 | 0.973 |
| VECM | 0.950 | 1.070 | 0.982 | 0.940 | 1.062 | 0.940 |
| XGBoost | 1.080 | 1.004 | 1.026 | 1.048 | 0.974 | 0.971 |
| LSTM | 0.990 | 0.974 | 0.944 | 0.985 | 0.966 | 0.963 |

Naive RMSE at h = 1: Brent 11.67, WTI 8.56. That's twice the test window's, because
of the spring-2026 spike: Brent went from $71 in February to $120 in April.

- **Nothing is significant after correction.** The lowest Clark-West Holm p-value
  is 0.88; the lowest raw value is 0.029 (VECM, WTI, h = 1).
- **Sixteen months are too few** to separate models that differ by a few percent.
- The test window re-scored on this data vintage (`results_latest/tables/scores_test.csv`)
  matches §6 closely. The one difference is the tuned LSTM for h = 1 (window 12
  instead of 6), which shows that choice is sensitive to small data revisions.

An earlier version of this section pooled 2016-2026 into one window and called one
uncorrected DM result (p = 0.048) significant. Both have been corrected; see §10.

---

## 8. Design intent from the full thesis

The full thesis (UiTM, examined 2 August 2025) explains choices that the code
alone does not. Read the critique above in that light:

| Choice | Reason given in the thesis | Revised view |
|---|---|---|
| Only multivariate models; no univariate or naive model | Scope (§1.6): univariate models were excluded because they ignore cross-variable interactions | The ranking *among the three* is correct as reported. A naive forecast is a yardstick, not a competing model: adding one stays within scope and shows whether the best model is useful |
| VARMAX given the actual exogenous prices in the test period | An interpretable econometric model with contemporaneous exogenous effects (the B term), aimed at impulse-response and scenario analysis | Valid as a **conditional (scenario) forecast**: "given what gas, coal and metals did, where would oil be?" It is not a real-time forecast, so it isn't comparable with the other two |
| XGBoost uses only last month's *other* commodities | A deliberate choice "to avoid data leakage"; tests how far related markets explain oil. The SHAP chapter is built on it | A coherent cross-market explanatory model. It answers a different question from the LSTM |
| LSTM on Brent and WTI's own 15-month history | Captures long-term dependencies in the oil series themselves | A one-step forecast, the task closest to the naive forecast |

The three models answer three different questions, so the RMSE ranking
mixes model quality with how hard each task is.

### Statements in the thesis that don't match the code or the charts

| Thesis says | Code and charts show |
|---|---|
| §4.3: in 2021-23 "XGBoost overshoots significantly, forecasting a peak of over 160 USD", while "VARMAX lags in responsiveness" | The overshoot is **VARMAX**: Brent 172.0 and WTI 157.3 in Apr 2022 (`tables/forecast_varmax.csv`). XGBoost stays around 110 or below |
| Ch. 5: the study "incorporated … exchange rates, industrial production, consumer price index (CPI), and interest rates" | Only the 10 commodity prices are used |
| §1.6: data from 2010 to 2024; §3.4: from 1960 to 2024 | The code uses 1982-01 to 2025-04 |
| §3.6: testing Jan 2016 to Dec 2024 | The test set runs to Apr 2025 |
| §3.5: winsorization and Min-Max scaling applied for all models | Neither is used by VARMAX or XGBoost; the LSTM scales only its own two inputs |
| §4.3: LSTM shows "robustness when retrained on new conditions" | The LSTM is trained once, on data to 2015, and never retrained |
| §4.3: residual ACF/PACF shows the VAR is misspecified | The plots show errors of 1- to 112-step forecasts, which are autocorrelated by construction |

---

## 9. Granger causality, impulse responses and variance decomposition

Code: `improved/causality.py`. Outputs: `improved/results{,_latest}/tables/granger_*.csv`,
`irf_*.csv`, `fevd_12m.csv` and the plots `granger_causality.png`,
`impulse_responses.png` and `variance_decomposition.png`.

These tests answer research question 1 directly: do related commodity markets
carry information about oil prices? All of them use monthly log-returns.

**Granger causality.** Each test uses a two-variable VAR with its lag chosen
by BIC, in both directions, and p-values are Holm-corrected across the 34 tests
in each sample (9 partners x 2 oil series x 2 directions, with the Brent-WTI pair
counted once per direction).

| Direction | Result (full sample, thesis data) |
|---|---|
| Related commodity → oil | **None significant after correction.** Copper is closest (raw p ≈ 0.004, Holm p ≈ 0.09) |
| All 8 commodities jointly → oil | Borderline: p = 0.04 (Brent), 0.06 (WTI); latest data 0.08; training window 0.24 |
| **Oil → related commodity** | **Strong.** Oil leads Japanese LNG (4-6 month lag, p < 0.001), US gas (2 months), European gas and South African coal |

Oil leading Japanese LNG by 4-6 months matches how LNG is priced: long-term
Japanese contracts are indexed to crude oil with a lag. Information flows
*from* oil *to* these markets, not the other way round.

**Impulse responses** (generalized, from the 10-variable VAR). These show the
cumulative change in the oil price over 12 months after a typical
(one standard deviation) monthly move in each market, with 90% bootstrap bands:

| Move in | Brent after 12 months | Share of it in the same month |
|---|---|---|
| Copper (+5.7%) | +5.9% | 43% |
| Coal, SA / AU (+6.7% / +6.6%) | +3.6% / +3.3% | about 50% |
| Palm oil (+6.9%) | +3.5% | 37% |
| Gold (+3.4%) | +2.4% | 54% |
| US gas (+13.8%) | +2.1% | 55% |
| European gas (+8.9%) | +0.1% (band includes 0) | — |
| Japanese LNG (+4.3%) | −1.0% (band includes 0) | — |

Most of the response is same-month co-movement, not a lead. Part of the
follow-through in later months reflects oil's own momentum, which the
World Bank's monthly averaging exaggerates.

**Variance decomposition** (share of 12-month forecast surprises): Brent and
WTI account for about 86% of each other's surprises. Copper is the largest
outside source at about 5%. European gas contributes 0.1-0.2%.

**What this adds to the thesis:**
- **Copper is the strongest candidate; European gas shows nothing.** The thesis's
  SHAP analysis ranked copper and European gas top for Brent. Copper is the largest
  outside co-mover and the closest thing to a leading indicator, but its lead is
  **not significant** after correction (Holm p ≈ 0.09-0.15). European gas carries
  almost no information in returns. A plausible explanation for its SHAP importance
  is that the thesis's XGBoost used price *levels*, which trend together (for example
  in 2021-22); that explanation has not been tested.
- **It explains the forecasting results.** No related market reliably leads
  oil, so adding them can't improve forecasts. That is why the 10-commodity
  VAR is no better than the single-series AR.
- **A positive finding to pair with the negative one.** Oil is the leading
  market: it predicts gas and LNG prices, not the reverse.

Caveats: Granger causality means predictive precedence, not causation. The
monthly averages can create spurious lead-lag relations. The VAR is linear. The
F-tests assume homoskedastic errors, but the residuals show strong ARCH effects
(§3 diagnostics), so the p-values are approximate; heteroskedasticity-robust tests
have not been run. The oil-indexed LNG explanation is a plausible domain
interpretation, not something the data can confirm.

---

## 10. Revision after the follow-up audit (September 2026)

A later audit of the re-audit itself found problems in the re-audit pipeline and in
this document. What was changed:

| Audit finding | Change | Where |
|---|---|---|
| Diebold-Mariano is undersized when the benchmark (no-change) is nested in the models | **Clark-West** test added as the primary test of predictive content; DM kept as a secondary accuracy test | `evaluate.py` |
| 30 DM tests per window, no multiplicity correction; one raw p = 0.048 was reported as "significant" | **Holm correction** across the 30 tests per window for both CW and DM; the p = 0.048 claim withdrawn | `evaluate.add_holm`; §6-7 |
| VECM: rank tested with an unrestricted constant but model fitted with a restricted one; rank forced to be ≥ 1 and chosen once; Johansen lag in diagnostics hard-coded at 2 | Model and test now use the same case (unrestricted constant). **Rank re-tested at every origin**, rank 0 allowed. Diagnostics use the VECM's own BIC lag, report trace and max-eigenvalue, and add a hand-written restricted-constant test (checked against statsmodels for the unrestricted case) | `models.run_vecm`, `diagnostics.johansen` |
| XGBoost and LSTM hyperparameters fixed without validation | **Validation window 2010-01 to 2015-12** (data truncated at 2015-12 before tuning); small grids; chosen per horizon | `tuning.py` |
| Leakage tests covered h = 3 and 12 origins only | 8 pipelines x h = 1, 3, 6 x three origin blocks, the tuning step, and a leaky-model control per horizon | `tests/test_no_lookahead.py` |
| The "naive beats all three thesis models" claim used the one-step naive forecast for VARMAX's multi-step task | Each thesis model compared with the naive forecast for its own task; VARMAX beats the static naive forecast (with oracle inputs) | §2.1; `run_all.original_vs_naive` |
| "Holm-corrected across 32 tests" | Corrected to 34 | §9 |
| "Copper confirmed" | Rewritten: largest co-mover, lead not significant after correction | §9 |
| Granger F-tests assume homoskedastic errors despite ARCH effects | Stated as a caveat (not re-run) | §9 |
| Test and holdout months were pooled in the 2016-2026 table | Scored separately: test 2016-01 to 2025-04, holdout 2025-05 onwards | `run_all.py`; §7 |
| `improved/results/` had been produced by an earlier code version, with an ad hoc format conversion | Both results folders deleted and regenerated from scratch by the current code | `improved/results*/` |
| Period breakdown presented as evidence about shocks | Labelled descriptive; windows chosen after seeing the data; month counts shown | `evaluate.py`; plots |
