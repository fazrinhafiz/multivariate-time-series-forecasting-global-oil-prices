# Question bank: thesis and 2026 re-audit

> Originally written in September 2026 as viva preparation, although the viva had
> already taken place (the thesis was examined on 2 August 2025). Section A has
> since been updated to the current re-audit results
> (`improved/results/tables/scores_test.csv`, test window 2016-01 to 2025-04).
> Section B is about the original thesis as submitted.

The questions an examiner is most likely to ask about this project, roughly
from most to least likely, each with a suggested answer. The answers assume
you present the **improved** results (see `improved/results/`). If you have to
defend the submitted version as it stands, section B covers that.

All numbers in section A come from the regenerated re-audit outputs.

---

## A. Questions on the (improved) study

**1. What did you benchmark against, and why?**
> The random walk, or no-change forecast: next month's price equals this
> month's. It's the standard benchmark in the oil literature (Alquist, Kilian &
> Vigfusson, 2013) because monthly oil prices are close to a martingale. Any
> model that claims to be useful has to beat it. I report every model's RMSE
> relative to the random walk and test it with Clark-West, the right test when
> the benchmark is nested in the model, with Diebold-Mariano as a secondary
> check. Both are Holm-corrected across the 30 comparisons.

**2. So which model is best?**
> None shows statistically significant predictive content over no-change after
> correcting for 30 comparisons. At 1, 3 and 6 months, relative RMSEs sit
> between 0.98 and 1.05 for everything except XGBoost (1.05-1.41). The strongest
> raw result is the LSTM for WTI at h = 1: Clark-West p = 0.006 and 60%
> directional accuracy, but p = 0.18 after Holm correction.
> In practical terms, the gains available at these horizons are a few percent
> at most, and they aren't stable across periods. That's consistent with
> Alquist et al. (2013) and Baumeister & Kilian (2012), who find VARs and
> other models hard-pressed to beat no-change at short horizons.

**3. Then what's the contribution?**
> A like-for-like comparison of econometric, tree-based and deep-learning models
> under one strictly out-of-sample protocol, on a test window with two major
> shocks (COVID-19 and the 2022 energy crisis). The comparison is tested for
> significance and broken down by period. The negative result is the
> contribution: adding nine related commodity prices and non-linear models
> didn't beat simple benchmarks. I also show how a mismatched protocol can
> manufacture a "winner", which was the problem in my first version.

**4. Why is your LSTM not better than a random walk, when many papers claim it is?**
> Many of those papers compare against weak baselines, predict price levels
> (where copying last month's value already gives a low RMSE), or tune on the
> test set. My LSTM predicts returns, is re-trained only on data available at
> each origin, uses early stopping on a chronological validation split, and
> averages 3 seeds. Its window and size were chosen on a 2010-2015 validation
> period, which picked the smallest candidate. Under those conditions there
> isn't much signal to learn. BIC selects zero lags for the VAR, which says the
> same thing.

**5. How did you prevent look-ahead bias?**
> Every feature at origin *t* uses data up to *t* only. Scalers, lag orders and
> the VECM rank are fitted on data up to each origin; hyperparameters on data up
> to 2015 only. Automated tests multiply every observation after the forecast
> origin by random factors and check that no forecast changes, for every
> pipeline at h = 1, 3 and 6 over three blocks of origins, and for the tuning
> step. A deliberately leaky model fails them. They prove the forecasts don't
> depend on the values of later data for the cases tested; they can't cover
> every possible origin.

**6. Why log-returns rather than prices?**
> Prices are non-stationary: ADF doesn't reject a unit root and KPSS rejects
> stationarity. Log-returns pass both tests. Trees also can't forecast outside
> their training range, which matters for prices after 2022.

**7. Brent and WTI are cointegrated, so why not only a VECM?**
> I include a VECM. The Johansen trace test rejects rank 0 (43.98 vs 15.49) but
> not rank 1, so there's one cointegrating vector: the Brent-WTI spread
> reverts to its mean. The rank is re-tested at every forecast origin (it's 1
> every time), and the test and model use the same deterministic terms. The
> VECM is among the stronger models during COVID, but it doesn't beat
> no-change overall.

**8. What does the period breakdown show? (RQ2)**
> It's descriptive only: the windows were chosen after seeing the data, and
> nothing is tested. During COVID (2020-21), the simple AR and the VECM were
> 8-9% better than no-change. In 2016-19 and 2022-25 they were within about 2%
> of it, and XGBoost was up to 20% worse in 2022-25. So no model is
> consistently better, and I can't claim any model handles shocks best.

**9. Is there heteroskedasticity?**
> Yes. ARCH-LM on the VAR residuals rejects homoskedasticity for both series
> (p ≈ 0.0015). That's volatility clustering. It doesn't bias point forecasts,
> but it does mean prediction intervals should be time-varying (e.g. GARCH),
> which is future work.

**10. What does SHAP tell you?**
> For XGBoost at h = 1, the largest contributions come from the current month's
> WTI and copper returns, but the "51 other features" bar is as large as any
> single feature. No single commodity carries much signal. That fits the
> forecast results. SHAP explains the model, not the market, so I don't read
> it causally.

**11. Why re-train the ML models yearly, not monthly?**
> Cost. That's 112 origins × 3 horizons × 3 seeds for the LSTM. Yearly
> re-training is common in the literature and is still strictly
> out-of-sample. The econometric models are re-estimated every month.

**12. Why is the sample 1984 onwards?**
> It's the first month in which all ten series exist (South African coal
> starts then). Starting earlier would need imputation, and back-filling would
> invent prices.

**13. What would you do with more time?**
> - Real-time vintage data and the *real* (CPI-deflated) oil price, as in
>   Baumeister & Kilian (2012).
> - Oil-specific predictors: inventories, the Kilian (2009) global activity
>   index, futures spreads and the GPR index.
> - Density forecasts (quantile regression, GARCH) instead of point forecasts.
> - Forecast combinations.
> - Longer horizons (12-24 months), where the literature finds VARs can help.

**14. Limitations?**
> - The test window has 112 months and only two big shocks, so the tests have
>   limited power; the holdout has 16 months, and an earlier version of the
>   re-audit had previously evaluated part of it.
> - It's a single data source, with monthly averages rather than end-of-month
>   prices.
> - Small hyperparameter grids, chosen on 2010-2015 validation data; the LSTM
>   choice for h = 1 changed between two data vintages that differ only by rounding.
> - Point forecasts only.

---

## B. If you have to defend the submitted version

Be upfront. Examiners respond better to "I found this problem myself and
here's what it changes" than to a defence that falls apart under questioning.

**"Did you compare against a naive benchmark?"**
> Not in the submitted report, and that's its main weakness. When I did,
> the one-step no-change forecast had lower RMSE than the two one-step models
> (Brent 5.93 vs the LSTM's 9.14 and XGBoost's 16.15). VARMAX solved a different,
> multi-step task: against the matching static naive forecast (Dec-2015 price held
> flat, RMSE 35.65) it does better, but only because it was given the actual future
> prices of the other commodities. I re-ran the study under one protocol with proper
> benchmarks, and the conclusion changes: no model shows significant predictive
> content over no-change.

**"Why did the LSTM do best?"**
> Because it was the only model given last month's oil price. VARMAX forecast
> up to 112 months ahead from December 2015, and XGBoost had no oil price
> history. The comparison wasn't like-for-like.

**"Your VARMAX uses future values of the exogenous variables."**
> Yes, `exog_test` holds the actual future prices, which wouldn't be known in
> practice. In the improved version all commodities are endogenous.

**"Your Breusch-Pagan test..."**
> It regressed the residuals on randomly generated regressors, so it wasn't a
> valid test, and it gives a different answer every run. I've replaced it with
> ARCH-LM on the model residuals, which shows clear heteroskedasticity.

**"The LSTM results can't be reproduced."**
> Correct, there was no seed. Re-running gives a Brent RMSE of about 7.7
> instead of 9.1. The improved version fixes the seeds, uses deterministic ops
> and averages 3 seeds.

**"Where are the IMF and GPR data you list?"**
> They were planned but not implemented. The data section should only list
> World Bank commodity prices.

---

## C. Quick definitions to have ready

- **RMSE vs MAE.** RMSE penalises large errors more. With oil's fat tails, the
  2020 and 2022 shocks dominate RMSE.
- **MAPE.** Undefined near zero prices, and inflated when prices are low
  (e.g. WTI in April 2020, a monthly average of about $17).
- **Diebold-Mariano.** Tests whether the mean loss differential is zero. It
  uses HAC variance with h−1 autocovariances; the HLN correction is for small
  samples. Against a nested benchmark it's undersized, hence Clark-West.
- **Clark-West.** Adjusts the MSPE difference for the noise of estimating a
  model that nests the benchmark. Against a zero-change forecast it amounts to
  testing whether the forecast is positively correlated with the actual change.
  So a significant result means "carries information", not "lower RMSE".
- **Holm correction.** Step-down adjustment for many tests; controls the chance
  of any false positive across the 30 comparisons.
- **ADF vs KPSS.** Opposite null hypotheses (unit root vs stationary). Using
  both avoids over-reading a single test.
- **Johansen trace test.** A sequence of tests on the cointegration rank.
- **Expanding window.** Each origin trains on all data up to that point.
  A rolling window would drop old data instead.
- **Direction accuracy.** The share of months with the correct sign of change.
  50% is a coin toss. The best current value is the LSTM's 60% for WTI at h = 1,
  which is not significant after correction (Clark-West Holm p = 0.18).
- **Why XGBoost gets worse with horizon.** A direct h-step tree model has
  fewer effective training samples and learns pre-2016 patterns. Trees can
  also only output averages of training targets, so they can't anticipate a
  run-up like 2021-22's.
