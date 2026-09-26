# Viva preparation (archived)

> **Archived.** Written in September 2026 for a viva that had in fact already taken
> place (the thesis was examined on 2 August 2025). Kept for its question list. Where
> it cites re-audit numbers, the current figures and tests (Clark-West, Holm
> correction, tuned hyperparameters) are in `improved/results*/tables/scores_test.csv`
> and the README; they supersede the numbers below.

The questions an examiner is most likely to ask about this project, roughly
from most to least likely, each with a suggested answer. The answers assume
you present the **improved** results (see `improved/results/`). If you have to
defend the submitted version as it stands, section B covers that.

Numbers in brackets referred to an earlier results file that has since been regenerated.

---

## A. Questions on the (improved) study

**1. What did you benchmark against, and why?**
> The random walk, or no-change forecast: next month's price equals this
> month's. It's the standard benchmark in the oil literature (Alquist, Kilian &
> Vigfusson, 2013) because monthly oil prices are close to a martingale. Any
> model that claims to be useful has to beat it. I report every model's RMSE
> relative to the random walk, and test the difference with Diebold-Mariano.

**2. So which model is best?**
> None is reliably better than no-change. At 1, 3 and 6 months, relative
> RMSEs sit between 0.96 and 1.03 for everything except XGBoost, and none of
> the 30 Diebold-Mariano tests rejects equal accuracy at 5% (the lowest p is
> 0.08). The LSTM comes closest at h = 1: 4% lower RMSE for Brent (p = 0.13),
> and 61% directional accuracy for WTI.
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
> averages 3 seeds. Under those conditions there isn't much signal to learn.
> BIC selects zero lags for the VAR, which says the same thing.

**5. How did you prevent look-ahead bias?**
> Three ways. Every feature at origin *t* uses data up to *t* only. Scalers and
> lag orders are fitted on training windows only. And there's an automated
> test that multiplies every observation after the last forecast origin by
> random factors and checks that no forecast changes. I checked the test
> itself by feeding it a deliberately leaky model, which it catches.

**6. Why log-returns rather than prices?**
> Prices are non-stationary: ADF doesn't reject a unit root and KPSS rejects
> stationarity. Log-returns pass both tests. Trees also can't forecast outside
> their training range, which matters for prices after 2022.

**7. Brent and WTI are cointegrated, so why not only a VECM?**
> I include a VECM. The Johansen trace test rejects rank 0 (41.1 vs 15.5) but
> not rank 1, so there's one cointegrating vector: the Brent-WTI spread
> reverts to its mean. The VECM is among the stronger models during COVID, when
> the spread blew out, but it doesn't beat no-change overall.

**8. What does the period breakdown show? (RQ2)**
> During COVID (2020-21), every model beat no-change, by up to 9%. The
> collapse and rebound had momentum, and the simple AR and the VECM did best
> (relative RMSE ≈ 0.91-0.92). In the 2022-25 energy crisis, the LSTM
> (0.95 for Brent) and the VAR did slightly better, while XGBoost was 12-13%
> worse. In the calm 2016-19 period, nothing helped. So the answer to RQ2 is
> that the best model depends on the type of shock, and no model is
> consistently better.

**9. Is there heteroskedasticity?**
> Yes. ARCH-LM on the VAR residuals rejects homoskedasticity for both series
> (p ≈ 0.001). That's volatility clustering. It doesn't bias point forecasts,
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
> - The test window has 112 months and only two big shocks, so DM tests have
>   limited power.
> - It's a single data source, with monthly averages rather than end-of-month
>   prices.
> - No hyperparameter search for the ML models; I used conservative defaults
>   to avoid tuning on the test set.
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
> benchmarks, and the conclusion changes: no model beats no-change.

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
  samples.
- **ADF vs KPSS.** Opposite null hypotheses (unit root vs stationary). Using
  both avoids over-reading a single test.
- **Johansen trace test.** A sequence of tests on the cointegration rank.
- **Expanding window.** Each origin trains on all data up to that point.
  A rolling window would drop old data instead.
- **Direction accuracy.** The share of months with the correct sign of change.
  50% is a coin toss. An earlier ad hoc binomial test (not in the repo code) gave
  p = 0.03 for the LSTM's 61% on WTI at h = 1; it ignored multiple comparisons and
  is superseded by the Clark-West results.
- **Why XGBoost gets worse with horizon.** A direct h-step tree model has
  fewer effective training samples and learns pre-2016 patterns. Trees can
  also only output averages of training targets, so they can't anticipate a
  run-up like 2021-22's.
