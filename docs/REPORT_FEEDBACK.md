# Report feedback (`Report.pdf`)

> **Scope.** This reviews the 23-page `Report.pdf` summary, not the full thesis
> (.docx, examined 2 August 2025), which is not in this repository and has its own
> reference list. The reference checks below apply to `Report.pdf` only. The thesis
> itself is historical work and is not being revised; see the README for how its
> claims relate to the code.

Section-by-section notes, most important first within each group. Methodology
problems in the code are covered in [METHODOLOGY_REVIEW.md](METHODOLOGY_REVIEW.md);
this file is about the write-up itself.

## A. Must fix: claims that are wrong or not supported

1. **Section 5.2 Data design.** It lists IMF macroeconomic variables and the
   Caldara & Iacoviello GPR index as sources. Neither is used. Remove both,
   or add them to the models. The GPR index is free to download and would be
   a genuine extension.
2. **Section 5.2 / 5.3 Period.** "1960–2024" and "Training: Jan 1960 – Dec 2015"
   are wrong. The model sample starts in 1982-01 in the original code (1984-01
   in the improved one), and the test set ends in **Apr 2025**, not Dec 2024.
3. **Objective 2, "integrate macroeconomic indicators".** Only commodity prices
   are used. Either rename the objective to "related commodity prices" or add
   macro data.
4. **Section 7.1.** Winsorization and Min-Max scaling are described as
   preprocessing, but no model uses the winsorized or scaled data.
5. **Section 10 Outcomes.** "LSTM ... particularly useful during times of
   volatility and regime change" isn't tested anywhere. The improved pipeline
   tests it by period, and no model reliably beats no-change in any period.
6. **Section 10, Breusch-Pagan paragraph.** The test regresses on random numbers
   and gives a different answer every run (see METHODOLOGY_REVIEW 2.4). Delete
   the paragraph and the argument built on it. Report ARCH-LM on model
   residuals instead: it shows strong heteroskedasticity.
7. **Section 6.3.** "STL decomposition confirmed seasonality". Measured seasonal
   strength is about 0.3, which is weak. Say that the seasonality is weak, and
   that this is why no seasonal terms were used.
8. **Table 4 (Durbin-Watson).** DW was run on raw price levels, not on residuals,
   so the table doesn't show what its caption claims. Replace it with Ljung-Box
   on model residuals.
9. **Tables 6-7.** The LSTM numbers can't be reproduced because no seed was set.
   Add a random-walk row, or better, replace the tables with the improved
   results.

## B. References

I checked each reference against publisher and RePEc records:

| As cited | Problem | Fix |
|---|---|---|
| Apergis & Payne (2014), *Energy Economics* 45, 307–312, "The role of coal in the global energy market..." | **I couldn't find this paper.** Apergis & Payne's 2014 *Energy Economics* paper is vol. 42, pp. 226–232, on renewable energy in Central America | Remove it, or replace it with a verified coal–oil source such as Mohammadi (2011) below |
| Wang, Wu & Yang (2013), "Oil and gold price relations: A time-varying copula approach", *Economic Modelling* 32 | **Wrong title and journal.** Wang, Wu & Yang (2013) is "Oil price shocks and stock market activities", *Journal of Comparative Economics* 41(4), 1220–1239 | Find a real oil–gold source you have read, and cite it correctly |
| Zhang & Cheng (2021), "Linkages between biofuel and fossil fuel prices...", *Energy Economics* 96, 105137 | **I couldn't find this paper** | Replace it with a biofuel–oil linkage source you have read |
| Mohammadi (2011), *Energy Economics* 33(3), 506–512 | **Wrong journal.** It's *Applied Economics* 43(2), 129–137 | Correct the entry |
| Yu, Wang & Lai (2008) | Real, but it's an EMD neural-network ensemble paper. It's cited for LSTM memory cells and for copper, and supports neither | Cite Hochreiter & Schmidhuber (1997) for LSTM |
| Kilian & Murphy (2012), Chen & Guestrin (2016) | Cited in the text but **missing from the reference list** | Add them: Kilian & Murphy, *JEEA* 10(5), 1166–1188; Chen & Guestrin, *Proc. KDD 2016*, 785–794 |
| Baumeister & Kilian (2016), Kilian (2009) | In the reference list but **never cited** in the text | Cite them where relevant (they suit the intro and the literature review) |
| "Doctor metal" (copper) | The usual phrase is "Dr. Copper" | Wording |

An examiner who finds one reference that can't be traced will check all of
them. Only cite papers you have actually opened.

**References worth adding** (all real and widely cited):

- Alquist, R., Kilian, L. & Vigfusson, R. J. (2013). Forecasting the price of oil. *Handbook of Economic Forecasting* 2A, 427–507. Establishes the no-change benchmark.
- Baumeister, C. & Kilian, L. (2012). Real-time forecasts of the real price of crude oil. *Journal of Business & Economic Statistics* 30(2), 326–336.
- Diebold, F. X. & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics* 13(3), 253–263.
- Harvey, D., Leybourne, S. & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting* 13(2), 281–291.
- Hochreiter, S. & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation* 9(8), 1735–1780.
- Johansen, S. (1991). Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models. *Econometrica* 59(6), 1551–1580.
- Lundberg, S. M. & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*.
- Cleveland, R. B., Cleveland, W. S., McRae, J. E. & Terpenning, I. (1990). STL: A seasonal-trend decomposition procedure based on loess. *Journal of Official Statistics* 6(1), 3–73.
- Wang, X., Smith, K. & Hyndman, R. (2006). Characteristic-based clustering for time series data. *Data Mining and Knowledge Discovery* 13(3), 335–364. The source of the seasonal-strength measure.

## C. Structure (what's missing)

Most MSc marking schemes expect the following, and the report currently has none of it:

- **Abstract**
- **Literature review.** This is currently one paragraph inside Section 4. It
  should cover econometric vs ML oil forecasting, the random-walk benchmark
  result, and the gap your study fills.
- **Limitations** and **Future work**
- **Conclusion** that answers RQ1 and RQ2 directly
- Section numbering jumps from 10 to 12, and there is no Section 11.
- Figures have no numbers or captions, except Figures 1 and 2. Every figure
  needs a caption and a mention in the text. Pages 5–8 and 11–17 are figures
  with no discussion.
- The CRISP-DM framework is mentioned (5.1) but never used to organise anything.
- The report has no author name, student ID, supervisor, date or word count.
  The manual still has "[Your Name or Team]" placeholders.

## D. Wording (examples)

- "policy-enduced" → "policy-induced"
- "V ARMAXis", "LSTMneuralnetworkscanmodel..." are missing spaces from LaTeX
  export. Check the whole PDF.
- "guarantees that the models are insensitive to heteroskedastic behavior". A
  volatility plot can't guarantee that. Say "motivates checking for
  heteroskedasticity".
- "ensured data completeness and purity". Say what was done ("forward-filled
  N gaps") instead.
- "This hybrid option". The study compares three separate models; nothing is
  hybridised.

## E. How to reframe the findings (recommended)

The honest result is also a defensible one, and it matches the literature:

> At horizons of 1–6 months, neither econometric nor machine-learning models
> reliably beat the no-change forecast for Brent or WTI (Diebold-Mariano
> tests are significant at 5%). Relative performance changes with the market
> regime. Every model gains during the COVID-19 collapse and rebound, with
> univariate AR and the VECM gaining most. XGBoost deteriorates sharply in
> the 2022 energy crisis and at longer horizons. A 10-commodity VAR is no
> better than a univariate AR, so related commodity prices don't improve
> point forecasts at these horizons.

This answers both research questions with evidence, which is what an
examiner needs to see. The claim that "LSTM is best" doesn't survive a
benchmark check.
