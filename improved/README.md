# Improved forecasting pipeline

A rebuild of the thesis pipeline that fixes the issues in
[`docs/METHODOLOGY_REVIEW.md`](../docs/METHODOLOGY_REVIEW.md). The original
`code_full.py` is kept unchanged at the repo root.

## Run it

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python improved/run_all.py --quick      # about 1 minute: h=1, one LSTM seed
python improved/run_all.py              # full run: h=1,3,6, 3-seed LSTM (about 15-25 min on CPU)
python improved/causality.py            # Granger causality, impulse responses, variance decomposition
python -m pytest improved/tests         # leak tests and causality checks
```

To use a newer World Bank Pink Sheet, download `CMO-Historical-Data-Monthly.xlsx`
from the World Bank commodity markets page and pass it with
`--data path/to/file.xlsx`. The sample end and test period extend automatically.

## Design

**Question.** At the end of month *t*, using only data published by then, what
will Brent and WTI cost in month *t+h*? Here h = 1, 3 and 6 months.

**Sample.** 1984-01 (the first month in which all ten series exist, so nothing
is imputed) to the last month in the file. The models are trained on data up
to 2015-12 and tested on every month from 2016-01 on.

**Evaluation.** Expanding window, strictly out-of-sample. AR, VAR and VECM are
re-estimated at every origin. XGBoost and the LSTM are re-trained every January
and then roll forward with new inputs each month.

| Model | Specification |
|---|---|
| Random walk | Next price = today's price (the benchmark) |
| AR | Univariate AR(p) on log-returns; p chosen by BIC |
| VAR | VAR(p) on log-returns of all 10 commodities; p chosen by BIC. Everything is endogenous, so no future values are fed in |
| VECM | Log Brent and log WTI with one cointegrating relation (Johansen) |
| XGBoost | Six monthly lags of all 10 returns, the Brent-WTI spread and 12-month Brent volatility. Forecasts the h-month log change directly |
| LSTM | 12-month windows of the 10 returns plus the spread. Two outputs (Brent, WTI), early stopping, and the average of 3 seeds |

**Metrics.** RMSE, MAE, MAPE, RMSE relative to the random walk, directional
accuracy, and a Diebold-Mariano test (with the HLN correction) against the
random walk. Scores are also split by period: 2016-19, 2020-21 (COVID) and
2022-25 (energy crisis).

## Files

| File | Purpose |
|---|---|
| `data.py` | Loads the Pink Sheet and builds the balanced sample and log-returns |
| `models.py` | The six models under one interface |
| `evaluate.py` | Metrics, the Diebold-Mariano test and the period breakdown |
| `diagnostics.py` | ADF/KPSS, Johansen, VAR lag selection, Ljung-Box and ARCH-LM |
| `run_all.py` | Runs everything and writes `results/tables` and `results/plots` |
| `causality.py` | Granger causality (both directions, Holm-corrected), generalized impulse responses with bootstrap bands, generalized variance decomposition |
| `tests/test_no_lookahead.py` | Changes all data after the forecast origin and checks that no forecast moves |
| `tests/test_causality.py` | Checks the impulse-response and variance-decomposition maths, and that a planted lead is detected |
