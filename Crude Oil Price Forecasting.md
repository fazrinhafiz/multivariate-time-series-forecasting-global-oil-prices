# Crude Oil Price Forecasting Project Manual

This manual provides a step-by-step guide to understanding and running the code used in a research project for forecasting Brent and WTI crude oil prices using three models: VARMAX, XGBoost, and LSTM. The project includes data preprocessing, model training, evaluation, and interpretability.


## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Environment Setup](#2-environment-setup)
3. [Data Preparation](#3-data-preparation)
4. [Descriptive Analysis](#4-descriptive-analysis)
5. [Data Preprocessing](#5-data-preprocessing)
6. [Model Development](#6-model-development)
7. [Forecast Comparison](#7-forecast-comparison)
8. [Evaluation & Diagnostics](#8-evaluation--diagnostics)
9. [Visual Outputs](#9-visual-outputs)
10. [Folder Structure](#10-folder-structure)

---

## 1. Project Overview

This project forecasts monthly crude oil prices (Brent and WTI) using:

* **VARMAX**: A time series model considering multiple variables.
* **XGBoost**: A machine learning regression model using lagged features.
* **LSTM**: A deep learning model capable of capturing sequential dependencies.

The dataset contains historical prices for commodities (oil, gas, metals) from the World Bank.

---

## 2. Environment Setup

Ensure you have Python 3.8+ installed and the following packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow openpyxl statsmodels shap
```

---

## 3. Data Preparation

### Source

* Load `CMO-Historical-Data-Monthly.xlsx` (World Bank Commodity Prices).
* Read "Monthly Prices" sheet, skipping metadata rows.

### Processing

* Extract columns for Brent, WTI, coal, gas, copper, gold, palm oil.
* Parse date and set as index.
* Convert all prices to numeric and handle missing values.

---

## 4. Descriptive Analysis

Generate insights:

* Summary statistics (mean, std, skewness, kurtosis).
* Line plot showing price trends with historical events.
* STL decomposition (seasonality, trend) for Brent.
* Correlation heatmaps (Pearson and Spearman).
* Boxplots to detect outliers.
* Rolling standard deviation (volatility).

Output files:

* `tables/summary_stats.csv`
* `plots/long_term_trends.png`

---

## 5. Data Preprocessing

### Winsorization

* Cap extreme 1% values to reduce outlier influence.

### Stationarity & Normality Tests

* ADF test: stationarity check.
* Shapiro test: normality check.
* Durbin-Watson test: residual autocorrelation.

### Scaling

* Use `MinMaxScaler` to scale features for LSTM/XGBoost.

Output files:

* `tables/adf_results.csv`
* `tables/shapiro_results.csv`
* `tables/durbin_watson_stats.csv`

---

## 6. Model Development

### A. VARMAX

* Use first-differenced values to ensure stationarity.
* Brent and WTI as endogenous; all other features exogenous.
* Train on data until 2015, forecast 2016 onward.
* Reconstruct forecast to original scale.

### B. XGBoost

* Create lag-1 features for all predictors.
* Train separate models for Brent and WTI.

### C. LSTM

* Use 15 time-step sequences.
* Train separate LSTM networks for Brent and WTI.
* Use 85/15 validation split.

Output:

* Forecast CSVs and loss plots for LSTM.

---

## 7. Forecast Comparison

Visualize actual vs. predicted:

* Plot all 3 model forecasts against actual prices.
* Highlight economic disruptions (e.g., COVID-19).

Plots:

* `plots/forecast_comparison_brent_all_models.png`
* `plots/forecast_comparison_wti_all_models.png`

---

## 8. Evaluation & Diagnostics

### Error Metrics

* RMSE, MSE, MAE, MAPE calculated for each model.

### Residual Analysis

* Breusch-Pagan test to check for heteroskedasticity.
* ACF/PACF plots for VARMAX residuals.

### Model Interpretability

* SHAP analysis for XGBoost to explain feature influence.

### Year-wise Stability

* RMSE over time to detect volatile periods.

Output files:

* `tables/evaluation_results_brent.csv`
* `tables/evaluation_results_wti.csv`
* `tables/breusch_pagan_combined.csv`
* `plots/acf_pacf_var_residuals.png`
* `plots/shap_xgboost-(brent|wti).png`
* `plots/yearly_rmse_xgb_brent.png`

---

## 9. Visual Outputs

All plots are saved under the `plots/` directory:

* Forecast plots
* SHAP summaries
* ACF/PACF diagnostics
* Training/validation loss (LSTM)
* Bar plots of metrics comparison

---

## 10. Folder Structure

```
project_root/
├── CMO-Historical-Data-Monthly.xlsx
├── main_forecasting.py  # or .ipynb script
├── plots/
│   └── *.png  # All visuals
├── tables/
│   └── *.csv  # All evaluation/exported tables
```

---

## 📌 Final Notes

* This project is modular: any model can be updated or replaced.
* Suitable for financial forecasting, energy economics, or machine learning case studies.
* Ensure reproducibility by fixing random seeds where needed.

If you encounter issues, confirm directory creation (e.g., `os.makedirs("plots", exist_ok=True)`) before saving.

---

**Author:** \[Your Name or Team]
**Date:** \[Insert Date]

