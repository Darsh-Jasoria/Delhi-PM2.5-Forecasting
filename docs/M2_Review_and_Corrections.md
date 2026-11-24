# Milestone 2 review: why strong models gave low accuracy

Question raised by our faculty: *the models used in Milestone 2 are powerful, so why is the accuracy still low?*

**Answer in one line:** the limit was the information given to the models, not the models themselves. Milestone 2 asked ARIMA, SARIMA and Holt-Winters to forecast tomorrow from one daily average per day. A stronger model fed the same single series reaches the same accuracy, and the one-week range needs weather forecasts that no pollution history can supply.

## Evidence (all numbers are in `results/`, four test seasons pooled, RMSE in µg/m³)

| Experiment | Next-day RMSE | Week-ahead RMSE |
|---|---|---|
| Naive: repeat yesterday | 24.41 | 41.45 |
| ARIMA (Milestone 2) | 22.90 | 32.66 |
| SARIMA (Milestone 2) | 22.35 | 30.79 |
| XGBoost given the same history only (no covariates) | 24.27 | 34.60 |
| XGBoost without intraday statistics | 21.61 | 30.75 |
| XGBoost without any weather | 20.14 | 33.87 |
| XGBoost with intraday statistics, no weather forecast | 20.49 | 34.03 |
| XGBoost with everything (intraday + archived weather forecast) | 19.17 | 30.27 |
| Ensemble (XGBoost, LightGBM, Random Forest, LSTM, GRU) | **18.36** | **27.54** |

A powerful model without better inputs does not beat ARIMA (24.27 against 22.90). The gain appears only when the inputs change.

## Issues found in Milestone 2 and what was done

| # | Issue | Where it shows | Fix in Milestone 3 |
|---|---|---|---|
| 1 | Univariate modelling: CO (r ≈ 0.69; 0.685 exactly) and NO₂ (r = 0.56) were found to track PM2.5 but were never used as inputs | M2 slide 7, `milestone2_analysis.py` models use `train` only | 79 features: co-pollutants, observed weather, intraday statistics, weather forecasts |
| 2 | Daily mean hides how the day ended; Delhi's PM2.5 swings from about 52 to 91 µg/m³ within a day | Top SHAP features are last-hour and evening PM2.5 | Intraday statistics built from the hourly data |
| 3 | No weather forecast: wind and mixing decide the multi-day range, and pollution history cannot predict them | Without the forecast, XGBoost's week-ahead RMSE is 34.03 instead of 30.27 | Archived ECMWF forecasts, used only if issued before the forecast origin; ensemble week-ahead 27.54 |
| 4 | Weekly seasonality used (SARIMA m = 7, Holt-Winters m = 7) although the lag-7 autocorrelation of the log-differenced series is about 0 and M2's own STL found a weak weekly cycle; Holt-Winters failed Ljung-Box (p ≈ 0) | M2 slides 6, 9, 10 | Weekly terms dropped; yearly terms kept |
| 5 | Data artefact: the first 4 daily values are one back-filled constant (43.6 µg/m³); the data were called "gap-free" although the first 77 hours and 817 CO₂ days were missing | `data/delhi_daily_air_quality.csv` rows 1 to 4; M2 slide 3 | Rows removed; series starts 5 Aug 2022 (1,538 days); gaps documented |
| 6 | A single 90-day test window (July to October), the easiest season; winter errors are about twice as large | Ensemble RMSE 10.5 in Jul-Oct against 19.8 in Oct-Jan | Four rolling folds, one per season; Diebold-Mariano significance tests |
| 7 | No decision-level metric | Only RMSE, MAE, MAPE | CPCB category accuracy, alert F1 and extreme-tail bias added |
| 8 | CO₂ listed as a feature but available for only 725 of 1,542 days and weakly correlated (r = 0.32) | M2 slides 3, 7 | Excluded from models, reason stated |

Checked and found **not** to be an error: comparing AIC across d = 0 and d = 1 in the order search. statsmodels' state-space likelihood uses the same 1,452 observations for both, so the comparison is acceptable.

## Milestone 2 numbers reproduced

Milestone 3 re-runs the Milestone 2 classical models on the same hold-out and obtains the identical results (ARIMA RMSE 14.95, naive 15.76, SARIMA 15.14, Holt-Winters 17.52), so every comparison between the two milestones is like for like.
