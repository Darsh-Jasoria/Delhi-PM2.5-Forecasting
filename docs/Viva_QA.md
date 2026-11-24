# Viva preparation: likely questions and honest answers

Every number below comes from `results/`. Short answers first; details in the report.

## Time plan (the course allows 10 to 12 minutes per team, demonstration plus Q&A)

| Slides | What | Time |
|---|---|---|
| 1, 3 | Title and the one-slide summary | 1 min |
| 4, 5, 6 | Problem, dataset, Milestone 2 recap | 1.5 min |
| 7, 8 | Why Milestone 2 was inaccurate, what we changed | 1.5 min |
| 9, 10 | Method, models and tuning | 1.2 min |
| 11, 12, 13 | Results: hold-out, four seasons, effect of weather forecasts | 2 min |
| 16, 17, 18 | Week ahead, what drives the forecast, alerts and extremes | 2 min |
| 19, 20 | Limitations, conclusion | 1 min |
| 21 | Live demo of the website (see below) | 1 min |
| 22 | Team | 0.3 min |

Skip slides 2, 14, 15 and 23 (agenda, two forecast plots, references) if time is short; they are in the deck for questions. Speaker notes are on every slide.

## Live demo plan

1. Before the session, run once so everything is cached: `cd code` then `python milestone3_analysis.py` (about 6 minutes; later runs take about a minute).
2. During the demo open `index.html`: read the "Start here" box, press the quick view "Next 7 days, all seasons" under Compare models, then open See forecasts (Old vs new, period Oct-Jan), then scroll to "Why was Milestone 2 not accurate?".
3. Show `results/predictions_all_folds.csv` and say that every reported number can be recomputed from it.
4. Mention `HOW_TO_RUN.txt`: set-up and run in a few commands, and the clean re-run gave identical results.

## About the question our faculty raised

**Q1. In Milestone 2 you used strong models but accuracy was low. Why?**
Because the models were given too little information. They saw one daily average per day: not how the day ended, not other pollutants, not the weather forecast. A strong model on that single series gives the same accuracy: XGBoost with PM2.5 history only has RMSE 24.27, no better than ARIMA's 22.90. Details: `docs/M2_Review_and_Corrections.md`.

**Q2. What exactly did you change to fix it?**
(1) Intraday statistics (last hour, evening, morning, max, min) from the hourly data. (2) Co-pollutants (PM10, NO2, CO, AQI) and observed weather. (3) Archived ECMWF weather forecasts, used only if issued before the forecast origin. (4) Tuned machine-learning and deep-learning models and an ensemble. (5) Four test seasons instead of one window, with significance tests.

**Q3. Is the accuracy now good?**
Next day: yes. Ensemble RMSE 18.36 against 22.90 for ARIMA over four seasons (20% lower), R² 0.77, MAPE 15%, right pollution category 67% of days. Week ahead: clearly better than before (16% lower than ARIMA, R² 0.48) but still harder: error grows with horizon for every model. Extreme days above 120 µg/m³ remain under-predicted.

## About the data

**Q4. Where does the data come from?**
Open-Meteo: air quality from CAMS (Copernicus), weather from ERA5, forecasts from ECMWF IFS (Previous Runs API). One grid cell in central New Delhi, hourly, 5 Aug 2022 to 20 Oct 2026, aggregated to 1,538 days. Licence CC BY 4.0.

**Q4b. Is the dataset recent enough?**
Yes. It runs from 1 Aug 2022 to 20 Oct 2026 (about 4.2 years), the last observation is about a month old, and it has 37,008 hourly rows (1,538 days). The link in the form downloads exactly this data.

**Q5. Are these real measurements?**
No. They are modelled grid values, not station readings, so extreme peaks are smoothed. This is a stated limitation.

**Q6. Why not use CO2?**
It exists only from October 2024 (725 days) and correlates weakly with PM2.5 (r = 0.32).

## About the method

**Q7. How do you know there is no data leakage?**
Features at a forecast origin are recomputed from data truncated at that day and match exactly (checked on random origins). Hyper-parameters are tuned only on data before the first test fold. Weather forecasts follow an issue-time rule: for a forecast h days ahead only the forecast issued h days earlier is used, and forecast skill decays with lead time (wind correlation with ERA5 0.90 at one day, 0.64 at seven), which a leak would not show. Every test day uses a real archived forecast.

**Q8. What is the weather-forecast stand-in?**
The archive starts in March 2024. For earlier training days we use the actual weather instead, about half of the earliest fold's training days and none of any test day. This is disclosed in the report.

**Q9. What is the ensemble and why is it fair?**
The equal-weight mean of the log-forecasts of XGBoost, LightGBM, Random Forest, LSTM and GRU. It was fixed before we saw the results and is our final model.

**Q10. Why four folds?**
Milestone 2 used one 90-day window (July to October), the easiest season. Four consecutive 90-day windows (Oct-Jan, Jan-Apr, Apr-Jul, Jul-Oct) test every season once. The last one is exactly the Milestone 2 window, so numbers are directly comparable.

**Q11. Why did you run the deep learning in a separate script?**
PyTorch and XGBoost/LightGBM ship conflicting OpenMP libraries and freeze when loaded together on macOS. The separate process is started automatically.

## About the results

**Q12. Which model is best?**
The ensemble: lowest next-day RMSE (18.36) and within 10% of the best learner in every fold. Best single models: GRU (18.55 next day, 26.69 week) pooled; LightGBM (10.21) on the Milestone 2 window.

**Q13. Why do deep networks not clearly beat trees?**
With about 1,400 training days the networks are data-limited. They help most at one week, where they combine the recent pollution state with a 7-day forecast vector.

**Q14. What drives the forecast?**
SHAP: last-hour and evening PM2.5 of the previous day, yesterday's mean, and the forecast temperature range and wind. Intraday statistics 36%, forecast weather 24%, history 18% of attribution. Physically: small temperature range and low wind mean stagnant air.

**Q15. Where does the model fail?**
On extreme days. Above 120 µg/m³ the ensemble under-predicts by 25 µg/m³ on average and catches 66% of them. Alert thresholds should be set below the official limits. It also is no better than ARIMA in late May to early June.

**Q16. Is the improvement statistically significant?**
Yes. Diebold-Mariano: ensemble against ARIMA p < 0.0001 next day and p = 0.003 week ahead; against SARIMA at one week p = 0.0004.

**Q17. What would you do next?**
Combine several weather models and add boundary-layer height, add station and fire data, probabilistic forecasts for extremes, and an operational daily pipeline.

## Short honest statements to make if asked about teamwork or limits
- Team leader: Mahak Soni. Contributions are on the team slide and in the report appendix: Mahak (leader) did the data pipeline, classical and tree models, tuning and the report; Darsh did the forecast integration, Prophet, LSTM and GRU, the analysis, the website and the slides; Palak's entry is GFG.
- The course requires every member present at the viva to take part.
- We did not validate against station data; grid values may differ from stations.
- The forecast archive's definition of "previous day" forecasts is relied on as documented by the provider.
