# Course guidelines checklist (Time Series Analysis, CSET 369, Phase 3: 12 marks)

Every requirement from the "Final Project Submission: Guidelines, Rubric and Report Format" and where it is met in this project.

## Submission files

| Requirement | Status | Where |
|---|---|---|
| Project report, PDF or DOCX, at most 10 pages | Met (DOCX; about 6 to 7 pages as opened in Word by the team; layout estimate at most 9) | `Mahak Soni_S24CSEU2741_Milestone3_Report.docx` |
| Source code, `.ipynb` or `.py` | Met | `code/` (4 scripts), `code/milestone2/` |
| Presentation slides, `.pptx` | Met (23 slides with speaker notes) | `Mahak Soni_S24CSEU2741_Milestone3.pptx` |
| Mode: lab demonstration plus Q&A, 10 to 12 minutes per team | Prepared | Time plan and live demo plan in `docs/Viva_QA.md`; speaker notes on every slide |
| File and folder names | The guidelines give no naming rules; the Milestone 2 convention is followed | `Mahak Soni_S24CSEU2741_Milestone3...` |

## Dataset (faculty's advice: not too old)

| Point | Status | Evidence |
|---|---|---|
| Recent data | Met | 1 Aug 2022 to 20 Oct 2026; last observation about one month before submission |
| Enough data | Met | 37,008 hourly rows, 1,538 usable days (course minimum 500) |
| A link to paste in the form | Met | The Open-Meteo CSV link in `README.md`; fetched and verified identical to `data/delhi_hourly_raw.csv` |

## Phase 3 work expected

| Requirement | Status | Evidence |
|---|---|---|
| Implement at least one advanced model (LSTM/GRU, XGBoost/ensemble, Prophet) | Met, all three families | Ridge, Random Forest, XGBoost, LightGBM, Prophet, LSTM, GRU, ensemble; report IV-C |
| Select models from data characteristics and justify | Met | Report IV-C "Model choice follows the data"; slide 10 column "Why it is here" |
| Hyper-parameter tuning (basic) | Met | Randomised search (trees) and grid (LSTM/GRU) before the first test fold; report IV-D, appendix; `results/best_params.json`, `dl_params.json` |
| Proper training and validation | Met | Four rolling 90-day folds, expanding window, no test data used for tuning; leakage checks |
| Compare classical (Phase 2) with advanced (Phase 3) using RMSE, MAE, MAPE | Met | Report Tables I and II; slides 11 to 13 (ablation on 13); `results/model_comparison_*.csv` |
| Highlight which model is better and why it differs | Met | Report V-A, V-B, V-F; Diebold-Mariano tests; slides 12, 16, 17 |
| Insights: patterns, trends, anomalies, real-world meaning | Met | SHAP and ablation (V-D), extremes and alert detection (V-E), seasonal behaviour (V-B), Milestone 2 review (V-G) |
| Limitations, data challenges, possible improvements | Met | Report VI; slide 19 |

## Report structure (IEEE format)

| Required section | Status | Where in the report |
|---|---|---|
| 1. Title page: project title, team members with name and roll number | Met | Title block (Mahak Soni is Team Leader; faculty Anurag Choubey) |
| 2. Abstract, 150 to 200 words | Met (161 words) | Abstract |
| 3. Introduction: background, importance, motivation, objectives, structure | Met | Section I |
| 4. Literature review: 4 to 5 related works, existing models, gaps | Met (5 works, gap stated) | Section II |
| 5. Dataset and preprocessing: source, type, time range, features, steps, a line plot | Met | Section III, Fig. 1 |
| 6. Methodology: classical and advanced models, training/testing, tuning, tools, workflow diagram | Met | Section IV, Fig. 2 |
| 7. Results and discussion: RMSE, MAE, MAPE, forecast plots, interpretation, comparison | Met | Section V, Tables I to IV, Figs. 3 to 7 |
| 8. Conclusion and future work | Met | Section VI |
| 9. References: 9 to 10 in IEEE format | Met (10) | References |
| 10. Appendix (optional) | Included | Hyper-parameters, reproduction, team contributions (Table V) |

## Presentation (the eight required parts)

| Required part | Slides |
|---|---|
| 1. Problem overview | 3, 4 |
| 2. Dataset summary | 5 |
| 3. Phase 2 recap (classical models) | 6, 7 |
| 4. Advanced models used | 8, 9, 10 |
| 5. Comparative results | 11 to 16 |
| 6. Key insights | 17, 18 |
| 7. Challenges and limitations | 19 |
| 8. Conclusion | 20 |
| Extras | 1 title, 2 agenda, 21 live demo, 22 team, 23 references |

## Rules and points to be ready for

- **Plagiarism and AI checks:** the report text was written with AI assistance for this project, with every figure and number generated from the project's own results and every source cited. Because reports are checked, each member should read the report and be able to explain it in their own words.
- **Every member takes part in the viva** (course rule).
- **Equal participation:** contributions are recorded in the team table (slide 22, report appendix, README) as they were reported to us.
- **Common mistakes to avoid:** advanced models used without understanding (Q&A prepared), no comparison with classical models (done), poorly structured report (IEEE structure above), copy-paste without citation (all sources cited), no interpretation of results (Section V-D to V-G).
- **Late submission** loses marks: submit before the deadline.
