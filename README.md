# Forecasting Delhi's PM2.5: Classical, Machine-Learning and Deep-Learning Models

Time Series Analysis (CSET 369), Milestone 3 (final), Bennett University
**Team Leader: Mahak Soni (S24CSEU2741)**, Darsh Jasoria (S24CSEU0844), Palak Aitwani (S24CSEU0326)
Faculty: Anurag Choubey

Open `index.html` for the interactive website (works offline). New here? Read `HOW_TO_RUN.txt`: it explains setup, running, GitHub and Vercel step by step.

## Team

| Member | Roll no. | Role | Contribution |
|---|---|---|---|
| Mahak Soni | S24CSEU2741 | **Team Leader** | Project lead and final submission; data pipeline and intraday features; classical recap (Naive, ARIMA, SARIMA, Holt-Winters); tree models and tuning; IEEE report |
| Darsh Jasoria | S24CSEU0844 | Team member | Weather-forecast download and issue-time checks; Prophet, LSTM, GRU; ensemble, SHAP, ablation, significance tests; website, slides, GitHub and Vercel setup; Milestone 2 review and viva Q&A |
| Palak Aitwani | S24CSEU0326 | Team member | GFG |

## What is in this folder (every file appears once)

| Path | What it is |
|---|---|
| `README.md` | This file: start here |
| `index.html` | The website: interactive model comparison, forecast explorer, insights, team. Double-click to open; Vercel publishes this page |
| `Mahak Soni_S24CSEU2741_Milestone3.pptx` | Final presentation (23 slides, with speaker notes) |
| `Mahak Soni_S24CSEU2741_Milestone3_Report.docx` | Final report (IEEE format) |
| `HOW_TO_RUN.txt` | Plain-text guide: setup, run, deploy, troubleshooting, project facts |
| `code/` | The project: `m3_data.py`, `m3_forecasts.py`, `milestone3_analysis.py`, `milestone3_dl.py`, plus `requirements.txt`, `milestone2/` (Milestone 2 code for the form) and `build/` (rebuilds the website, report and slides from `results/`) |
| `data/` | Hourly air quality (the dataset), hourly weather, ECMWF forecast archive, Milestone 2 daily file |
| `results/` | Final tables, predictions for every fold and tuned hyper-parameters; `figures/` (the 11 report figures) |
| `docs/` | `Guidelines_Checklist.md` (every course requirement and where it is met), `M2_Review_and_Corrections.md` (why Milestone 2 accuracy was low), `Viva_QA.md` (time plan, live demo plan, likely questions), `code_pdf/` (the code as PDF, in case the form rejects `.py` files) |
| `LICENSE`, `vercel.json`, `.vercelignore`, `.gitignore` | MIT license for the code; deployment and git settings (these must stay at the top level) |

## Microsoft Form: what to upload, field by field

The course guidelines give no file-name rules (only the types: report PDF/DOCX up to 10 pages, code `.ipynb`/`.py`, slides `.pptx`). Names follow the Milestone 2 convention. Fill any other fields above "Project Title" as usual.

| Form field | Limit shown on the form | What to enter or upload |
|---|---|---|
| Project Title | text | Forecasting Delhi's PM2.5 with Classical, Machine-Learning and Deep-Learning Time Series Models |
| Dataset Link | text | `https://air-quality-api.open-meteo.com/v1/air-quality?latitude=28.60&longitude=77.20&start_date=2022-08-01&end_date=2026-10-20&hourly=pm2_5,pm10,nitrogen_dioxide,carbon_monoxide,carbon_dioxide,us_aqi&timezone=Asia%2FKolkata&format=csv` (opens the dataset as a CSV file: hourly Delhi air quality, 1 Aug 2022 to 20 Oct 2026, 37,008 rows; the same data is in `data/delhi_hourly_raw.csv`) |
| Milestone 2 code | up to 5 files, 100 MB each | 2 files: `code/milestone2/milestone2_analysis.py` and `code/milestone2/milestone2_analysis.ipynb` |
| Milestone 3 Report | 1 document, 10 MB | 1 file: `Mahak Soni_S24CSEU2741_Milestone3_Report.docx` (1.2 MB) |
| Milestone 3 Code | up to 5 files, 100 MB each | 5 files: `code/m3_data.py`, `code/m3_forecasts.py`, `code/milestone3_analysis.py`, `code/milestone3_dl.py`, `code/requirements.txt` |
| Milestone 3 PPT | 1 file, PDF or presentation, 10 MB | 1 file: `Mahak Soni_S24CSEU2741_Milestone3.pptx` (1.6 MB) |

If a code field rejects `.py` / `.ipynb`, upload the PDF versions instead: `docs/code_pdf/Milestone2_code.pdf` in the Milestone 2 field and `docs/code_pdf/Milestone3_code.pdf` in the Milestone 3 field (one file each; the PDF contains all the scripts). If the form allows one response per team, the team leader (Mahak Soni) submits it.

## Dataset (recent, 2022 to 2026)

| | |
|---|---|
| Air quality | Open-Meteo Air Quality API (CAMS), hourly, Delhi grid cell 28.60 N 77.20 E; the form link above downloads exactly this file |
| Period | 1 Aug 2022 to 20 Oct 2026 (about 4.2 years; 1,538 usable days; last observation about one month before submission) |
| Size | 37,008 hourly rows; the course minimum is 500 |
| Also used | Weather: https://open-meteo.com/en/docs/historical-weather-api (ERA5); archived forecasts: https://open-meteo.com/en/docs/previous-runs-api (ECMWF IFS, from March 2024) |
| Verified | The CSV link was fetched and compared with `data/delhi_hourly_raw.csv`: all 37,008 rows identical |

## Headline results (PM2.5, original µg/m³ scale)

| | ARIMA (Milestone 2) | Milestone 3 Ensemble |
|---|---|---|
| Next-day RMSE, M2 hold-out (23 Jul to 20 Oct 2026) | 14.95 | **10.53** (30% lower; best single model LightGBM 10.21) |
| Next-day MAPE, M2 hold-out | 23.1 % | 14.6 % |
| Next-day RMSE, four seasons pooled (360 days) | 22.90 | **18.36** (20% lower) |
| Week-ahead RMSE, four seasons pooled | 32.66 (SARIMA 30.79) | **27.54** (16% lower than ARIMA; GRU 26.69) |
| Week-ahead R², four seasons | 0.27 | 0.48 |
| Correct CPCB pollution category (next day) | 57 % | 67 % |

The week-ahead gain comes from archived ECMWF weather forecasts: removing them from the final XGBoost model raises its week-ahead error from 30.27 to 34.03 (ablation in the report). A forecast made h days ahead uses only the weather forecast issued h days earlier, so nothing from the future leaks in. Details and limits are in the report.

## Reproduce

```bash
pip install -r code/requirements.txt
cd code
python milestone3_analysis.py          # everything, with weather forecasts (about 6 minutes)
cd ..
python code/build/build_dashboard.py   # rebuild index.html
python code/build/build_report.py      # rebuild the report
python code/build/build_ppt.py         # rebuild the slides
```

`python code/m3_forecasts.py download ecmwf_ifs025` re-downloads the forecast archive (internet needed; the file is already in `data/`). PyTorch and XGBoost/LightGBM deadlock in one process on macOS, which is why the LSTM/GRU code is a separate script that starts automatically.

## Publish on GitHub and Vercel

```bash
git init && git add . && git commit -m "Milestone 3: forecasting Delhi PM2.5"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

On vercel.com choose Add New, Project, import the repository and click Deploy. `index.html` is at the root, so no build step or setting is needed; `.vercelignore` publishes only the website, report, slides and guide. Details: `HOW_TO_RUN.txt`, section 9.

## Data attribution

Air quality: Open-Meteo Air Quality API (CAMS, Copernicus Atmosphere Monitoring Service). Weather: Open-Meteo Historical Weather API (ERA5). Forecasts: Open-Meteo Previous Runs API (ECMWF IFS). Open-Meteo.com data are licensed CC BY 4.0.

## Design choices to state in the viva

- Same protocol as Milestone 2 (next-day and week-ahead, 90-day hold-out), plus three earlier folds so each season is tested once.
- Hyper-parameters were tuned only on data before the first fold. The weather model (ECMWF over GFS) was chosen on a January to June 2025 pilot, also before any test fold.
- The final model, the Ensemble, was fixed before any result was seen.
- The forecast archive starts in March 2024: earlier training days use the actual weather as a stand-in (about half of the earliest fold's training days); every test day uses a real archived forecast.
- Data are modelled grid values (CAMS / ERA5), not station readings: peaks are muted and days above 120 µg/m³ are under-predicted.
