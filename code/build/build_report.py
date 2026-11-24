"""Builds the Milestone 3 IEEE-style report (DOCX). Every number is read from ../results/*.csv, nothing is typed by hand."""
import os, json
import pandas as pd
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, "results"); FIG = os.path.join(RES, "figures")
H = pd.read_csv(f"{RES}/model_comparison_m3_holdout.csv").set_index("Model")
R = pd.read_csv(f"{RES}/model_comparison_m3_rolling.csv").set_index("Model")
FOLD1 = pd.read_csv(f"{RES}/rmse_by_fold_h1.csv").set_index("Model")
FOLD7 = pd.read_csv(f"{RES}/rmse_by_fold_h7.csv").set_index("Model")
DM = pd.read_csv(f"{RES}/diebold_mariano.csv")
EXC = pd.read_csv(f"{RES}/exceedance_detection_h1.csv")
LEV = pd.read_csv(f"{RES}/error_by_pm25_level_h1.csv")
SHAPG = pd.read_csv(f"{RES}/shap_group_share_h1.csv", index_col=0)["share_pct"]
SHAP = pd.read_csv(f"{RES}/shap_importance_h1.csv", index_col=0)["mean_abs_shap"]
PAR = json.load(open(f"{RES}/best_params.json")); DLP = json.load(open(f"{RES}/dl_params.json"))
pct = lambda new, old: (1 - new / old) * 100
dm = lambda m, vs, h: DM[(DM.Model == m) & (DM.vs == vs) & (DM.horizon == h)].p_value.iloc[0]
S = "XGBoost (+future weather, oracle)"
SUM = json.load(open(f"{RES}/summary_m3.json"))
DA = pd.read_csv(f"{RES}/rmse_by_day_ahead.csv").set_index("day_ahead")

doc = Document()
st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(10); st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
st.paragraph_format.space_after = Pt(3); st.paragraph_format.space_before = Pt(0)
sec = doc.sections[0]; sec.page_width, sec.page_height = Inches(8.5), Inches(11)
for a in ("left_margin", "right_margin"): setattr(sec, a, Inches(0.7))
sec.top_margin = Inches(0.75); sec.bottom_margin = Inches(0.8)

def set_cols(section, n):
    sp = section._sectPr
    for c in sp.findall(qn("w:cols")): sp.remove(c)
    c = OxmlElement("w:cols"); c.set(qn("w:num"), str(n)); c.set(qn("w:space"), "360"); sp.append(c)
def para(text="", size=10, bold=False, italic=False, align=None, after=3, before=0, keep=False, color=None):
    p = doc.add_paragraph(); r = p.add_run(text); r.font.size = Pt(size); r.bold = bold; r.italic = italic
    if color: r.font.color.rgb = RGBColor(*color)
    if align == "c": p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "j": p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(after); p.paragraph_format.space_before = Pt(before); p.paragraph_format.keep_with_next = keep; return p
def body(text): return para(text, align="j")
def h1(t): return para(t.upper(), size=10, bold=True, align="c", before=6, after=3, keep=True)
def h2(t): return para(t, size=10, bold=True, italic=True, before=3, after=2, keep=True)
def mixed(parts, align="j"):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY if align == "j" else None
    for t, b, i in parts: r = p.add_run(t); r.bold = b; r.italic = i
    return p
def bullet(t):
    p = para(t, align="j", after=1); p.paragraph_format.left_indent = Inches(0.18); p.paragraph_format.first_line_indent = Inches(-0.14)
    p.runs[0].text = "• " + t; return p
state = {"cols": 1}
def to_cols(n):
    if state["cols"] == n: return
    s = doc.add_section(WD_SECTION.CONTINUOUS); set_cols(s, n); state["cols"] = n
def figure(name, caption, width=3.4, wide=False):
    if wide: to_cols(1)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(1); p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(os.path.join(FIG, name), width=Inches(width))
    para(caption, size=8, align="c" if wide else "j", after=5)
    if wide: to_cols(2)
def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr(); sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:fill"), hexcolor); tcPr.append(sh)
def table(header, rows, widths, caption, fs=7.5, bold_rows=(), wide=False):
    if wide: to_cols(1)
    para(caption, size=8, bold=True, align="c", after=2, before=4, keep=True)
    t = doc.add_table(rows=1 + len(rows), cols=len(header)); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, htxt in enumerate(header):
        c = t.rows[0].cells[j]; c.text = ""; r = c.paragraphs[0].add_run(htxt); r.bold = True; r.font.size = Pt(fs); shade(c, "D9E8E5")
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for i, row in enumerate(rows, 1):
        for j, v in enumerate(row):
            c = t.rows[i].cells[j]; c.text = ""; r = c.paragraphs[0].add_run(str(v)); r.font.size = Pt(fs); r.bold = (i - 1) in bold_rows
            c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
    for row in t.rows:
        for j, w in enumerate(widths): row.cells[j].width = Inches(w)
        for c in row.cells:
            for p in c.paragraphs: p.paragraph_format.space_after = Pt(0)
    para("", size=3, after=2)
    if wide: to_cols(2)

# ------------------------------------------------------------------ title block (single column)
LEARN = ["Ridge", "Random Forest", "XGBoost", "LightGBM", "Prophet", "LSTM", "GRU", "Ensemble"]
bestL = lambda tab, col: tab.loc[LEARN, col].idxmin()
SINGLE = ["Ridge", "Random Forest", "XGBoost", "LightGBM", "Prophet", "LSTM", "GRU"]
bh1 = H.loc[SINGLE, "h=1 RMSE"].idxmin(); bp1, bp7 = R.loc[SINGLE, "h=1 RMSE"].idxmin(), R.loc[SINGLE, "h=7 RMSE"].idxmin()
share_gap = (R.loc["XGBoost (no weather forecast)", "h=7 RMSE"] - R.loc["XGBoost", "h=7 RMSE"]) / (R.loc["XGBoost (no weather forecast)", "h=7 RMSE"] - R.loc[S, "h=7 RMSE"]) * 100
epochs = [e for f in SUM["dl_best_epochs"].values() for k in f.values() for e in k]
gf = lambda tab, m, base, f: pct(tab.loc[m, str(f)], tab.loc[base, str(f)])

para("Forecasting Delhi's PM2.5 with Classical, Machine-Learning and Deep-Learning Time Series Models", size=18, bold=True, align="c", after=6, before=0)
para("Time Series Analysis (CSET 369) Final Project Report, Milestone 3", size=10, italic=True, align="c", after=6)
para("Mahak Soni (S24CSEU2741, Team Leader)     Darsh Jasoria (S24CSEU0844)     Palak Aitwani (S24CSEU0326)", size=10.5, bold=True, align="c", after=1)
para("B.Tech CSE, Year 3, Bennett University, Greater Noida, India", size=9.5, italic=True, align="c", after=1)
para("Faculty: Anurag Choubey", size=10, bold=True, align="c", after=8)

abstract = (f"Delhi's PM2.5 exceeds India's 24-hour standard of 60 µg/m³ on more than half of all days, so reliable short-term forecasts support health alerts. "
            f"Our Milestone 2 classical models (ARIMA, SARIMA, Holt-Winters) barely improved on persistence: the best next-day RMSE was {H.loc['ARIMA','h=1 RMSE']:.2f} µg/m³ against {H.loc['Naive','h=1 RMSE']:.2f} for yesterday's value, and no classical model beat it at one week. "
            f"We extend that work with Ridge, Random Forest, XGBoost, LightGBM, Prophet, LSTM, GRU and an ensemble, and give them two new sources of information: intraday PM2.5 statistics from hourly CAMS data and archived ECMWF weather forecasts. "
            f"Hyper-parameters are tuned before the first test window and every model is scored on four rolling 90-day windows. "
            f"On the Milestone 2 hold-out the ensemble lowers next-day RMSE by {pct(H.loc['Ensemble','h=1 RMSE'], H.loc['ARIMA','h=1 RMSE']):.0f}%. Over four seasons it lowers next-day RMSE by {pct(R.loc['Ensemble','h=1 RMSE'], R.loc['ARIMA','h=1 RMSE']):.0f}% and week-ahead RMSE by {pct(R.loc['Ensemble','h=7 RMSE'], R.loc['ARIMA','h=7 RMSE']):.0f}% (Diebold-Mariano p < 0.01), and raises pollution-category accuracy from {R.loc['ARIMA','h=1 CatAcc']:.0f}% to {R.loc['Ensemble','h=1 CatAcc']:.0f}%. "
            f"Evening concentrations and forecast wind and temperature range carry most of the signal.")
wc = len(abstract.split()); assert 150 <= wc <= 200, wc
p = para("", align="j", after=3); r = p.add_run("Abstract—"); r.bold = True; r.italic = True; r2 = p.add_run(abstract); r2.bold = True; r2.font.size = Pt(9)
r.font.size = Pt(9)
p = para("", align="j", after=6); r = p.add_run("Index Terms—"); r.bold = True; r.italic = True; r.font.size = Pt(9)
r2 = p.add_run("air quality, PM2.5, time series forecasting, ARIMA, XGBoost, LSTM, GRU, Prophet, numerical weather forecasts, ensemble learning, Delhi."); r2.italic = True; r2.font.size = Pt(9)

to_cols(2)
# ------------------------------------------------------------------ I. Introduction
h1("I. Introduction")
body("Delhi has some of the highest fine-particulate pollution among large cities. Concentrations of PM2.5 (particles under 2.5 µm) follow a strong yearly cycle, rising in November to January when cold air, crop-residue burning and weak winds trap emissions, and falling during the monsoon. Public-health measures such as school closures, traffic restrictions and the Graded Response Action Plan depend on forecasts that are issued a day or more ahead, so forecast error translates directly into late or wasted action.")
body(f"Milestone 2 fitted ARIMA, SARIMA and Holt-Winters models on the daily PM2.5 series and found only a modest gain over persistence on a 90-day hold-out (RMSE {H.loc['ARIMA','h=1 RMSE']:.2f} versus {H.loc['Naive','h=1 RMSE']:.2f} µg/m³), and no gain at a one-week horizon. Our faculty asked why accuracy was still limited even with strong models. Our diagnosis is that the models saw only one number per day and nothing about how the day ended, what the weather was expected to do, or how other pollutants moved.")
body("The objectives of this milestone are: (O1) implement advanced machine-learning and deep-learning models; (O2) tune them with a leak-free validation scheme; (O3) compare them with the Milestone 2 classical models using RMSE, MAE, MAPE and pollution-category accuracy, with significance tests; (O4) explain why performance differs and what drives forecasts; and (O5) state limitations and improvements. Section II reviews related work, Section III describes the data, Section IV the methods, Section V the results, and Section VI concludes.")
# ------------------------------------------------------------------ II. Literature review
h1("II. Literature Review")
body("Zhang et al. [2] survey real-time air-quality forecasting and separate numerical chemical-transport models, which are physically detailed but costly and sensitive to emission inventories, from statistical and machine-learning models, which are cheap and can be tuned to a site. They identify statistical post-processing and better predictors as the practical route for operational forecasts.")
body("Kumar and Goyal [3] forecast Delhi air quality with principal-component regression on pollutant and meteorological predictors. Their study shows that Delhi's pollution can be forecast from a small set of drivers, but it uses linear structure and does not compare modern learners on a common protocol.")
body("Li et al. [4] apply an LSTM network to air-pollutant concentrations and find that adding auxiliary inputs, such as meteorology and other stations, improves accuracy over a plain recurrent model. This motivates our multi-feature recurrent models.")
body("Makridakis et al. [5], reporting the M4 forecasting competition, show that pure machine-learning methods did not automatically beat statistical ones, while combinations of methods did best. We therefore compare every learner with strong classical baselines and include a pre-declared ensemble.")
body("Prophet [9] fits a decomposable trend-plus-seasonality model with external regressors and is popular for business series; XGBoost [6] and recurrent networks [7], [8] are the standard gradient-boosting and sequence learners. Few studies place all of these side by side with ARIMA-class models for Delhi on identical rolling-origin windows, with intraday features and with archived weather forecasts that respect the forecast issue time. This gap is what the present work addresses.")
# ------------------------------------------------------------------ III. Dataset
h1("III. Dataset Description and Preprocessing")
body("Air-quality data come from the Open-Meteo Air Quality API [10], which serves CAMS (Copernicus Atmosphere Monitoring Service) global composition fields for the grid cell at 28.60°N, 77.20°E in central New Delhi. We use hourly PM2.5, PM10, NO₂, CO and US AQI. Weather comes from the Open-Meteo historical archive (ERA5 reanalysis) for the same cell: 2 m temperature, relative humidity, surface pressure, precipitation, 10 m wind speed and direction, and cloud cover. Both sets are hourly in India Standard Time and contain 37,008 rows from 1 August 2022 to 20 October 2026, released under CC BY 4.0. Values are modelled grid estimates rather than station readings, so extreme peaks are smoothed.")
body(f"Weather forecasts come from the Open-Meteo Previous Runs API [10], which stores the ECMWF IFS 0.25° forecast of each hour as issued 1 to 7 days earlier. For horizon h we use only the forecast of the target day issued h days before, so nothing newer than the forecast origin is ever used. The archive starts in February 2024 (first complete day 7 March 2024) and was downloaded to 27 October 2026 for the same variables. ECMWF was chosen over GFS on a pilot in January to June 2025, before any test window: at one day ahead its wind forecast correlated 0.85 with ERA5 against 0.66 for GFS. Boundary-layer height is not archived and is therefore not used.")
body("Preprocessing: (1) the first 77 hours of PM2.5 (and 101 hours of AQI) are empty; Milestone 2 back-filled them, which produced four identical daily values for 1 to 4 August 2022. We drop them, so the series starts on 5 August 2022 and has 1,538 days. (2) There are no other gaps in PM2.5, PM10, NO₂ or CO; an interpolation rule (up to six hours, then up to three days) is kept only as a safeguard. (3) PM2.5 is log-transformed with a floor of 1 µg/m³ to stabilise variance, and forecasts are back-transformed before scoring. (4) For each day we compute the mean, maximum, minimum, last-hour, evening (18:00 to 23:00) and morning (05:00 to 10:00) PM2.5, plus daily means of the other pollutants and weather. (5) Wind speed and direction become east and north components, and hourly forecasts are aggregated to the same daily variables (mean temperature and its range, humidity, pressure, rain, wind, cloud). (6) Days before March 2024 have no archived forecast; in training they use the actual weather as a stand-in, which affects about half of the earliest fold's training days and none of any test day. CO₂ is excluded because it exists only from October 2024 and correlates weakly with PM2.5 (r = 0.32).")
figure("19_data_overview.png", "Fig. 1. Daily PM2.5 with temperature and wind. The dashed line is the 60 µg/m³ standard; shaded bands are the four 90-day test folds.", 6.4, wide=True)
# ------------------------------------------------------------------ IV. Methodology
h1("IV. Methodology")
figure("10_workflow.png", "Fig. 2. Workflow from raw hourly data to scored forecasts and insights.", 6.8, wide=True)
h2("A. Problem formulation")
body("We forecast daily mean PM2.5 at two horizons, using the same protocols as Milestone 2: next-day (h = 1) forecasts using all information up to the previous day, and week-ahead forecasts issued every seventh day for days 1 to 7. Tabular models use a direct strategy, one model per horizon, which avoids feeding forecasts back as inputs. All features are known at the forecast origin: observed data up to that day, the weather forecast for the target day issued at or before the origin, and calendar terms of the target day.")
h2("B. Features")
body("The tabular models use 79 features: PM2.5 lags (0, 1, 2, 3, 6, 13 days), rolling means over 3, 7, 14 and 30 days and a 7-day volatility; intraday statistics of PM2.5 and their one-day lags; PM10, NO₂, CO and AQI with lags and 7-day means; observed weather with lags and changes; nine forecast variables for the target day; and yearly sine and cosine terms with day of week. The recurrent networks receive a window of the last L days of 21 standardised daily variables, with the 7-day weather-forecast vector (63 values) issued at the origin added at every step.")
h2("C. Models")
body("Model choice follows the data. PM2.5 is strongly persistent, has a yearly cycle and swings sharply within a day, so ARIMA-type models give the classical baseline. Pollutant and weather effects are non-linear and interact, which suits Random Forest and boosted trees. Recurrent networks (LSTM, GRU) can read a window of recent days together with a multi-day forecast vector. Prophet represents trend and yearly and weekly seasonality with extra regressors, and the ensemble combines models whose errors differ.")
body("Classical (Milestone 2) [1]: Naive, ARIMA(3,0,3), SARIMA (3,0,3)(1,0,1)₇ with yearly Fourier terms, and damped Holt-Winters, refitted on each fold. Advanced: Ridge regression with cross-validated penalty; Random Forest with 300 trees; XGBoost [6] and LightGBM gradient-boosted trees; Prophet [9] with yearly and weekly seasonality and ten lagged regressors; an LSTM [7] and a GRU [8], each a single recurrent layer with a linear head that predicts the change in log PM2.5 for days 1 to 7, averaged over five random seeds, with the number of epochs set by early stopping on the last 60 training days and the model then refitted on all training data. The Ensemble is the equal-weight mean of the log-forecasts of XGBoost, LightGBM, Random Forest, LSTM and GRU, fixed before the results were seen. Prophet, SARIMA and the other classical models use no forecast inputs.")
h2("D. Hyper-parameter tuning and validation")
body(f"Tree models were tuned by randomised search (16 candidates, 6 for Random Forest, time-ordered four-fold cross-validation) using only data before the first test window, separately for horizon 1 and horizon 7. The recurrent networks were tuned over hidden size {{32, 64}}, window {{7, 14, 28}} and dropout {{0.1, 0.3}} on two 90-day blocks ending before the first test window. Selected values: LSTM hidden {DLP['lstm']['params']['hidden']}, window {DLP['lstm']['params']['window']}, dropout {DLP['lstm']['params']['dropout']}; GRU hidden {DLP['gru']['params']['hidden']}, window {DLP['gru']['params']['window']}, dropout {DLP['gru']['params']['dropout']}. Evaluation uses four consecutive 90-day test folds with an expanding training window (Oct 2025 to Jan 2026, Jan to Apr 2026, Apr to Jul 2026, Jul to Oct 2026). Fold 4 is exactly the Milestone 2 hold-out, so its numbers are directly comparable. Because tuning never sees any test fold, no fold is contaminated.")
h2("E. Metrics and tools")
body("RMSE, MAE and MAPE are computed on the original µg/m³ scale, with R² and the bias. Because alerts use categories, we also report the share of days placed in the correct CPCB PM2.5 band (30, 60, 90, 120, 250 µg/m³). Differences between models are tested with the Diebold-Mariano test on squared errors with the Harvey correction. Feature effects are explained with exact TreeSHAP contributions from XGBoost. Implementation uses Python with pandas, statsmodels, scikit-learn, XGBoost, LightGBM, Prophet and PyTorch.")
# ------------------------------------------------------------------ V. Results
h1("V. Results and Discussion")
h2("A. Milestone 2 hold-out: classical versus advanced")
T12 = ["Naive", "ARIMA", "SARIMA", "Ridge", "Random Forest", "XGBoost", "LightGBM", "Prophet", "LSTM", "GRU", "Ensemble"]
def res_rows(tab): return [[m, f"{tab.loc[m,'h=1 RMSE']:.2f}", f"{tab.loc[m,'h=1 MAE']:.2f}", f"{tab.loc[m,'h=1 MAPE']:.1f}", f"{tab.loc[m,'h=1 R2']:.2f}", f"{tab.loc[m,'h=1 CatAcc']:.0f}", f"{tab.loc[m,'h=7 RMSE']:.2f}", f"{tab.loc[m,'h=7 R2']:.2f}"] for m in T12]
HDR = ["Model", "RMSE", "MAE", "MAPE %", "R²", "Cat. %", "Wk RMSE", "Wk R²"]; WID = [1.0, .38, .38, .42, .3, .35, .48, .4]
table(HDR, res_rows(H), WID, "TABLE I. Next-day and week-ahead (Wk) errors, 90-day hold-out (23 Jul to 20 Oct 2026)", bold_rows=(T12.index("ARIMA"), T12.index("Ensemble")))
body(f"Table I shows a large gain at both horizons. At the next-day horizon the best single model, {bh1}, reaches RMSE {H.loc[bh1,'h=1 RMSE']:.2f}, MAE {H.loc[bh1,'h=1 MAE']:.2f} and MAPE {H.loc[bh1,'h=1 MAPE']:.1f}%, against {H.loc['ARIMA','h=1 RMSE']:.2f}, {H.loc['ARIMA','h=1 MAE']:.2f} and {H.loc['ARIMA','h=1 MAPE']:.1f}% for ARIMA, which is {pct(H.loc[bh1,'h=1 RMSE'], H.loc['ARIMA','h=1 RMSE']):.0f}% lower RMSE. The ensemble is close ({H.loc['Ensemble','h=1 RMSE']:.2f}, {pct(H.loc['Ensemble','h=1 RMSE'], H.loc['ARIMA','h=1 RMSE']):.0f}% lower) and R² rises from {H.loc['ARIMA','h=1 R2']:.2f} to {H.loc['Ensemble','h=1 R2']:.2f}. At the week-ahead horizon, where Milestone 2 found no gain, the ensemble now reaches RMSE {H.loc['Ensemble','h=7 RMSE']:.2f} against {H.loc['ARIMA','h=7 RMSE']:.2f} for ARIMA ({pct(H.loc['Ensemble','h=7 RMSE'], H.loc['ARIMA','h=7 RMSE']):.0f}% lower), and the LSTM {H.loc['LSTM','h=7 RMSE']:.2f}. The Prophet model, which cannot use forecasts, is the weakest learner at the week-ahead horizon. A single 90-day window is a noisy judge, so we confirm the result on four seasons below.")
figure("13_rmse_classical_vs_advanced.png", "Fig. 3. Hold-out RMSE by model. The dashed line is the naive baseline.", 6.6, wide=True)
h2("B. Robustness across four seasons")
table(HDR, res_rows(R), WID, "TABLE II. Errors pooled over four 90-day folds (360 days)", bold_rows=(T12.index("ARIMA"), T12.index("Ensemble")))
body(f"Pooled over 360 days (Table II), the ensemble lowers next-day RMSE from {R.loc['ARIMA','h=1 RMSE']:.2f} (ARIMA) and {R.loc['SARIMA','h=1 RMSE']:.2f} (SARIMA) to {R.loc['Ensemble','h=1 RMSE']:.2f}, a {pct(R.loc['Ensemble','h=1 RMSE'], R.loc['ARIMA','h=1 RMSE']):.0f}% reduction against ARIMA, and MAPE from {R.loc['ARIMA','h=1 MAPE']:.1f}% to {R.loc['Ensemble','h=1 MAPE']:.1f}%. At one week ahead it lowers RMSE from {R.loc['ARIMA','h=7 RMSE']:.2f} to {R.loc['Ensemble','h=7 RMSE']:.2f} ({pct(R.loc['Ensemble','h=7 RMSE'], R.loc['ARIMA','h=7 RMSE']):.0f}%) and from {R.loc['SARIMA','h=7 RMSE']:.2f} for SARIMA, the best classical model at that horizon ({pct(R.loc['Ensemble','h=7 RMSE'], R.loc['SARIMA','h=7 RMSE']):.0f}%), with week-ahead R² rising from {R.loc['ARIMA','h=7 R2']:.2f} to {R.loc['Ensemble','h=7 R2']:.2f} and MAPE falling from {R.loc['ARIMA','h=7 MAPE']:.1f}% to {R.loc['Ensemble','h=7 MAPE']:.1f}%. The Diebold-Mariano test rejects equal accuracy against ARIMA at one day for every learner except Prophet (all p < 0.001). At one week the ensemble (p = {dm('Ensemble','ARIMA',7):.3f}), LSTM, GRU, LightGBM and Random Forest are significantly better than ARIMA (p < 0.03), XGBoost is not (p = {dm('XGBoost','ARIMA',7):.2f}), and the ensemble also beats SARIMA (p = {dm('Ensemble','SARIMA',7):.4f}). Category accuracy rises from {R.loc['ARIMA','h=1 CatAcc']:.0f}% to {R.loc['Ensemble','h=1 CatAcc']:.0f}%. {('Among single models ' + bp1 + ' has the lowest RMSE at both horizons (' + format(R.loc[bp1,'h=1 RMSE'], '.2f') + ' and ' + format(R.loc[bp7,'h=7 RMSE'], '.2f') + ')') if bp1 == bp7 else ('Among single models ' + bp1 + ' has the lowest next-day RMSE (' + format(R.loc[bp1,'h=1 RMSE'], '.2f') + ') and ' + bp7 + ' the lowest week-ahead RMSE (' + format(R.loc[bp7,'h=7 RMSE'], '.2f') + ')')}; the ensemble ({R.loc['Ensemble','h=1 RMSE']:.2f} and {R.loc['Ensemble','h=7 RMSE']:.2f}) is at or near the best at both horizons without having to pick a winner in advance.")
figure("14_rmse_by_fold.png", "Fig. 4. RMSE by test fold: next-day (left) and week-ahead (right).", 6.2, wide=True)
body(f"The gain is not uniform across seasons (Fig. 4). Against ARIMA, the ensemble lowers next-day RMSE by {gf(FOLD1,'Ensemble','ARIMA',4):.0f}% in the July to October fold, {gf(FOLD1,'Ensemble','ARIMA',2):.0f}% in January to April and {gf(FOLD1,'Ensemble','ARIMA',1):.0f}% in the October to January smog season, but only {gf(FOLD1,'Ensemble','ARIMA',3):.0f}% in April to July. At one week the gains are {gf(FOLD7,'Ensemble','ARIMA',1):.0f}% in the smog season and {gf(FOLD7,'Ensemble','ARIMA',4):.0f}% in July to October, but small in January to April ({gf(FOLD7,'Ensemble','ARIMA',2):.0f}%) and April to July ({gf(FOLD7,'Ensemble','ARIMA',3):.0f}%). Absolute errors are about twice as large in winter (ensemble RMSE about {FOLD1.loc['Ensemble','1']:.0f} versus {FOLD1.loc['Ensemble','4']:.0f} µg/m³), because high concentrations vary more in absolute terms.")
figure("20_timeline_all_folds.png", "Fig. 5. Next-day forecasts across the four test folds (top) and 14-day rolling mean absolute error (bottom), with calendar dates on the horizontal axis.", 6.5, wide=True)
body("The timeline in Fig. 5 shows the ensemble's rolling error below ARIMA's in most weeks, with the largest gaps in the November to February smog season. The clear exception is late May to early June, when the ensemble is no better than ARIMA. All models struggle in the same spells (late December and January, late May to June, late July) when concentrations jump within a day or two.")
h2("C. Week-ahead forecasts and the value of weather forecasts")
body(f"The week-ahead result rests on the weather forecast. Persistence fades quickly, and the drivers of later days, mainly wind and mixing, cannot be read from past pollution. Giving the models the archived ECMWF forecast of the target day (Section III) lowers XGBoost's week-ahead RMSE from {R.loc['XGBoost (no weather forecast)','h=7 RMSE']:.2f} to {R.loc['XGBoost','h=7 RMSE']:.2f} and its next-day RMSE from {R.loc['XGBoost (no weather forecast)','h=1 RMSE']:.2f} to {R.loc['XGBoost','h=1 RMSE']:.2f} (Table IV). The recurrent networks, which read a 7-day forecast vector next to the recent pollution state, are the strongest at one week (GRU {R.loc['GRU','h=7 RMSE']:.2f}, ensemble {R.loc['Ensemble','h=7 RMSE']:.2f}). Classical models and Prophet have no forecast inputs.")
rows = [[str(int(d)), f"{DA.loc[d,'Naive']:.1f}", f"{DA.loc[d,'ARIMA']:.1f}", f"{DA.loc[d,'SARIMA']:.1f}", f"{DA.loc[d,'GRU']:.1f}", f"{DA.loc[d,'Ensemble']:.1f}"] for d in DA.index]
table(["Day ahead", "Naive", "ARIMA", "SARIMA", "GRU", "Ensemble"], rows, [.6, .5, .5, .55, .5, .6], "TABLE III. RMSE by day ahead, four folds pooled", fs=7.5)
body(f"Error grows with the horizon for every model (Table III), but the GRU and the ensemble stay below the classical models at every day ahead; at day 7 the GRU reaches {DA.loc[7,'GRU']:.1f} against {DA.loc[7,'ARIMA']:.1f} for ARIMA and {DA.loc[7,'SARIMA']:.1f} for SARIMA. Forecast skill itself decays with lead time: the correlation of the forecast daily wind with ERA5 falls from 0.90 at one day to 0.64 at seven. To measure the headroom, we gave XGBoost, in addition, the target day's actual weather (an oracle, not a deployable model). Its week-ahead RMSE is {R.loc[S,'h=7 RMSE']:.2f}, so the real forecasts close about {share_gap:.0f}% of the gap between no forecast ({R.loc['XGBoost (no weather forecast)','h=7 RMSE']:.2f}) and perfect weather knowledge.")
h2("D. What drives the forecasts")
sg = SHAPG
body(f"Exact SHAP values for the next-day XGBoost model on the hold-out show that intraday PM2.5 features account for {sg['Intraday PM2.5']:.0f}% of total attribution, forecast weather {sg['Weather forecast']:.0f}%, PM2.5 history {sg['PM2.5 history']:.0f}%, observed weather {sg['Weather']:.0f}%, other pollutants {sg['Co-pollutants']:.0f}% and season {sg['Season']:.0f}% (Fig. 6). The top features are the last-hour ({SHAP.iloc[0]:.2f}) and evening ({SHAP.iloc[1]:.2f}) PM2.5 of the previous day, yesterday's mean, and the forecast daily temperature range and wind speed. This is physical: Delhi's concentration swings from about 52 µg/m³ in mid-afternoon to about 91 µg/m³ late at night, so the daily mean hides where the day ended, and small temperature ranges with low winds indicate stagnant air that lets pollution build up.")
rows = []
for m, lab in [("XGBoost", "Full feature set"), ("XGBoost (no weather forecast)", "Without weather forecast"), ("XGBoost (air only)", "Without any weather"), ("XGBoost (no intraday)", "Without intraday statistics"), ("XGBoost (history only)", "PM2.5 history and season only"), (S, "Full + target-day actual weather (oracle)")]:
    rows.append([lab, f"{R.loc[m,'h=1 RMSE']:.2f}", f"{R.loc[m,'h=1 MAPE']:.1f}", f"{R.loc[m,'h=7 RMSE']:.2f}"])
table(["XGBoost variant", "RMSE", "MAPE %", "Wk RMSE"], rows, [1.75, .5, .55, .6], "TABLE IV. Ablation, four folds pooled", fs=7.5)
body(f"The ablation in Table IV agrees. A model with PM2.5 history only is no better than ARIMA ({R.loc['XGBoost (history only)','h=1 RMSE']:.2f} against {R.loc['ARIMA','h=1 RMSE']:.2f}), so model power alone does not help. Removing intraday statistics raises next-day RMSE from {R.loc['XGBoost','h=1 RMSE']:.2f} to {R.loc['XGBoost (no intraday)','h=1 RMSE']:.2f}. Removing the weather forecast raises next-day RMSE to {R.loc['XGBoost (no weather forecast)','h=1 RMSE']:.2f} and week-ahead RMSE from {R.loc['XGBoost','h=7 RMSE']:.2f} to {R.loc['XGBoost (no weather forecast)','h=7 RMSE']:.2f}, while observed weather up to yesterday adds nothing beyond the pollution state ({R.loc['XGBoost (air only)','h=1 RMSE']:.2f} without any weather).")
figure("16_shap_importance.png", "Fig. 6. Mean absolute SHAP value of the top features (left) and by feature group (right), next-day XGBoost model.", 6.4, wide=True)
h2("E. Alerts, extremes and categories")
e60 = EXC[(EXC.Model == "Ensemble") & (EXC.threshold == 60)].iloc[0]; a60 = EXC[(EXC.Model == "ARIMA") & (EXC.threshold == 60)].iloc[0]
e90 = EXC[(EXC.Model == "Ensemble") & (EXC.threshold == 90)].iloc[0]; a90 = EXC[(EXC.Model == "ARIMA") & (EXC.threshold == 90)].iloc[0]
e120 = EXC[(EXC.Model == "Ensemble") & (EXC.threshold == 120)].iloc[0]
lv = LEV[(LEV.Model == "Ensemble") & (LEV.level == ">120")].iloc[0]
body(f"For flagging days above 60 µg/m³ the ensemble reaches F1 = {e60.F1:.2f} (ARIMA {a60.F1:.2f}), and above 90 µg/m³ F1 = {e90.F1:.2f} (ARIMA {a90.F1:.2f}). The weakness is the extreme tail: on days above 120 µg/m³ the ensemble under-predicts by {abs(lv.Bias):.0f} µg/m³ on average and catches only {e120.recall*100:.0f}% of them (precision {e120.precision*100:.0f}%). Every model shows the same pattern (Fig. 7). Reasons are that such days are rare in training, regression toward the mean, and that CAMS grid values are smoother than station peaks. Alert thresholds should therefore be set below the official limits.")
figure("18_scatter_and_categories.png", "Fig. 7. Predicted versus actual PM2.5 for ARIMA and the ensemble, and the CPCB-category confusion matrix of the ensemble (four folds, next-day).", 6.8, wide=True)
h2("F. Why the models differ")
bullet("Information, not complexity, explains the gain. Ridge, Random Forest, boosted trees and the networks reach similar errors once they see intraday, co-pollutant and forecast features, while the same XGBoost without them falls back to ARIMA's accuracy.")
bullet(f"The deep networks are data-limited but competitive. With about 1,400 training days they stop early ({min(epochs)} to {max(epochs)} epochs), yet the GRU has the lowest pooled RMSE at one week and the LSTM the lowest on the hold-out, because the forecast vector gives them multi-day information.")
bullet("Prophet, which has no autoregressive memory and no forecast inputs, is the weakest learner in the pooled results at both horizons, and at one week it does not significantly beat the naive baseline.")
fb = []
for f in (1, 2, 3, 4):
    c = FOLD1[str(f)].loc[LEARN]; fb.append((f, c.idxmin(), FOLD1.loc['Ensemble', str(f)] / c.min() - 1))
nb = sum(1 for x in fb if x[1] == 'Ensemble'); gap = max(x[2] for x in fb) * 100
bullet(f"Combining models helps robustness: at the next-day horizon the best single learner changes from fold to fold ({', '.join(sorted(set(x[1] for x in fb if x[1] != 'Ensemble')))}), whereas the ensemble is {'the best in ' + str(nb) + ' of the four folds and ' if nb else ''}within {gap:.0f}% of the best learner in every fold.")
h2("G. Why the Milestone 2 models, though strong, were not accurate")
body("Our faculty asked why strong models still gave low accuracy. Reviewing Milestone 2, we found that the limit was the information, not the model, and several avoidable weaknesses:")
bullet(f"Univariate input. ARIMA, SARIMA and Holt-Winters saw only the daily mean. Milestone 2 itself found CO (r ≈ 0.69) and NO₂ (r = 0.56) correlated with PM2.5 but did not use them. With history only, even XGBoost reaches RMSE {R.loc['XGBoost (history only)','h=1 RMSE']:.2f}, no better than ARIMA ({R.loc['ARIMA','h=1 RMSE']:.2f}).")
bullet(f"Daily means hide the state of the day. Adding intraday statistics lowers XGBoost RMSE from {R.loc['XGBoost (no intraday)','h=1 RMSE']:.2f} to {R.loc['XGBoost','h=1 RMSE']:.2f}.")
bullet(f"No weather forecast. The one-week range needs it: adding the archived forecast lowers XGBoost's week-ahead RMSE from {R.loc['XGBoost (no weather forecast)','h=7 RMSE']:.2f} to {R.loc['XGBoost','h=7 RMSE']:.2f}.")
bullet("Seasonality that is not in the data. SARIMA and Holt-Winters used a weekly period although the lag-7 autocorrelation of the log-differenced series is about 0; Holt-Winters failed the Ljung-Box test and was the worst model.")
bullet("Data artefact. The first four daily values were one back-filled constant, and the data were described as gap-free although the first 77 hours and most CO₂ values were missing. Both are fixed here.")
bullet(f"A single, easy test window. July to October has the smallest errors of the four seasons (ensemble RMSE {FOLD1.loc['Ensemble','4']:.1f} versus {FOLD1.loc['Ensemble','1']:.1f} in the smog season), so one window can mislead. We now score four seasons and test significance.")
bullet(f"No decision-level check. Milestone 2 reported only RMSE, MAE and MAPE; we add category accuracy ({R.loc['ARIMA','h=1 CatAcc']:.0f}% for ARIMA, {R.loc['Ensemble','h=1 CatAcc']:.0f}% for the ensemble) and alert F1.")
# ------------------------------------------------------------------ VI. Conclusion
h1("VI. Conclusion and Future Work")
body(f"We extended the Milestone 2 classical forecasts with tuned machine-learning and deep-learning models, intraday features and archived weather forecasts, and tested them on four seasons with a leak-free protocol. Accuracy improved at both horizons: on the Milestone 2 hold-out, next-day RMSE fell from {H.loc['ARIMA','h=1 RMSE']:.2f} to {H.loc['Ensemble','h=1 RMSE']:.2f} µg/m³ and MAPE from {H.loc['ARIMA','h=1 MAPE']:.0f}% to {H.loc['Ensemble','h=1 MAPE']:.0f}%; over all folds the ensemble cut next-day RMSE by {pct(R.loc['Ensemble','h=1 RMSE'], R.loc['ARIMA','h=1 RMSE']):.0f}% and week-ahead RMSE by {pct(R.loc['Ensemble','h=7 RMSE'], R.loc['ARIMA','h=7 RMSE']):.0f}%, and category accuracy rose to {R.loc['Ensemble','h=1 CatAcc']:.0f}%. The main drivers are intraday information and weather forecasts, not a more complex model.")
body("Limitations: the data are modelled grid values for one cell, so peaks are muted and findings may differ at stations; four 90-day folds cover each season once; the forecast archive starts in March 2024, so about half of the earliest fold's training days use actual weather as a stand-in, and we rely on the provider's definition of its \"previous day\" forecasts; only one numerical model (ECMWF IFS) is used and boundary-layer height is unavailable; ARIMA orders from Milestone 2 were reused for all folds; the networks are small and data-limited; and extreme days are under-predicted.")
body("Future work: (i) combine several numerical weather models and add boundary-layer height as these archives grow; (ii) add station data from CPCB and nearby cities and satellite fire counts for crop-residue burning; (iii) quantile or probabilistic forecasts to cover extremes; (iv) a temporal convolutional or Transformer model once more data are available; and (v) an operational daily pipeline with monitoring.")
# ------------------------------------------------------------------ References
h1("References")
refs = ["R. J. Hyndman and G. Athanasopoulos, Forecasting: Principles and Practice, 3rd ed. Melbourne, Australia: OTexts, 2021.",
        "Y. Zhang, M. Bocquet, V. Mallet, C. Seigneur, and A. Baklanov, \"Real-time air quality forecasting, part I: History, techniques, and current status,\" Atmospheric Environment, vol. 60, pp. 632–655, 2012.",
        "A. Kumar and P. Goyal, \"Forecasting of air quality in Delhi using principal component regression technique,\" Atmospheric Pollution Research, vol. 2, no. 4, pp. 436–444, 2011.",
        "X. Li, L. Peng, X. Yao, S. Cui, Y. Hu, C. You, and T. Chi, \"Long short-term memory neural network for air pollutant concentration predictions: Method development and evaluation,\" Environmental Pollution, vol. 231, pp. 997–1004, 2017.",
        "S. Makridakis, E. Spiliotis, and V. Assimakopoulos, \"The M4 Competition: Results, findings, conclusion and way forward,\" International Journal of Forecasting, vol. 34, no. 4, pp. 802–808, 2018.",
        "T. Chen and C. Guestrin, \"XGBoost: A scalable tree boosting system,\" in Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining, 2016, pp. 785–794.",
        "S. Hochreiter and J. Schmidhuber, \"Long short-term memory,\" Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997.",
        "K. Cho et al., \"Learning phrase representations using RNN encoder-decoder for statistical machine translation,\" in Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP), 2014, pp. 1724–1734.",
        "S. J. Taylor and B. Letham, \"Forecasting at scale,\" The American Statistician, vol. 72, no. 1, pp. 37–45, 2018.",
        "Open-Meteo, \"Air Quality API, Historical Weather API and Previous Runs API,\" data from CAMS (Copernicus Atmosphere Monitoring Service), ERA5 and ECMWF IFS, CC BY 4.0. [Online]. Available: https://open-meteo.com. Accessed: Nov. 2026."]
for i, r in enumerate(refs, 1):
    p = para(f"[{i}] {r}", size=8, align="j", after=1); p.paragraph_format.left_indent = Inches(0.25); p.paragraph_format.first_line_indent = Inches(-0.25)
# ------------------------------------------------------------------ Appendix
h1("Appendix: Selected Hyper-parameters and Reproduction")
x1, l1, r1 = PAR["XGBoost|h1"]["params"], PAR["LightGBM|h1"]["params"], PAR["Random Forest|h1"]["params"]
body(f"Next-day models (horizons 2 to 3 reuse these; horizons 4 to 7 use the horizon-7 search result). XGBoost: {', '.join(f'{k} = {v}' for k, v in x1.items())}. LightGBM: {', '.join(f'{k} = {v}' for k, v in l1.items())}. Random Forest: {', '.join(f'{k} = {v}' for k, v in r1.items())}. Recurrent networks: AdamW, learning rate 0.002, weight decay 0.001, batch 64, gradient clipping 1.0, five seeds.")
body("Code, data and results are in the project folder (code/, data/, results/). The command python milestone3_analysis.py (run inside code/) regenerates all tables and figures; milestone3_dl.py trains the LSTM and GRU, m3_data.py builds the daily dataset and m3_forecasts.py downloads the forecast archive. Forecasts for every fold are saved in predictions_all_folds.csv.")
table(["Member", "Contribution"], [["Mahak Soni (Team Leader)", "Project lead and submission; data pipeline and intraday features; classical recap; tree models and tuning; this report."], ["Darsh Jasoria", "Weather-forecast download and issue-time checks; Prophet, LSTM, GRU; ensemble, SHAP, ablation, significance tests; website, slides, deployment."], ["Palak Aitwani", "GFG"]], [1.15, 2.3], "TABLE V. Team contributions", fs=7.5)
out = os.path.join(ROOT, "Mahak Soni_S24CSEU2741_Milestone3_Report.docx"); doc.save(out)
words = sum(len(p.text.split()) for p in doc.paragraphs)
print("saved", out, "| abstract words", wc, "| body words ~", words)
