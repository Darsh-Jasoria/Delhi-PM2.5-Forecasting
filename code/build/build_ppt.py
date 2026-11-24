"""Builds the Milestone 3 final presentation (PPTX) in the Milestone 2 design language. Numbers are read from ../results/*.csv."""
import os, json
import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, "results"); FIG = os.path.join(RES, "figures")
H = pd.read_csv(f"{RES}/model_comparison_m3_holdout.csv").set_index("Model")
R = pd.read_csv(f"{RES}/model_comparison_m3_rolling.csv").set_index("Model")
F1 = pd.read_csv(f"{RES}/rmse_by_fold_h1.csv").set_index("Model")
SHAPG = pd.read_csv(f"{RES}/shap_group_share_h1.csv", index_col=0)["share_pct"]
EXC = pd.read_csv(f"{RES}/exceedance_detection_h1.csv"); LEV = pd.read_csv(f"{RES}/error_by_pm25_level_h1.csv")
PRED = pd.read_csv(f"{RES}/predictions_all_folds.csv", parse_dates=["date"])
DLP = json.load(open(f"{RES}/dl_params.json")); PAR = json.load(open(f"{RES}/best_params.json"))
pct = lambda new, old: (1 - new / old) * 100
S = "XGBoost (+future weather, oracle)"
F7 = pd.read_csv(f"{RES}/rmse_by_fold_h7.csv").set_index("Model")
DMX = pd.read_csv(f"{RES}/diebold_mariano.csv")
dmp = lambda m, vs, h: DMX[(DMX.Model == m) & (DMX.vs == vs) & (DMX.horizon == h)].p_value.iloc[0]
FACULTY = "Anurag Choubey"
rgb = lambda h: RGBColor.from_string(h)
DARK, LIGHT, INK, MUT, TEAL, MINT, AMBER, ORANGE, BLUE, RED, SOFT, LINE, WHITE = (rgb(c) for c in ["123B45", "F3F7F6", "1E3A40", "5B7075", "1F8A7D", "6FD6C8", "F2B84B", "C4801A", "2F7BA6", "D1503F", "E8F3F1", "DCE6E4", "FFFFFF"])

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tempfile
SLIDE_DIR = tempfile.mkdtemp(prefix='m3slides_')       # slide-sized charts are build artefacts: drawn here, embedded in the deck, not stored in the repo
def slide_figs():
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25, "font.size": 8.5, "figure.dpi": 200})
    TE, OR, NA, GR, RD = "#1F8A7D", "#C4801A", "#27476E", "#5B7075", "#D1503F"
    F7_ = pd.read_csv(f"{RES}/rmse_by_fold_h7.csv").set_index("Model"); F1_ = pd.read_csv(f"{RES}/rmse_by_fold_h1.csv").set_index("Model"); SH = pd.read_csv(f"{RES}/shap_importance_h1.csv", index_col=0)["mean_abs_shap"]; SG = pd.read_csv(f"{RES}/shap_group_share_h1.csv", index_col=0)["share_pct"]
    fig, ax = plt.subplots(figsize=(12.1, 2.3)); ax.axis("off"); ax.grid(False)
    boxes = ["Open-Meteo\nCAMS air\nERA5 weather\nECMWF forecasts", "Daily table\n+ intraday stats\n(evening, max,\nmorning)", "79 features\nlags, intraday,\nweather,\nforecast", "4 rolling folds\n90 days each\nexpanding\ntraining window", "Tune on data\nbefore fold 1\ntrees, LSTM,\nGRU grid", "12 models\none per horizon\n1 to 7 days", "Score\nRMSE, MAE, MAPE\nR², category,\nDiebold-Mariano", "Insights\nSHAP, ablation,\nerrors by\nregime"]
    for i, b in enumerate(boxes):
        ax.add_patch(plt.Rectangle((i + .04, .06), .88, .88, color=[TE, TE, OR, OR, NA, NA, "#7A5C9E", "#7A5C9E"][i], alpha=.92)); ax.text(i + .48, .5, b, ha="center", va="center", color="white", fontsize=8.6, weight="bold", linespacing=1.25)
        if i < 7: ax.annotate("", xy=(i + 1.04, .5), xytext=(i + .93, .5), arrowprops=dict(arrowstyle="->", color="black", lw=1.3))
    ax.set_xlim(0, 8.05); ax.set_ylim(0, 1); fig.subplots_adjust(left=.005, right=.995, top=.99, bottom=.01); fig.savefig(os.path.join(SLIDE_DIR, "s09_workflow.png")); plt.close(fig)
    ms = ["Naive", "ARIMA", "SARIMA", "Ridge", "Random Forest", "XGBoost", "LightGBM", "Prophet", "LSTM", "GRU", "Ensemble"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4)); y = np.arange(len(ms))[::-1]
    b1 = ax.barh(y + .2, [H.loc[m, "h=1 RMSE"] for m in ms], .38, color=TE, label="Next day"); b2 = ax.barh(y - .2, [H.loc[m, "h=7 RMSE"] for m in ms], .38, color=OR, label="Week ahead")
    for yy, m in zip(y, ms): ax.text(H.loc[m, "h=1 RMSE"] + .3, yy + .2, f"{H.loc[m,'h=1 RMSE']:.1f}", va="center", fontsize=7); ax.text(H.loc[m, "h=7 RMSE"] + .3, yy - .2, f"{H.loc[m,'h=7 RMSE']:.1f}", va="center", fontsize=7)
    ax.set_yticks(y, ms); ax.set_xlabel("RMSE (µg/m³), Milestone 2 hold-out"); ax.legend(loc="lower right", fontsize=8); ax.set_xlim(0, 30); ax.grid(axis="y", alpha=0); ax.set_title("Lower is better", fontsize=9, weight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(SLIDE_DIR, "s11_holdout.png")); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.5)); xl = ["Oct-Jan\nsmog", "Jan-Apr", "Apr-Jul", "Jul-Oct"]
    for a_, T, t in ((ax[0], F1_, "Next day"), (ax[1], F7_, "Week ahead")):
        for m, c, lw in (("Naive", "#9AA5A8", 1.2), ("ARIMA", GR, 1.4), ("XGBoost", OR, 1.4), ("GRU", "#3C78B4", 1.4), ("Ensemble", TE, 2.4)): a_.plot(range(4), [T.loc[m, str(f)] for f in (1, 2, 3, 4)], marker="o", ms=3.5, color=c, lw=lw, label=m)
        a_.set_xticks(range(4), xl, fontsize=7.5); a_.set_title(t, fontsize=9, weight="bold"); a_.set_ylabel("RMSE (µg/m³)", fontsize=8)
    ax[0].legend(fontsize=7, loc="upper right"); fig.tight_layout(); fig.savefig(os.path.join(SLIDE_DIR, "s12_folds.png")); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(12.0, 3.0), gridspec_kw=dict(width_ratios=[1.5, 1])); top = SH.head(12)[::-1]
    ax[0].barh(top.index, top.values, color=TE); ax[0].set_title("Top features, next-day XGBoost (mean |SHAP|)", fontsize=8.5, weight="bold"); ax[0].tick_params(labelsize=7.5); ax[0].grid(axis="y", alpha=0)
    ax[1].barh(SG.index[::-1], SG.values[::-1], color=OR); ax[1].set_title("Share by feature group (%)", fontsize=8.5, weight="bold"); ax[1].tick_params(labelsize=7.5); ax[1].grid(axis="y", alpha=0)
    for i, v in enumerate(SG.values[::-1]): ax[1].text(v + .5, i, f"{v:.0f}", va="center", fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(SLIDE_DIR, "s17_shap.png")); plt.close(fig)
slide_figs()
prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]; counter = [0]
SERIF, SANS = "Georgia", "Calibri"

def tb(slide, x, y, w, h, text="", size=16, bold=False, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, font=SANS, spc=None, italic=False):
    s = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = s.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.04); tf.margin_top = tf.margin_bottom = Inches(0.02)
    p = tf.paragraphs[0]; p.alignment = align; r = p.add_run(); r.text = text; f = r.font; f.size = Pt(size); f.bold = bold; f.italic = italic; f.color.rgb = color; f.name = font
    if spc: r._r.get_or_add_rPr().set("spc", str(spc))
    return s
def rect(slide, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE, radius=None, lw=0.75):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h)); s.fill.solid(); s.fill.fore_color.rgb = fill
    if line is None: s.line.fill.background()
    else: s.line.color.rgb = line; s.line.width = Pt(lw)
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE: s.adjustments[0] = radius
    s.shadow.inherit = False; return s
def bullets(slide, x, y, w, h, items, size=14, color=INK, gap=5, bullet_color=TEAL):
    s = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = s.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.04)
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.space_after = Pt(gap)
        bold_part, rest = it if isinstance(it, tuple) else ("", it)
        r0 = p.add_run(); r0.text = "•  "; r0.font.size = Pt(size); r0.font.color.rgb = bullet_color; r0.font.bold = True; r0.font.name = SANS
        if bold_part: rb = p.add_run(); rb.text = bold_part + " "; rb.font.bold = True; rb.font.size = Pt(size); rb.font.color.rgb = color; rb.font.name = SANS
        r = p.add_run(); r.text = rest; r.font.size = Pt(size); r.font.color.rgb = color; r.font.name = SANS
    return s
def slide(tag, title, takeaway=None):
    s = prs.slides.add_slide(BLANK); counter[0] += 1
    s.background.fill.solid(); s.background.fill.fore_color.rgb = LIGHT
    tb(s, 0.6, 0.30, 9, 0.28, tag.upper(), 11, True, TEAL, spc=300)
    tb(s, 0.6, 0.58, 12.1, 0.72, title, 28, True, INK, font=SERIF, anchor=MSO_ANCHOR.MIDDLE)
    tb(s, 12.1, 7.02, 0.63, 0.3, str(counter[0]), 10.5, False, MUT, PP_ALIGN.RIGHT)
    tb(s, 0.6, 7.02, 9, 0.3, f"Time Series Analysis (CSET 369)   |   Milestone 3   |   Faculty: {FACULTY}", 9.5, False, MUT)
    if takeaway:
        rect(s, 0.6, 6.32, 12.13, 0.56, SOFT); rect(s, 0.6, 6.32, 0.08, 0.56, TEAL)
        tb(s, 0.85, 6.32, 11.7, 0.56, takeaway, 14, True, INK, anchor=MSO_ANCHOR.MIDDLE)
    return s
def panel(s, x, y, w, h, heading, items=None, size=13.5, gap=5):
    rect(s, x, y, w, h, WHITE, line=LINE)
    if heading: tb(s, x + 0.22, y + 0.14, w - 0.4, 0.28, heading.upper(), 10.5, True, TEAL, spc=200)
    if items: bullets(s, x + 0.22, y + 0.5, w - 0.4, h - 0.6, items, size, gap=gap)
def image(s, name, x, y, w, h, card=True, pad=0.1):
    if card: rect(s, x, y, w, h, WHITE, line=LINE)
    path = name if os.path.isabs(name) else os.path.join(FIG, name); iw, ih = Image.open(path).size; r = min((w - 2 * pad) / iw, (h - 2 * pad) / ih); pw, ph = iw * r, ih * r
    s.shapes.add_picture(path, Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))
def table(s, x, y, w, header, rows, colw, size=12, hl=(), rowh=0.34, left=()):
    t = s.shapes.add_table(len(rows) + 1, len(header), Inches(x), Inches(y), Inches(w), Inches(rowh * (len(rows) + 1))).table
    for j, cw in enumerate(colw): t.columns[j].width = Inches(cw)
    for j, htxt in enumerate(header):
        c = t.cell(0, j); c.text = htxt; c.fill.solid(); c.fill.fore_color.rgb = DARK; c.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.LEFT if (j == 0 or j in left) else PP_ALIGN.CENTER; p.runs[0].font.size = Pt(size); p.runs[0].font.bold = True; p.runs[0].font.color.rgb = WHITE; p.runs[0].font.name = SANS
    for i, row in enumerate(rows, 1):
        for j, v in enumerate(row):
            c = t.cell(i, j); c.text = str(v); c.fill.solid(); c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.fore_color.rgb = SOFT if (i - 1) in hl else (WHITE if i % 2 else rgb("F7FAF9"))
            p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.LEFT if (j == 0 or j in left) else PP_ALIGN.CENTER
            if not p.runs: p.add_run().text = " "
            f = p.runs[0].font; f.size = Pt(size); f.color.rgb = INK; f.bold = (i - 1) in hl; f.name = SANS
        t.rows[i].height = Inches(rowh)
    t.rows[0].height = Inches(rowh); return t
def card(s, x, y, w, big, small, color=TEAL, h=1.45):
    rect(s, x, y, w, h, WHITE, line=LINE); rect(s, x, y, 0.09, h, color)
    tb(s, x + 0.25, y + 0.1, w - 0.35, 0.7, big, 30, True, color, font=SERIF); tb(s, x + 0.25, y + 0.85, w - 0.35, h - 0.85, small, 12, False, MUT)

# ---------------------------------------------------------------- 1 title (dark, M2 style)
s = prs.slides.add_slide(BLANK); counter[0] += 1
s.background.fill.solid(); s.background.fill.fore_color.rgb = DARK
tb(s, 0.6, 0.62, 8.5, 0.3, "TIME SERIES & ANALYSIS   |   MILESTONE 3   |   FINAL EVALUATION", 12, True, MINT, spc=300)
tb(s, 0.6, 1.1, 7.9, 2.3, "Forecasting Delhi's PM2.5: from Classical Models to Machine and Deep Learning", 38, True, WHITE, font=SERIF)
tb(s, 0.6, 3.55, 7.6, 0.9, "Why strong models stayed inaccurate, and how intraday features, archived weather forecasts and tuned ML / DL models fixed it", 16, False, rgb("C9DEDB"), font=SERIF)
for i, (n, r_) in enumerate([("Mahak Soni", "S24CSEU2741"), ("Darsh Jasoria", "S24CSEU0844"), ("Palak Aitwani", "S24CSEU0326")]):
    x = 0.6 + i * 2.6; rect(s, x, 4.75, 2.45, 1.3, rgb("24505A"), line=rgb("3F6A73"), shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
    tb(s, x + 0.18, 4.87, 2.1, 0.35, n, 15, True, WHITE, font=SERIF); tb(s, x + 0.18, 5.27, 2.1, 0.3, r_, 12.5, True, MINT); tb(s, x + 0.18, 5.62, 2.2, 0.3, ("Team Leader  |  Year 3" if i == 0 else "B.Tech CSE  |  Year 3"), 11, False, rgb("C9DEDB"))
tb(s, 0.6, 6.25, 8.0, 0.3, f"Faculty:  {FACULTY}", 13.5, True, MINT)
tb(s, 0.6, 6.62, 8.0, 0.3, "Bennett University   |   November 2026   |   Phase 3 final evaluation", 11.5, False, rgb("C9DEDB"))
last = PRED[(PRED.fold == 4) & (PRED.horizon == 1)].sort_values("date").tail(36)
mx = last.actual.max(); bx0 = 8.75
for i, (v) in enumerate(last.actual):
    hh = 3.5 * v / mx; rect(s, bx0 + i * 0.1135, 6.05 - hh, 0.085, hh, AMBER if v > 90 else MINT)
tb(s, 8.7, 6.15, 4.2, 0.5, "Real data: daily PM2.5 in Delhi, last 36 days (15 Sep to 20 Oct 2026). Amber = above 90 µg/m³.", 10, False, rgb("C9DEDB"))

# ---------------------------------------------------------------- 2 agenda
s = slide("Agenda", "What we will cover", "Eight parts, in the order of the final-evaluation guidelines.")
items = [("1", "Problem overview"), ("2", "Dataset summary"), ("3", "Phase 2 recap and diagnosis"), ("4", "Advanced models used"), ("5", "Comparative results"), ("6", "Key insights"), ("7", "Challenges and limitations"), ("8", "Conclusion and future work")]
for i, (n, t_) in enumerate(items):
    x, y = 0.6 + (i % 4) * 3.07, 1.55 + (i // 4) * 2.2; rect(s, x, y, 2.9, 1.95, WHITE, line=LINE); rect(s, x, y, 2.9, 0.08, TEAL if i % 2 == 0 else ORANGE)
    tb(s, x + 0.25, y + 0.3, 1, 0.7, n, 34, True, TEAL if i % 2 == 0 else ORANGE, font=SERIF); tb(s, x + 0.25, y + 1.1, 2.5, 0.75, t_, 15, True, INK)

# ---------------------------------------------------------------- 3 summary
s = slide("Summary at a glance", "Better accuracy at both horizons", "Gains are statistically significant; archived weather forecasts are what unlocked the week ahead.")
card(s, 0.6, 1.5, 2.9, f"−{pct(H.loc['Ensemble','h=1 RMSE'], H.loc['ARIMA','h=1 RMSE']):.0f}%", f"Next-day RMSE, M2 hold-out: {H.loc['ARIMA','h=1 RMSE']:.2f} → {H.loc['Ensemble','h=1 RMSE']:.2f} µg/m³", TEAL, h=1.6)
card(s, 3.65, 1.5, 2.9, f"−{pct(R.loc['Ensemble','h=1 RMSE'], R.loc['ARIMA','h=1 RMSE']):.0f}%", f"Next-day RMSE, four seasons: {R.loc['ARIMA','h=1 RMSE']:.1f} → {R.loc['Ensemble','h=1 RMSE']:.1f} (p < 0.0001)", BLUE, h=1.6)
card(s, 6.7, 1.5, 2.9, f"−{pct(R.loc['Ensemble','h=7 RMSE'], R.loc['ARIMA','h=7 RMSE']):.0f}%", f"Week-ahead RMSE, four seasons: {R.loc['ARIMA','h=7 RMSE']:.1f} → {R.loc['Ensemble','h=7 RMSE']:.1f} (p = {dmp('Ensemble','ARIMA',7):.3f})", ORANGE, h=1.6)
card(s, 9.75, 1.5, 2.98, f"{R.loc['ARIMA','h=1 CatAcc']:.0f}% to {R.loc['Ensemble','h=1 CatAcc']:.0f}%", "Correct CPCB pollution category, next day", rgb("7A5C9E"), h=1.6)
panel(s, 0.6, 3.25, 6.0, 2.9, "What we did", ["Re-ran the M2 classical models: identical numbers.", "Added intraday statistics, co-pollutants, weather and archived ECMWF forecasts: 79 features.", "Tuned Ridge, Random Forest, XGBoost, LightGBM, Prophet, LSTM, GRU and an ensemble.", "Scored four 90-day seasons with significance tests."], 13)
panel(s, 6.75, 3.25, 5.98, 2.9, "What we found", ["The M2 limit was information, not model power.", "Last-hour and evening PM2.5 plus forecast wind and temperature range carry the signal.", f"Weather forecasts make the week ahead work: without them XGBoost's week-ahead error rises from {R.loc['XGBoost','h=7 RMSE']:.1f} to {R.loc['XGBoost (no weather forecast)','h=7 RMSE']:.1f}.", "Extreme days (> 120 µg/m³) are still under-predicted."], 13)

# ---------------------------------------------------------------- 4 problem
s = slide("1  Problem overview", "Delhi's PM2.5 forecasting problem", "Alerts need a forecast a day ahead; M2's best model was only 5% better than repeating yesterday.")
panel(s, 0.6, 1.5, 7.3, 4.65, "Why it matters", [("Health burden.", "PM2.5 exceeds India's 24-hour standard (60 µg/m³) on 56% of days in our data."), ("Decisions are time-bound.", "School closures, traffic limits and GRAP stages need a forecast a day or more ahead."),
                                 ("Milestone 2 result.", f"ARIMA next-day RMSE {H.loc['ARIMA','h=1 RMSE']:.2f} against {H.loc['Naive','h=1 RMSE']:.2f} for 'same as yesterday'."), ("Question from our faculty.", "Why is accuracy still low with strong models?"), ("Our hypothesis.", "The models saw one number per day, not how the day ended, the weather forecast, or other pollutants.")], 14, gap=7)
card(s, 8.15, 1.5, 4.58, f"{H.loc['ARIMA','h=1 RMSE']:.1f}", "ARIMA next-day RMSE (µg/m³), Milestone 2", MUT, h=1.4)
card(s, 8.15, 3.05, 4.58, "56%", "of days above the 60 µg/m³ standard", RED, h=1.4)
card(s, 8.15, 4.6, 4.58, "5", "objectives: implement, tune, compare, explain, state limits", BLUE, h=1.55)

# ---------------------------------------------------------------- 5 dataset
s = slide("2  Dataset summary", "Delhi air quality, weather and forecasts, 2022 to 2026", "1,538 days built from 37,008 hourly rows; forecasts respect the issue time, so nothing from the future is used.")
table(s, 0.6, 1.5, 6.0, ["Item", "Detail"], [["Air quality", "Open-Meteo / CAMS, hourly"], ["Weather", "Open-Meteo / ERA5, hourly"], ["Weather forecasts", "ECMWF IFS via Previous Runs API; issued 1 to 7 days ahead; from Mar 2024"], ["Location", "28.60°N 77.20°E, New Delhi"], ["Period", "5 Aug 2022 to 20 Oct 2026 (1,538 days)"], ["Target", "Daily mean PM2.5 (log scale)"], ["Inputs", "PM2.5 + intraday stats, PM10, NO₂, CO, AQI, 9 weather + 9 forecast variables"], ["Licence", "CC BY 4.0"]], [1.7, 4.3], 11.5, rowh=0.44, left=(1,))
panel(s, 0.6, 5.68, 6.0, 0.58, "", None); bullets(s, 0.75, 5.74, 5.8, 0.5, [("Fixes:", "dropped back-filled days; CO₂ left out (725 days only).")], 12, gap=2)
image(s, "19_data_overview.png", 6.8, 1.5, 5.93, 4.65)

# ---------------------------------------------------------------- 6 recap
s = slide("3  Phase 2 recap", "Classical models from Milestone 2", "Classical models gave only a modest gain over persistence, and none at one week.")
rows = [[m, f"{H.loc[m,'h=1 RMSE']:.2f}", f"{H.loc[m,'h=1 MAE']:.2f}", f"{H.loc[m,'h=1 MAPE']:.1f}", f"{H.loc[m,'h=7 RMSE']:.2f}"] for m in ["Naive", "ARIMA", "SARIMA", "Holt-Winters"]]
table(s, 0.6, 1.5, 7.3, ["Model (M2 hold-out)", "RMSE", "MAE", "MAPE %", "Week RMSE"], rows, [2.6, 1.0, 1.0, 1.2, 1.5], 14, hl=(1,), rowh=0.55)
panel(s, 0.6, 4.4, 7.3, 1.75, "Re-run check", ["ARIMA(3,0,3) chosen by AIC; residuals pass Ljung-Box.", "We re-ran these models inside M3: identical numbers, so comparisons are like for like."], 13)
panel(s, 8.15, 1.5, 4.58, 4.65, "Why accuracy stayed low", ["One daily mean hides where the day ended.", "No weather forecast, no other pollutants.", "Weekly terms with no weekly signal.", "A single 90-day window: easy to over-read.", "Linear, short-memory structure."], 15, gap=14)

# ---------------------------------------------------------------- 7 diagnosis
s = slide("3  Phase 2 diagnosis", "Why strong models were not accurate in Milestone 2", "The limit was information, not model power: a stronger model on the same single series gives the same accuracy.")
rows = [["Univariate input", f"CO (r ≈ 0.69) and NO₂ (0.56) found in M2 but unused. History-only XGBoost RMSE {R.loc['XGBoost (history only)','h=1 RMSE']:.1f} vs ARIMA {R.loc['ARIMA','h=1 RMSE']:.1f}", "Co-pollutants + weather + forecasts"],
        ["Daily mean hides the day's state", f"Without intraday features XGBoost RMSE {R.loc['XGBoost (no intraday)','h=1 RMSE']:.1f}; with them {R.loc['XGBoost','h=1 RMSE']:.1f}", "5 intraday statistics from hourly data"],
        ["No weather forecast", f"Week-ahead XGBoost RMSE {R.loc['XGBoost (no weather forecast)','h=7 RMSE']:.1f} without forecasts, {R.loc['XGBoost','h=7 RMSE']:.1f} with them", "Archived ECMWF forecasts, issue-time safe"],
        ["Weekly terms, no weekly signal", "Lag-7 autocorrelation ≈ 0, yet SARIMA / Holt-Winters used m = 7; Holt-Winters failed Ljung-Box", "Weekly seasonality dropped"],
        ["Data artefact", "First 4 daily values one back-filled constant; 'gap-free' claim wrong (77 h, 817 CO₂ days missing)", "Removed; gaps documented"],
        ["One easy test window", f"Jul-Oct RMSE {F1.loc['Ensemble','4']:.1f} vs smog season {F1.loc['Ensemble','1']:.1f}: one window misleads", "4 rolling folds + Diebold-Mariano"],
        ["No decision metric", f"Only RMSE / MAE / MAPE; category accuracy {R.loc['ARIMA','h=1 CatAcc']:.0f}% for ARIMA", "Category accuracy, alert F1, tail bias"]]
table(s, 0.6, 1.45, 12.13, ["Issue in Milestone 2", "Evidence", "Fix in Milestone 3"], rows, [2.8, 6.2, 3.13], 11, rowh=0.55, left=(1, 2))

# ---------------------------------------------------------------- 8 what changed
s = slide("Approach", "What we changed to improve accuracy", "Six changes; better inputs, intraday statistics and weather forecasts, produced most of the gain.")
for i, (t, d, c) in enumerate([("1  Intraday features", "Mean, max, min, last-hour, evening and morning PM2.5 for each day.", TEAL), ("2  Weather forecasts", "Archived ECMWF forecast of the target day, issued h days earlier: no look-ahead.", ORANGE),
                               ("3  Richer inputs", "Lagged PM10, NO₂, CO, AQI plus observed temperature, humidity, pressure, wind, rain, cloud.", BLUE), ("4  Tuned learners", "XGBoost, LightGBM, Random Forest, Ridge, Prophet, LSTM, GRU, ensemble.", TEAL),
                               ("5  Honest validation", "Four rolling 90-day folds; tuning only on data before fold 1.", ORANGE), ("6  Significance tests", "Diebold-Mariano against Naive, ARIMA and SARIMA.", BLUE)]):
    x, y = 0.6 + (i % 3) * 4.07, 1.5 + (i // 3) * 2.4; rect(s, x, y, 3.9, 2.2, WHITE, line=LINE); rect(s, x, y, 3.9, 0.09, c)
    tb(s, x + 0.22, y + 0.25, 3.5, 0.5, t, 17, True, c, font=SERIF); tb(s, x + 0.22, y + 0.85, 3.5, 1.3, d, 13.5, False, INK)

# ---------------------------------------------------------------- 9 workflow
s = slide("4  Methodology", "Workflow and evaluation design", "Same protocol as Milestone 2, plus three earlier folds so every season is tested once.")
image(s, os.path.join(SLIDE_DIR, "s09_workflow.png"), 0.6, 1.5, 12.13, 2.35, pad=0.1)
panel(s, 0.6, 4.0, 5.95, 2.15, "Targets and features", [("Targets:", "next-day (h = 1) and week-ahead (days 1 to 7, every 7 days)."), ("Features:", "79 tabular; 21-variable window + 7-day forecast vector for LSTM / GRU.")], 12.5, gap=4)
panel(s, 6.78, 4.0, 5.95, 2.15, "Validation and metrics", [("Folds:", "Oct-Jan, Jan-Apr, Apr-Jul, Jul-Oct; fold 4 = M2 hold-out."), ("Metrics:", "RMSE, MAE, MAPE, R², category accuracy, Diebold-Mariano.")], 12.5, gap=4)

# ---------------------------------------------------------------- 10 models
s = slide("4  Advanced models", "Models used and how they were tuned", "All tuning used only data before the first test fold, so no test data leaked into model choice.")
x1, l1, r1 = PAR["XGBoost|h1"]["params"], PAR["LightGBM|h1"]["params"], PAR["Random Forest|h1"]["params"]
rows = [["Ridge", "Linear, penalty by time-ordered CV", "Strong, simple benchmark"], ["Random Forest", f"300 trees; leaf {r1['min_samples_leaf']}, depth {r1['max_depth']}", "Robust to noise"], ["XGBoost", f"depth {x1['max_depth']}, lr {x1['learning_rate']}, {x1['n_estimators']} trees", "Non-linear interactions"],
        ["LightGBM", f"{l1['num_leaves']} leaves, lr {l1['learning_rate']}, {l1['n_estimators']} trees", "Fast boosted trees"], ["Prophet", "Yearly + weekly, 10 lagged regressors (no forecasts)", "Trend + season baseline"],
        ["LSTM", f"hidden {DLP['lstm']['params']['hidden']}, window {DLP['lstm']['params']['window']}, dropout {DLP['lstm']['params']['dropout']}", "Sequence memory"], ["GRU", f"hidden {DLP['gru']['params']['hidden']}, window {DLP['gru']['params']['window']}, dropout {DLP['gru']['params']['dropout']}", "Lighter recurrent unit"],
        ["Ensemble", "Mean of XGB, LGBM, RF, LSTM, GRU (fixed in advance)", "Variance reduction"]]
table(s, 0.6, 1.5, 12.13, ["Model", "Setting (selected by tuning)", "Why it is here"], rows, [2.0, 6.2, 3.93], 12.5, hl=(7,), rowh=0.4, left=(1, 2))
tb(s, 0.6, 5.65, 12.13, 0.6, "Trees: randomised search, 16 candidates, time-ordered CV. LSTM / GRU: grid of hidden size, window, dropout; five seeds averaged; early stopping on the last 60 training days. Classical models and Prophet use no forecast inputs.", 12, False, MUT)

# ---------------------------------------------------------------- 11 hold-out
s = slide("5  Comparative results", "Milestone 2 hold-out: classical versus advanced", f"On the identical test window the ensemble cuts next-day RMSE by {pct(H.loc['Ensemble','h=1 RMSE'], H.loc['ARIMA','h=1 RMSE']):.0f}% and week-ahead RMSE by {pct(H.loc['Ensemble','h=7 RMSE'], H.loc['ARIMA','h=7 RMSE']):.0f}%.")
T11 = ["Naive", "ARIMA", "SARIMA", "Ridge", "Random Forest", "XGBoost", "LightGBM", "Prophet", "LSTM", "GRU", "Ensemble"]
rows = [[m, f"{H.loc[m,'h=1 RMSE']:.2f}", f"{H.loc[m,'h=1 MAE']:.2f}", f"{H.loc[m,'h=1 MAPE']:.1f}", f"{H.loc[m,'h=1 R2']:.2f}", f"{H.loc[m,'h=1 CatAcc']:.0f}", f"{H.loc[m,'h=7 RMSE']:.2f}"] for m in T11]
table(s, 0.6, 1.5, 6.9, ["Model", "RMSE", "MAE", "MAPE %", "R²", "Cat. %", "Week RMSE"], rows, [1.7, 0.8, 0.8, 0.9, 0.7, 0.8, 1.2], 11.5, hl=(1, 10), rowh=0.37)
image(s, os.path.join(SLIDE_DIR, "s11_holdout.png"), 7.7, 1.5, 5.03, 3.2)
panel(s, 7.7, 4.8, 5.03, 1.4, "", None); bullets(s, 7.85, 4.9, 4.8, 1.3, [f"Next-day R²: {H.loc['ARIMA','h=1 R2']:.2f} → {H.loc['Ensemble','h=1 R2']:.2f}.", f"Best single models: LightGBM next day ({H.loc['LightGBM','h=1 RMSE']:.2f}), LSTM week ahead ({H.loc['LSTM','h=7 RMSE']:.2f})."], 12, gap=3)

# ---------------------------------------------------------------- 12 rolling
s = slide("5  Comparative results", "Robust across four seasons (360 test days)", f"Ensemble vs ARIMA: next-day RMSE −{pct(R.loc['Ensemble','h=1 RMSE'], R.loc['ARIMA','h=1 RMSE']):.0f}%, week-ahead −{pct(R.loc['Ensemble','h=7 RMSE'], R.loc['ARIMA','h=7 RMSE']):.0f}%, both significant (Diebold-Mariano p < 0.01).")
rows = [[m, f"{R.loc[m,'h=1 RMSE']:.2f}", f"{R.loc[m,'h=1 MAPE']:.1f}", f"{R.loc[m,'h=1 CatAcc']:.0f}", f"{R.loc[m,'h=7 RMSE']:.2f}", f"{R.loc[m,'h=7 R2']:.2f}"] for m in ["Naive", "ARIMA", "SARIMA", "Ridge", "Random Forest", "XGBoost", "LightGBM", "LSTM", "GRU", "Ensemble"]]
table(s, 0.6, 1.5, 5.9, ["Model (pooled)", "RMSE", "MAPE %", "Cat. %", "Week RMSE", "Week R²"], rows, [1.7, 0.8, 0.85, 0.75, 0.95, 0.85], 12, hl=(1, 9), rowh=0.4)
image(s, os.path.join(SLIDE_DIR, "s12_folds.png"), 6.7, 1.5, 6.03, 2.45)
g1 = [pct(F1.loc['Ensemble', str(f)], F1.loc['ARIMA', str(f)]) for f in (1, 2, 3, 4)]; g7 = [pct(F7.loc['Ensemble', str(f)], F7.loc['ARIMA', str(f)]) for f in (1, 2, 3, 4)]
panel(s, 6.7, 4.05, 6.03, 2.1, "Reading the folds", [f"Next day: gains of {g1[0]:.0f}%, {g1[1]:.0f}%, {g1[2]:.0f}%, {g1[3]:.0f}% in folds 1 to 4 (smallest in Apr-Jul).", f"Week ahead: {g7[0]:.0f}%, {g7[1]:.0f}%, {g7[2]:.0f}%, {g7[3]:.0f}%: large in the smog season and Jul-Oct, small in between."], 12.5, gap=4)

# ---------------------------------------------------------------- 13 ablation
s = slide("5  Comparative results", "What each input adds (ablation)", f"Removing the weather forecast raises XGBoost's week-ahead error from {R.loc['XGBoost','h=7 RMSE']:.1f} to {R.loc['XGBoost (no weather forecast)','h=7 RMSE']:.1f}; PM2.5 history alone is no better than ARIMA.")
ABL13 = [("XGBoost", "Final model: all inputs"), ("XGBoost (no weather forecast)", "Without the weather forecast"), ("XGBoost (air only)", "Without any weather"), ("XGBoost (no intraday)", "Without intraday statistics"), ("XGBoost (history only)", "PM2.5 history and season only"), (S, "All inputs + actual weather (oracle)")]
chg = lambda m, c: ("" if m == "XGBoost" else f"{(R.loc[m, c] / R.loc['XGBoost', c] - 1) * 100:+.0f}%")
rows = [[lab, f"{R.loc[m,'h=1 RMSE']:.2f}", chg(m, 'h=1 RMSE'), f"{R.loc[m,'h=7 RMSE']:.2f}", chg(m, 'h=7 RMSE')] for m, lab in ABL13]
table(s, 0.6, 1.5, 7.6, ["XGBoost variant (four folds pooled)", "Tomorrow", "change", "Next 7 days", "change"], rows, [3.1, 1.1, 0.9, 1.3, 1.2], 12, hl=(0,), rowh=0.5, left=())
panel(s, 8.4, 1.5, 4.33, 4.65, "How to read it", ["Typical error (RMSE, µg/m³), lower is better; change is against the final model.", "The oracle adds the real weather of the target day: an upper limit, not a real model.", "A forecast for day t is used only if issued h days before t, so it is known at the forecast origin.", "Forecast skill decays with lead: wind correlation 0.90 at 1 day, 0.64 at 7 days."], 13, gap=11)

# ---------------------------------------------------------------- 14 forecast
s = slide("5  Comparative results", "Next-day forecasts against actual values", "Learned models follow rises and drops that ARIMA lags by a day; peaks above about 110 µg/m³ are still under-predicted.")
image(s, "11_forecast_h1.png", 0.6, 1.5, 8.5, 4.65); panel(s, 9.3, 1.5, 3.43, 4.65, "On this plot", ["Same hold-out as Milestone 2.", "Calendar dates on the x axis.", "Ensemble smooths single-model errors.", "ARIMA reacts one day late to turns."], 12.5, gap=7)

# ---------------------------------------------------------------- 15 timeline
s = slide("5  Comparative results", "Four seasons on a date axis", "The ensemble's rolling error is below ARIMA's in most weeks; late May to early June is the exception.")
image(s, "20_timeline_all_folds.png", 0.6, 1.5, 8.5, 4.65); panel(s, 9.3, 1.5, 3.43, 4.65, "Reading it", ["Oct 2025 to Oct 2026, one fold per season.", "Largest gaps in the Nov to Feb smog season.", "Hard spells for all models: late Dec-Jan, late May-June, late July.", "Smog-season errors are about twice as large."], 12.5, gap=7)

# ---------------------------------------------------------------- 16 week ahead
share_gap = (R.loc["XGBoost (no weather forecast)", "h=7 RMSE"] - R.loc["XGBoost", "h=7 RMSE"]) / (R.loc["XGBoost (no weather forecast)", "h=7 RMSE"] - R.loc[S, "h=7 RMSE"]) * 100
s = slide("5  Comparative results", "Week-ahead: weather forecasts close the gap", f"With archived ECMWF forecasts the ensemble's week-ahead RMSE is {R.loc['Ensemble','h=7 RMSE']:.1f} against {R.loc['ARIMA','h=7 RMSE']:.1f} for ARIMA ({-pct(R.loc['Ensemble','h=7 RMSE'], R.loc['ARIMA','h=7 RMSE']):.0f}%).")
image(s, "15_rmse_by_day_ahead.png", 0.6, 1.5, 5.9, 4.65)
panel(s, 6.7, 1.5, 6.03, 4.65, "What the numbers say", [f"Ensemble {R.loc['Ensemble','h=7 RMSE']:.2f}, GRU {R.loc['GRU','h=7 RMSE']:.2f} vs SARIMA {R.loc['SARIMA','h=7 RMSE']:.2f}, ARIMA {R.loc['ARIMA','h=7 RMSE']:.2f}, naive {R.loc['Naive','h=7 RMSE']:.2f}.", f"Significant vs ARIMA (p = {dmp('Ensemble','ARIMA',7):.3f}) and SARIMA (p = {dmp('Ensemble','SARIMA',7):.4f}); week-ahead R² {R.loc['ARIMA','h=7 R2']:.2f} → {R.loc['Ensemble','h=7 R2']:.2f}.",
                                                                f"Learners stay below the classical models at every day ahead (GRU, ensemble).", f"Real forecasts close about {share_gap:.0f}% of the gap to perfect weather knowledge (oracle: {R.loc[S,'h=7 RMSE']:.1f}).", "Gains are small in Jan-Apr and Apr-Jul folds."], 13, gap=7)

# ---------------------------------------------------------------- 17 drivers
s = slide("6  Key insights", "What drives the forecast", "Better inputs, not a bigger model, produced the gain: intraday statistics and weather forecasts.")
image(s, os.path.join(SLIDE_DIR, "s17_shap.png"), 0.6, 1.5, 12.13, 3.05)
panel(s, 0.6, 4.65, 12.13, 1.5, "", None)
bullets(s, 0.8, 4.72, 11.8, 1.4, [("Intraday and forecasts dominate.", f"SHAP shares: intraday PM2.5 {SHAPG['Intraday PM2.5']:.0f}%, forecast weather {SHAPG['Weather forecast']:.0f}%, history {SHAPG['PM2.5 history']:.0f}%; top features are last-hour and evening PM2.5, then forecast temperature range and wind."), ("Why:", "PM2.5 swings from about 52 µg/m³ mid-afternoon to about 91 late at night; a small temperature range with low wind means stagnant air."),
                                  ("Ablation:", f"history only {R.loc['XGBoost (history only)','h=1 RMSE']:.1f} (≈ ARIMA); no intraday {R.loc['XGBoost (no intraday)','h=1 RMSE']:.1f}; no forecast {R.loc['XGBoost (no weather forecast)','h=1 RMSE']:.1f}; full {R.loc['XGBoost','h=1 RMSE']:.1f} (next-day RMSE).")], 12, gap=3)

# ---------------------------------------------------------------- 18 alerts
e = lambda m, t: EXC[(EXC.Model == m) & (EXC.threshold == t)].iloc[0]; lv = LEV[(LEV.Model == "Ensemble") & (LEV.level == ">120")].iloc[0]
s = slide("6  Key insights", "Alerts, categories and the extreme tail", f"Set alert thresholds below the official limits: on days above 120 µg/m³ the ensemble catches only {e('Ensemble',120).recall*100:.0f}%.")
image(s, "18_scatter_and_categories.png", 0.6, 1.5, 12.13, 3.2)
panel(s, 0.6, 4.8, 12.13, 1.4, "", None)
bullets(s, 0.8, 4.9, 11.8, 1.3, [f"Category accuracy (CPCB bands): {R.loc['ARIMA','h=1 CatAcc']:.0f}% (ARIMA) → {R.loc['Ensemble','h=1 CatAcc']:.0f}% (ensemble).", f"Flagging days above 60 µg/m³: F1 {e('ARIMA',60).F1:.2f} → {e('Ensemble',60).F1:.2f}; above 90: {e('ARIMA',90).F1:.2f} → {e('Ensemble',90).F1:.2f}.",
                                f"Weak spot: days above 120 µg/m³ are under-predicted by {abs(lv.Bias):.0f} µg/m³ on average."], 12.5, gap=3)

# ---------------------------------------------------------------- 19 limitations
s = slide("7  Challenges and limitations", "What the results do not show", "We state these openly so the viva can focus on how the work was done.")
panel(s, 0.6, 1.5, 5.95, 4.65, "Data and models", [("Modelled grid data,", "not station readings: peaks are muted."), ("One city, one cell;", "each season covered once."), ("Forecast archive starts Mar 2024:", "about half of the earliest fold's training days use actual weather as a stand-in."), ("One weather model", "(ECMWF); no boundary-layer height."), ("Extremes under-predicted.", "")], 15.5, gap=12)
panel(s, 6.78, 1.5, 5.95, 4.65, "Choices to disclose", [("ARIMA orders", "reused from M2 for all folds."), ("Final model:", "the ensemble was fixed in advance, before any result was seen."), ("Provider definition:", "we rely on Open-Meteo's 'previous day N' forecasts; skill decays with lead as expected."), ("Small data for deep learning:", "about 1,400 days."), ("CO₂ excluded:", "only 725 days, weak correlation.")], 15.5, gap=12)

# ---------------------------------------------------------------- 20 conclusion
s = slide("8  Conclusion", "Findings and future work", "Accuracy improved at both horizons; the main drivers are intraday information and weather forecasts.")
panel(s, 0.6, 1.5, 6.1, 4.65, "Findings", [("Next day:", f"RMSE {H.loc['ARIMA','h=1 RMSE']:.2f} → {H.loc['Ensemble','h=1 RMSE']:.2f} on the M2 hold-out; −{pct(R.loc['Ensemble','h=1 RMSE'], R.loc['ARIMA','h=1 RMSE']):.0f}% over four seasons."), ("Week ahead:", f"−{pct(R.loc['Ensemble','h=7 RMSE'], R.loc['ARIMA','h=7 RMSE']):.0f}% over four seasons, thanks to archived weather forecasts."), ("Driver:", "information, not model complexity."), ("Ensemble", "is within 10% of the best learner in every fold.")], 15.5, gap=14)
panel(s, 6.85, 1.5, 5.88, 4.65, "Future work", ["Several weather models and boundary-layer height.", "CPCB station data and fire counts for crop-residue burning.", "Probabilistic (quantile) forecasts for extremes.", "Temporal convolution or Transformer with more data.", "Daily operational pipeline with monitoring."], 15.5, gap=14)

# ---------------------------------------------------------------- 21 live demo
s = slide("Live demo", "Live demo: modules and how to run", "Everything regenerates from one command, and a clean re-run reproduced identical results.")
panel(s, 0.6, 1.5, 6.0, 4.65, "Demo modules, in order", [("1  Load and preprocess:", "m3_data.py builds the daily table."), ("2  Weather forecasts:", "m3_forecasts.py, issue-time safe."), ("3  Features, tuning, 12 models:", "milestone3_analysis.py, milestone3_dl.py."), ("4  Scoring and tests:", "metrics, Diebold-Mariano, SHAP in results/."), ("5  Website:", "index.html, model comparison and forecast explorer.")], 13.5, gap=9)
panel(s, 6.75, 1.5, 5.98, 4.65, "How to run", ["pip install -r code/requirements.txt", "cd code, then python milestone3_analysis.py (about 6 minutes; re-runs are fast).", "Open index.html: works offline; also deployable on Vercel.", "HOW_TO_RUN.txt explains setup, GitHub and Vercel step by step.", "results/ holds every table; predictions for each fold included."], 13.5, gap=9)

# ---------------------------------------------------------------- 22 team
s = slide("Team", "Team and contributions", f"Team leader: Mahak Soni.  Faculty: {FACULTY}.")
CONTRIB = [("Mahak Soni", "S24CSEU2741", "Team Leader", ["Project lead and final submission", "Data pipeline: hourly to daily, intraday features", "Classical recap: Naive, ARIMA, SARIMA, Holt-Winters", "Tree models and hyper-parameter tuning", "IEEE report, all sections"]),
           ("Darsh Jasoria", "S24CSEU0844", "Team member", ["Weather-forecast download and issue-time checks", "Prophet, LSTM and GRU", "Ensemble, SHAP, ablation, significance tests", "Website, slides, GitHub and Vercel setup", "Milestone 2 review and viva Q&A"]),
           ("Palak Aitwani", "S24CSEU0326", "Team member", None)]
for i, (n, r_, role, items) in enumerate(CONTRIB):
    x = 0.6 + i * 4.1; rect(s, x, 1.5, 3.95, 4.65, WHITE, line=LINE); rect(s, x, 1.5, 3.95, 0.09, [TEAL, ORANGE, BLUE][i])
    tb(s, x + 0.22, 1.72, 3.5, 0.45, n, 19, True, INK, font=SERIF); tb(s, x + 0.22, 2.2, 3.5, 0.3, r_, 12.5, True, [TEAL, ORANGE, BLUE][i])
    tb(s, x + 0.22, 2.5, 3.5, 0.3, role.upper(), 10.5, True, MUT, spc=200)
    if items: bullets(s, x + 0.22, 2.95, 3.55, 3.1, items, 13, gap=7)
    else: tb(s, x + 0.22, 3.4, 3.5, 1.2, "GFG", 44, True, MUT, font=SERIF)

# ---------------------------------------------------------------- 23 references
s = slide("References", "References", None)
refs = ["[1] R. J. Hyndman and G. Athanasopoulos, Forecasting: Principles and Practice, 3rd ed., OTexts, 2021.", "[2] Y. Zhang et al., \"Real-time air quality forecasting, part I,\" Atmospheric Environment, vol. 60, 2012.", "[3] A. Kumar and P. Goyal, \"Forecasting of air quality in Delhi using principal component regression,\" Atmos. Pollut. Res., vol. 2, no. 4, 2011.",
        "[4] X. Li et al., \"LSTM neural network for air pollutant concentration predictions,\" Environmental Pollution, vol. 231, 2017.", "[5] S. Makridakis, E. Spiliotis, V. Assimakopoulos, \"The M4 Competition,\" Int. J. Forecasting, vol. 34, no. 4, 2018.", "[6] T. Chen and C. Guestrin, \"XGBoost,\" in Proc. ACM SIGKDD, 2016.",
        "[7] S. Hochreiter and J. Schmidhuber, \"Long short-term memory,\" Neural Computation, vol. 9, no. 8, 1997.", "[8] K. Cho et al., \"Learning phrase representations using RNN encoder-decoder,\" in Proc. EMNLP, 2014.", "[9] S. J. Taylor and B. Letham, \"Forecasting at scale,\" The American Statistician, vol. 72, no. 1, 2018.",
        "[10] Open-Meteo Air Quality, Historical Weather and Previous Runs APIs; CAMS, ERA5 and ECMWF IFS data, CC BY 4.0. open-meteo.com"]
rect(s, 0.6, 1.5, 12.13, 5.2, WHITE, line=LINE); bullets(s, 0.85, 1.7, 11.7, 5.0, refs, 14, gap=11)


NOTES = [
f"[Time plan: 10 to 12 minutes in total. About 11 minutes of slides plus a one-minute website demo. Skip slides 2, 14 and 23 if time is short. About 20 s here.] Welcome. This is Milestone 3 of our Time Series Analysis project: forecasting Delhi's PM2.5. Our faculty is {FACULTY} and our team leader is Mahak Soni. In Milestone 2 we used classical models; now we add machine learning and deep learning, and we answer the question: why was accuracy low even with strong models?",
"We follow the eight parts of the final-evaluation guidelines: problem, dataset, Phase 2 recap, advanced models, comparative results, insights, limitations, conclusion.",
f"The whole project in one slide. Next-day RMSE on the Milestone 2 test window falls from {H.loc['ARIMA','h=1 RMSE']:.2f} to {H.loc['Ensemble','h=1 RMSE']:.2f}. Over four seasons the ensemble is {pct(R.loc['Ensemble','h=1 RMSE'], R.loc['ARIMA','h=1 RMSE']):.0f}% better next day and {pct(R.loc['Ensemble','h=7 RMSE'], R.loc['ARIMA','h=7 RMSE']):.0f}% better at one week, both statistically significant. The key was better information, not a bigger model.",
"Delhi's PM2.5 is above the 60 microgram standard on more than half of days. Authorities need a forecast a day or more ahead. In Milestone 2 our best model was only 5 percent better than repeating yesterday's value, so we asked why.",
"Our data are hourly air quality from CAMS and hourly weather from ERA5 for one grid cell in central Delhi, plus archived ECMWF weather forecasts. We aggregate to 1,538 days. The forecasts respect the issue time, so nothing from the future is used. Caveat: these are modelled grid values, not station readings, so peaks are muted.",
f"Recap of Milestone 2: Naive, ARIMA, SARIMA and Holt-Winters on one daily average. ARIMA reached {H.loc['ARIMA','h=1 RMSE']:.2f}. We re-ran these models in this project and got identical numbers, so every comparison is like for like.",
"This is our answer to the question. First, only one number per day went into the models. Second, the daily mean hides how the day ended. Third, no weather forecast. Fourth, weekly seasonal terms with no weekly signal. Fifth, a data artefact in the first days. Sixth, only one easy test window. Seventh, no decision-level metric. A stronger model on the same single series gives the same accuracy.",
"We changed six things: intraday features, archived weather forecasts, richer inputs, tuned learners, honest four-season validation, and significance tests. The first three are about information, and they produced most of the gain.",
"Workflow: hourly data to a daily feature table, four rolling test folds, tuning only on data before the first fold, 12 models, then metrics and insights. Same protocol as Milestone 2, with three extra folds so every season is tested once.",
"The models: Ridge, Random Forest, XGBoost, LightGBM, Prophet, LSTM and GRU, plus an equal-weight ensemble that we fixed before seeing results. Tuning used only data before fold one, so no test leakage.",
f"On the identical Milestone 2 test window the ensemble cuts next-day RMSE by {pct(H.loc['Ensemble','h=1 RMSE'], H.loc['ARIMA','h=1 RMSE']):.0f}% and week-ahead RMSE by {pct(H.loc['Ensemble','h=7 RMSE'], H.loc['ARIMA','h=7 RMSE']):.0f}%. The best single models are LightGBM next day and LSTM at one week. One 90-day window is noisy, so the next slide confirms it on four seasons.",
f"Pooled over 360 days across four seasons: next-day RMSE {R.loc['ARIMA','h=1 RMSE']:.1f} to {R.loc['Ensemble','h=1 RMSE']:.1f}, week-ahead {R.loc['ARIMA','h=7 RMSE']:.1f} to {R.loc['Ensemble','h=7 RMSE']:.1f}, both significant by the Diebold-Mariano test. The gain is smallest in April to July.",
f"This table removes one group of inputs at a time from the final XGBoost model. Without the weather forecast, week-ahead error rises from {R.loc['XGBoost','h=7 RMSE']:.1f} to {R.loc['XGBoost (no weather forecast)','h=7 RMSE']:.1f}; with PM2.5 history only it is no better than ARIMA. The oracle row adds the real weather of the target day and is only an upper limit. A forecast is used only if it was issued before the forecast origin.",
"Next-day forecasts against actual values on the hold-out, with calendar dates. The learned models follow rises and drops that ARIMA lags by a day. Peaks above about 110 are still under-predicted.",
"Four seasons on a date axis. The ensemble's rolling error is below ARIMA's in most weeks, especially in the November to February smog season. The exception is late May to early June, where it is no better.",
f"Week ahead. With archived ECMWF forecasts the ensemble reaches {R.loc['Ensemble','h=7 RMSE']:.1f} against {R.loc['ARIMA','h=7 RMSE']:.1f}. Forecast skill itself fades with lead time, from 0.90 wind correlation at one day to 0.64 at seven. The oracle test shows how much perfect weather knowledge would add.",
"What drives the forecast: last-hour and evening PM2.5 of the previous day, then forecast temperature range and wind. That is physical: small temperature range and low wind mean stagnant air. The ablation shows history-only models are no better than ARIMA.",
f"Alerts and categories. The ensemble places {R.loc['Ensemble','h=1 CatAcc']:.0f}% of days in the right pollution category and flags days above 60 with F1 {EXC[(EXC.Model=='Ensemble')&(EXC.threshold==60)].iloc[0].F1:.2f}. The weak spot is the extreme tail above 120, which every model under-predicts, so alert thresholds should be set below the official limits.",
"Limitations, stated openly: modelled grid data, one city, each season covered once, the forecast archive starts in March 2024 so some early training days use actual weather as a stand-in, one weather model, and extremes under-predicted.",
"Conclusion: accuracy improved at both horizons, driven by intraday information and weather forecasts rather than model complexity. Future work: several weather models, station and fire data, probabilistic forecasts, and an operational pipeline.",
"[About 60 s, live demo.] Show index.html: press the quick view 'Next 7 days, all seasons' under Compare models, then See forecasts with 'Old vs new', then 'Why was Milestone 2 not accurate?'. Then mention the one-command re-run: python milestone3_analysis.py. A clean re-run reproduced identical numbers, and the website is one offline file that can also be deployed on Vercel.",
"The team. Mahak Soni is the team leader. Contributions by member are shown on this slide.",
"References. Data: Open-Meteo with CAMS, ERA5 and ECMWF, licensed CC BY 4.0.",
]
assert len(NOTES) == len(prs.slides), (len(NOTES), len(prs.slides))
for sl, t in zip(prs.slides, NOTES): sl.notes_slide.notes_text_frame.text = t

out = os.path.join(ROOT, "Mahak Soni_S24CSEU2741_Milestone3.pptx"); prs.save(out); print("saved", out, "| slides", len(prs.slides))
