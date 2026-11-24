"""Builds index.html (the website) at the project root: a single offline file (no internet, no libraries) with every result of Milestone 3."""
import os, json
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); RES = os.path.join(ROOT, "results")
rd = lambda f, **k: pd.read_csv(os.path.join(RES, f), **k)
def mt(f):
    d = rd(f).set_index("Model"); return {m: {c.replace("=", "").replace(" ", "_"): float(v) for c, v in r.items()} for m, r in d.iterrows()}
P = rd("predictions_all_folds.csv", parse_dates=["date"])
dates = sorted(P.date.unique()); models = [c for c in P.columns if c not in ("date", "fold", "horizon", "actual")]
preds = {}
for h in (1, 7):
    d = P[P.horizon == h].sort_values("date")
    preds[str(h)] = dict(dates=[str(x)[:10] for x in d.date], fold=[int(x) for x in d.fold], actual=[round(float(x), 1) for x in d.actual],
                         **{m: [round(float(x), 1) for x in d[m]] for m in models})
summary = json.load(open(os.path.join(RES, "summary_m3.json")))
data = dict(hold=mt("model_comparison_m3_holdout.csv"), roll=mt("model_comparison_m3_rolling.csv"),
            fold1=rd("rmse_by_fold_h1.csv").set_index("Model").to_dict("index"), fold7=rd("rmse_by_fold_h7.csv").set_index("Model").to_dict("index"),
            dayahead=rd("rmse_by_day_ahead.csv").to_dict("list"), shap=rd("shap_importance_h1.csv").head(15).values.tolist(),
            shapg=rd("shap_group_share_h1.csv").values.tolist(), exc=rd("exceedance_detection_h1.csv").to_dict("records"),
            lev=rd("error_by_pm25_level_h1.csv").to_dict("records"), dm=rd("diebold_mariano.csv").to_dict("records"),
            preds=preds, folds=summary["folds"], models=models, rows=summary["rows"], start=summary["start"], end=summary["end"])
html = open(os.path.join(HERE, "dashboard_template.html"), encoding="utf8").read().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"), default=str))
out = os.path.join(ROOT, "index.html"); open(out, "w", encoding="utf8").write(html); print("saved", out, round(len(html) / 1024), "KB")
