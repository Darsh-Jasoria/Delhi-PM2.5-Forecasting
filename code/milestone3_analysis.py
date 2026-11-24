"""
Time Series & Analysis Project - Milestone 3 (Final): advanced ML / DL models for Delhi PM2.5
Target : next-day (h=1) and week-ahead (h=1..7, reissued every 7 days) PM2.5, same protocol as Milestone 2.
Models : Classical recap (Naive, ARIMA, SARIMA, Holt-Winters)  vs  Ridge, Random Forest, XGBoost, LightGBM,
         Prophet (+regressors), LSTM, GRU (milestone3_dl.py) and an equal-weight ensemble.
Data   : Open-Meteo / CAMS air quality + Open-Meteo / ERA5 weather, Delhi grid cell, 2022-08 .. 2026-10 (daily).
Validation: 4 rolling-origin folds of 90 days (fold 4 = the exact Milestone 2 hold-out), expanding training window.
Run    : python milestone3_analysis.py [--fast]      (stages are cached in results/cache, safe to re-run)
"""
import json, pickle, logging
from m3_data import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import xgboost as xgb, lightgbm as lgb
np.random.seed(SEED)
TEAL, ORANGE, GREY, RED, NAVY, PURPLE = "#1F8A7D", "#C4801A", "#5B7075", "#D1503F", "#27476E", "#7A5C9E"
plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "font.size": 10})


def origin_features(D):
    F = pd.DataFrame(index=D.index)
    for L in (0, 1, 2, 3, 6, 13): F[f"ly_{L}"] = D.ly.shift(L)
    for w in (3, 7, 14, 30): F[f"ly_m{w}"] = D.ly.rolling(w).mean()
    F["ly_sd7"] = D.ly.rolling(7).std(); F["ly_d1"] = D.ly - D.ly.shift(1); F["ly_m7_m30"] = F.ly_m7 - F.ly_m30
    for c in INTRA: F[f"{c}"] = D[c]; F[f"{c}_1"] = D[c].shift(1)
    for c in COPOL: F[c] = D[c]; F[f"{c}_1"] = D[c].shift(1); F[f"{c}_m7"] = D[c].rolling(7).mean()
    for c in WCOLS: F[f"w_{c}"] = D[c]; F[f"w_{c}_1"] = D[c].shift(1); F[f"w_{c}_m3"] = D[c].rolling(3).mean()
    F["w_dpres3"] = D.pres - D.pres.shift(3); F["w_rain7"] = D.rain.rolling(7).sum(); F["w_dtemp3"] = D.temp - D.temp.shift(3)
    return F
F0 = origin_features(D)
def groups(cols):
    g = {}
    for c in cols:
        if c.startswith("fc_"): g[c] = "Weather forecast"
        elif c.startswith("w_"): g[c] = "Weather"
        elif c.startswith("ly_"): g[c] = "PM2.5 history"
        elif c.startswith("lpm_"): g[c] = "Intraday PM2.5"
        elif c.startswith(("lpm10", "lno2", "lco", "aqi")): g[c] = "Co-pollutants"
        else: g[c] = "Season"
    return g
def tab_dataset(h):
    """rows indexed by TARGET date t; features known at end of day t-h; plus calendar terms of t (known in advance)."""
    X = F0.copy(); X.index = X.index + pd.Timedelta(days=h)
    doy = X.index.dayofyear.values
    for k in (1, 2): X[f"sin{k}"] = np.sin(2 * np.pi * k * doy / 365.25); X[f"cos{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    X["dow"] = X.index.dayofweek
    if FC:
        for c in FC_NAMES: X[c] = FCD[h][c].reindex(X.index)
    for c in WCOLS: X[f"fw_{c}"] = D[c].reindex(X.index)          # weather OF the target day: only for the 'oracle' upper-bound experiment
    y = D.ly.reindex(X.index)
    ok = X.notna().all(axis=1) & y.notna(); return X[ok], y[ok]
DS = {h: tab_dataset(h) for h in range(1, HMAX + 1)}
COLS_ORACLE = list(DS[1][0].columns); COLS_ALL = [c for c in COLS_ORACLE if not c.startswith("fw_")]
COLS_AIR = [c for c in COLS_ALL if not c.startswith(("w_", "fc_"))]
COLS_NOFC = [c for c in COLS_ALL if not c.startswith("fc_")]
COLS_NOINTRA = [c for c in COLS_ALL if c.split("_1")[0] not in INTRA and c not in INTRA]
COLS_HIST = [c for c in COLS_ALL if c.startswith(("ly_", "sin", "cos", "dow"))]

# =========================================================== 2. metrics
BANDS = [30, 60, 90, 120, 250]       # CPCB PM2.5 24-h categories: good / satisfactory / moderate / poor / very poor / severe
def band(x): return np.digitize(np.asarray(x), BANDS)
def metrics(a_log, p_log):
    a, p = np.exp(np.asarray(a_log)), np.exp(np.asarray(p_log)); e = a - p
    ba, bp = band(a), band(p)
    return dict(RMSE=float(np.sqrt((e ** 2).mean())), MAE=float(np.abs(e).mean()), MAPE=float((np.abs(e) / a).mean() * 100),
                R2=float(1 - (e ** 2).sum() / ((a - a.mean()) ** 2).sum()), Bias=float((p - a).mean()),
                CatAcc=float((ba == bp).mean() * 100), CatWithin1=float((np.abs(ba - bp) <= 1).mean() * 100),
                ExceedAcc=float(((a > 60) == (p > 60)).mean() * 100))
def dm_test(ea, eb, h=1):
    """Diebold-Mariano on squared errors, Harvey small-sample correction. negative stat => A better than B."""
    d = np.asarray(ea) ** 2 - np.asarray(eb) ** 2; n = len(d); dm = d.mean(); v = ((d - dm) ** 2).mean()
    for l in range(1, h): v += 2 * ((d[l:] - dm) * (d[:-l] - dm)).mean()
    v = max(v / n, 1e-12); s = dm / np.sqrt(v) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(s), float(2 * (1 - stats.t.cdf(abs(s), n - 1)))

# =========================================================== 3. hyper-parameter tuning (before fold 1 only)
XGB_SPACE = dict(n_estimators=[200, 400, 700], max_depth=[2, 3, 4, 5], learning_rate=[.01, .02, .04, .08], subsample=[.6, .8, 1.0],
                 colsample_bytree=[.4, .6, .8], min_child_weight=[1, 5, 10], reg_lambda=[1, 5, 20])
LGB_SPACE = dict(n_estimators=[200, 400, 700], num_leaves=[4, 8, 15, 31], learning_rate=[.01, .02, .04, .08], subsample=[.6, .8, 1.0],
                 colsample_bytree=[.4, .6, .8], min_child_samples=[10, 20, 40], reg_lambda=[1, 5, 20])
RF_SPACE = dict(min_samples_leaf=[3, 5, 10], max_features=[.3, .5, .8], max_depth=[8, 12, None])


def stage_tune():
    path = os.path.join(RES, "best_params.json")
    if os.path.exists(path): return json.load(open(path))
    log("tuning tree models (data up to", TUNE_END.date(), ")"); P = {}; NI = 4 if FAST else 16
    cv = TimeSeriesSplit(4, test_size=90)
    for h in (1, 7):
        X, y = DS[h]; m = X.index <= TUNE_END; X, y = X[m], y[m]
        for name, est, space in (("XGBoost", xgb.XGBRegressor(n_jobs=4, random_state=0, tree_method="hist"), XGB_SPACE),
                                 ("LightGBM", lgb.LGBMRegressor(n_jobs=4, random_state=0, verbose=-1, subsample_freq=1), LGB_SPACE),
                                 ("Random Forest", RandomForestRegressor(250, n_jobs=-1, random_state=0), RF_SPACE)):
            s = RandomizedSearchCV(est, space, n_iter=min(NI, 6) if name == "Random Forest" else NI, cv=cv,
                                   scoring="neg_root_mean_squared_error", random_state=0, n_jobs=1).fit(X, y)
            P[f"{name}|h{h}"] = dict(params=s.best_params_, cv_rmse_log=round(float(-s.best_score_), 4)); log(" ", name, h, P[f"{name}|h{h}"])
    json.dump(P, open(path, "w"), indent=1, default=int); return P

def ensure_dl():
    """LSTM / GRU run in their own process (PyTorch and XGBoost/LightGBM ship conflicting OpenMP runtimes on macOS)."""
    if all(os.path.exists(os.path.join(CACHE, f"dl_fold{f}.pkl")) for f in FOLDS): return
    import subprocess; log("starting deep-learning process")
    subprocess.run([sys.executable, os.path.join(HERE, "milestone3_dl.py")] + (["--fast"] if FAST else []), check=True)

# =========================================================== 4. per-fold training and prediction
def classical_fold(tr_end, test_idx):
    """Milestone-2 specifications re-fitted on the fold's training data (M2 daily file, identical to M2)."""
    d2 = pd.read_csv(os.path.join(DATA, "delhi_daily_air_quality.csv"), parse_dates=["date"]).set_index("date")["PM2.5"]
    ly = np.log(d2.clip(lower=1.0)); train = ly.loc[:tr_end]
    def fourier(idx, K=2):
        t = np.asarray((idx - pd.Timestamp("2022-01-01")).days, float)
        return pd.DataFrame({f"{f}{k}": getattr(np, f)(2 * np.pi * k * t / 365.25) for k in range(1, K + 1) for f in ("sin", "cos")}, index=idx)
    X = fourier(ly.index); order = (3, 0, 3)
    hw = lambda s: ExponentialSmoothing(s, trend="add", damped_trend=True, seasonal="add", seasonal_periods=7, initialization_method="estimated")
    arima = SARIMAX(train, order=order, trend="c").fit(disp=False)
    sarima = SARIMAX(train, exog=X.loc[train.index], order=order, seasonal_order=(1, 0, 1, 7)).fit(disp=False, maxiter=300)
    hwf = hw(train).fit()
    def hw_apply(hist):
        pr = hwf.params; return hw(hist).fit(smoothing_level=pr["smoothing_level"], smoothing_trend=pr["smoothing_trend"], smoothing_seasonal=pr["smoothing_seasonal"], damping_trend=pr["damping_trend"], optimized=False)
    out = {1: {}, 7: {}}
    out[1]["Naive"] = ly.shift(1).loc[test_idx]
    for name, r, ex in (("ARIMA", arima, None), ("SARIMA", sarima, X)):
        ap = r.apply(ly, exog=None if ex is None else ex.loc[ly.index], refit=False); out[1][name] = ap.fittedvalues.loc[test_idx]
    out[1]["Holt-Winters"] = hw_apply(ly).fittedvalues.loc[test_idx]
    n = len(test_idx)
    for name in ("Naive", "ARIMA", "SARIMA", "Holt-Winters"):
        s = pd.Series(index=test_idx, dtype=float)
        for i in range(0, n, HMAX):
            origin = test_idx[i]; hist = ly.loc[:origin - pd.Timedelta(days=1)]; idx = test_idx[i:i + HMAX]
            if name == "Naive": f = np.repeat(hist.iloc[-1], len(idx))
            elif name == "Holt-Winters": f = hw_apply(hist).forecast(len(idx))
            else:
                r, ex = (arima, None) if name == "ARIMA" else (sarima, X)
                ap = r.apply(hist, exog=None if ex is None else ex.loc[hist.index], refit=False); f = ap.forecast(len(idx), exog=None if ex is None else ex.loc[idx])
            s.loc[idx] = np.asarray(f)
        out[7][name] = s
    return out

def tab_models(P):
    mk = {"XGBoost": lambda p: xgb.XGBRegressor(n_jobs=4, random_state=SEED, tree_method="hist", **p),
          "LightGBM": lambda p: lgb.LGBMRegressor(n_jobs=4, random_state=SEED, verbose=-1, subsample_freq=1, **p),
          "Random Forest": lambda p: RandomForestRegressor(300, n_jobs=-1, random_state=SEED, **p)}
    return mk
def prophet_fold(tr_end, test_idx):
    from prophet import Prophet
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR); logging.getLogger("prophet").setLevel(logging.ERROR)
    regs = ["ly_0", "ly_m7", "lpm_eve", "lco", "lno2", "lpm10", "w_temp", "w_rh", "w_wind", "w_pres"]
    out = {}
    for proto, h in ((1, 1), (7, 7)):          # protocol 7 uses regressors lagged by 7 days: available for every day of the block
        X, y = DS[h]; tr = X.index <= tr_end; Xtr = X.loc[tr, regs]
        df = pd.DataFrame({"ds": Xtr.index, "y": y.loc[tr].values}); df[regs] = Xtr.values
        m = Prophet(yearly_seasonality=10, weekly_seasonality=True, daily_seasonality=False, changepoint_prior_scale=0.05, seasonality_mode="additive")
        for r in regs: m.add_regressor(r)
        m.fit(df); fut = pd.DataFrame({"ds": test_idx}); fut[regs] = X.loc[test_idx, regs].values
        out[proto] = pd.Series(m.predict(fut)["yhat"].values, index=test_idx)
    return out

def run_fold(fi, P):
    path = os.path.join(CACHE, f"fold{fi}.pkl")
    if os.path.exists(path): return pickle.load(open(path, "rb"))
    ts, te = FOLDS[fi]; tr_end = ts - pd.Timedelta(days=1); idx = pd.date_range(ts, te); n = len(idx); log(f"fold {fi}:", ts.date(), "->", te.date())
    pred = {1: {}, 7: {}}; extra = {}
    c = classical_fold(tr_end, idx); [pred[p].update({k: v}) for p in (1, 7) for k, v in c[p].items()]; log("  classical done")
    # ---- tabular models, direct strategy: one model per horizon h = 1..7
    mk = tab_models(P); allh = {}                       # allh[name][h] -> Series on test index
    specs = [("Ridge", COLS_ALL), ("Random Forest", COLS_ALL), ("XGBoost", COLS_ALL), ("LightGBM", COLS_ALL),
             ("XGBoost (air only)", COLS_AIR), ("XGBoost (no intraday)", COLS_NOINTRA), ("XGBoost (history only)", COLS_HIST),
             ("XGBoost (+future weather, oracle)", COLS_ORACLE)]
    if FC: specs.append(("XGBoost (no weather forecast)", COLS_NOFC))
    for name, cols in specs:
        allh[name] = {}
        for h in range(1, HMAX + 1):
            X, y = DS[h]; tr = X.index <= tr_end; base = name.split(" (")[0]; key = f"{base}|h{1 if h <= 3 else 7}"
            if base == "Ridge": mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30), cv=TimeSeriesSplit(4)))
            else: mdl = mk[base](P[key]["params"])
            mdl.fit(X.loc[tr, cols], y.loc[tr]); allh[name][h] = pd.Series(mdl.predict(X.loc[idx, cols]), index=idx)
            if name == "XGBoost" and h == 1 and fi == 4:                       # SHAP (native tree contributions) on the M2 hold-out
                sh = mdl.get_booster().predict(xgb.DMatrix(X.loc[idx, cols]), pred_contribs=True)[:, :-1]
                extra["shap"] = pd.DataFrame(sh, index=idx, columns=cols); extra["xgb_h1"] = mdl
        pred[1][name] = allh[name][1]
        pred[7][name] = pd.Series([allh[name][(i % HMAX) + 1].iloc[i] for i in range(n)], index=idx)
        log("  ", name, "done")
    # ---- Prophet
    pp = prophet_fold(tr_end, idx); pred[1]["Prophet"] = pp[1]; pred[7]["Prophet"] = pp[7]; log("   Prophet done")
    # ---- LSTM / GRU come from the separate deep-learning process
    dl = pickle.load(open(os.path.join(CACHE, f"dl_fold{fi}.pkl"), "rb")); extra["epochs"] = dl["epochs"]
    for p in (1, 7):
        for k in ("LSTM", "GRU"): pred[p][k] = dl["pred"][p][k]
    # ---- ensemble (fixed a priori: mean of log-forecasts of XGBoost, LightGBM, Random Forest, LSTM, GRU)
    for p in (1, 7):
        pred[p]["Ensemble"] = pd.concat([pred[p][m] for m in ("XGBoost", "LightGBM", "Random Forest", "LSTM", "GRU")], axis=1).mean(1)
    res = dict(pred=pred, idx=idx, y=D.ly.loc[idx], extra=extra); pickle.dump(res, open(path, "wb")); return res

# =========================================================== 5. reporting
ORDER = ["Naive", "ARIMA", "SARIMA", "Holt-Winters", "Ridge", "Random Forest", "XGBoost", "LightGBM", "Prophet", "LSTM", "GRU", "Ensemble"]
ABL = ["XGBoost", "XGBoost (air only)", "XGBoost (no intraday)", "XGBoost (history only)", "XGBoost (+future weather, oracle)"]
if FC: ABL.insert(1, "XGBoost (no weather forecast)")
CLASSICAL = ["Naive", "ARIMA", "SARIMA", "Holt-Winters"]
def report(R, P):
    rows, per_fold = [], []
    for fi, r in R.items():
        for p in (1, 7):
            for m, s in r["pred"][p].items():
                per_fold.append(dict(fold=fi, horizon=p, Model=m, **metrics(r["y"], s)))
    pf = pd.DataFrame(per_fold); pf.to_csv(os.path.join(RES, "all_folds_metrics.csv"), index=False)
    pooled = []
    for p in (1, 7):
        for m in R[4]["pred"][p]:
            a = np.concatenate([R[f]["y"].values for f in R]); q = np.concatenate([R[f]["pred"][p][m].values for f in R]); pooled.append(dict(horizon=p, Model=m, **metrics(a, q)))
    pooled = pd.DataFrame(pooled); pooled.to_csv(os.path.join(RES, "rolling_pooled_metrics.csv"), index=False)
    def wide(df):
        out = []
        for m in ORDER + ABL[1:]:
            d1, d7 = df[(df.horizon == 1) & (df.Model == m)], df[(df.horizon == 7) & (df.Model == m)]
            if len(d1): out.append(dict(Model=m, **{f"h=1 {k}": round(float(d1[k].iloc[0]), 2) for k in ("RMSE", "MAE", "MAPE", "R2", "CatAcc")}, **{f"h=7 {k}": round(float(d7[k].iloc[0]), 2) for k in ("RMSE", "MAE", "MAPE", "R2", "CatAcc")}))
        return pd.DataFrame(out)
    main = wide(pf[pf.fold == 4]); main.to_csv(os.path.join(RES, "model_comparison_m3_holdout.csv"), index=False)
    roll = wide(pooled); roll.to_csv(os.path.join(RES, "model_comparison_m3_rolling.csv"), index=False)
    print("\nM2 HOLD-OUT (fold 4)\n", main.to_string(index=False)); print("\nROLLING (4 folds pooled, 360 days)\n", roll.to_string(index=False))
    # ---- RMSE per fold
    fr = pf[(pf.horizon == 1)].pivot(index="Model", columns="fold", values="RMSE").reindex(ORDER + ABL[1:]).round(2); fr.to_csv(os.path.join(RES, "rmse_by_fold_h1.csv"))
    fr7 = pf[(pf.horizon == 7)].pivot(index="Model", columns="fold", values="RMSE").reindex(ORDER + ABL[1:]).round(2); fr7.to_csv(os.path.join(RES, "rmse_by_fold_h7.csv"))
    print("\nRMSE by fold h=1\n", fr, "\nRMSE by fold h=7\n", fr7)
    # ---- Diebold-Mariano (pooled folds)
    dm = []
    for p in (1, 7):
        a = np.concatenate([R[f]["y"].values for f in R]); err = lambda m: np.exp(a) - np.exp(np.concatenate([R[f]["pred"][p][m].values for f in R]))
        for m in ORDER:
            for base in ("Naive", "ARIMA", "SARIMA"):
                if m != base: s, pv = dm_test(err(m), err(base), 1 if p == 1 else HMAX); dm.append(dict(horizon=p, Model=m, vs=base, DM_stat=round(s, 2), p_value=round(pv, 4)))
    pd.DataFrame(dm).to_csv(os.path.join(RES, "diebold_mariano.csv"), index=False)
    # ---- error vs day ahead (k = 1..7 inside the 7-day protocol)
    kr = {}
    for m in ["Naive", "ARIMA", "SARIMA", "XGBoost", "LightGBM", "GRU", "LSTM", "Ensemble"]:
        a = np.concatenate([R[f]["y"].values for f in R]); q = np.concatenate([R[f]["pred"][7][m].values for f in R]); k = np.concatenate([np.arange(len(R[f]["idx"])) % HMAX + 1 for f in R])
        kr[m] = [float(np.sqrt(np.mean((np.exp(a[k == j]) - np.exp(q[k == j])) ** 2))) for j in range(1, HMAX + 1)]
    kr = pd.DataFrame(kr, index=range(1, HMAX + 1)).round(2); kr.index.name = "day_ahead"; kr.to_csv(os.path.join(RES, "rmse_by_day_ahead.csv")); print("\nRMSE by day ahead\n", kr)
    # ---- errors by pollution level and by month (pooled h=1)
    a = np.exp(np.concatenate([R[f]["y"].values for f in R])); dates = np.concatenate([R[f]["idx"].values for f in R]); dates = pd.DatetimeIndex(dates)
    lev = pd.cut(a, [0, 60, 90, 120, 1000], labels=["<=60", "60-90", "90-120", ">120"]); tabs = []
    for m in ("Naive", "ARIMA", "XGBoost", "GRU", "Ensemble"):
        q = np.exp(np.concatenate([R[f]["pred"][1][m].values for f in R]))
        tabs.append(pd.DataFrame(dict(level=lev, mon=dates.month, ae=np.abs(a - q), se=(a - q) ** 2, bias=q - a, Model=m)))
    T = pd.concat(tabs); bylev = T.groupby(["Model", "level"], observed=True).agg(MAE=("ae", "mean"), RMSE=("se", lambda x: np.sqrt(x.mean())), Bias=("bias", "mean"), n=("ae", "size")).round(2)
    bylev.to_csv(os.path.join(RES, "error_by_pm25_level_h1.csv")); print("\nError by PM2.5 level (h=1)\n", bylev)
    bymon = T.groupby(["Model", "mon"]).agg(MAE=("ae", "mean")).round(2).unstack(0); bymon.to_csv(os.path.join(RES, "mae_by_month_h1.csv"))
    # ---- SHAP importance on M2 hold-out
    sh = R[4]["extra"]["shap"]; imp = sh.abs().mean().sort_values(ascending=False); gmap = groups(sh.columns)
    imp.round(4).to_csv(os.path.join(RES, "shap_importance_h1.csv"), header=["mean_abs_shap"])
    gi = imp.groupby(gmap).sum().sort_values(ascending=False); gi = (gi / gi.sum() * 100).round(1); gi.to_csv(os.path.join(RES, "shap_group_share_h1.csv"), header=["share_pct"]); print("\nSHAP group share %\n", gi, "\nTop 12\n", imp.head(12).round(3))
    # ---- exceedance (>60) / unhealthy (>90) detection pooled h=1
    ex = []
    for m in ORDER:
        q = np.exp(np.concatenate([R[f]["pred"][1][m].values for f in R]))
        for thr in (60, 90, 120):
            tp = ((a > thr) & (q > thr)).sum(); fp = ((a <= thr) & (q > thr)).sum(); fn = ((a > thr) & (q <= thr)).sum()
            ex.append(dict(Model=m, threshold=thr, precision=round(tp / max(tp + fp, 1), 3), recall=round(tp / max(tp + fn, 1), 3), F1=round(2 * tp / max(2 * tp + fp + fn, 1), 3)))
    pd.DataFrame(ex).to_csv(os.path.join(RES, "exceedance_detection_h1.csv"), index=False)
    # ---- predictions file (original scale)
    allp = []
    for fi, r in R.items():
        for p in (1, 7):
            d = pd.DataFrame({m: np.exp(s).round(4) for m, s in r["pred"][p].items()}); d.insert(0, "actual", np.exp(r["y"]).round(4)); d.insert(0, "horizon", p); d.insert(0, "fold", fi); allp.append(d)
    pd.concat(allp).rename_axis("date").to_csv(os.path.join(RES, "predictions_all_folds.csv"))
    figures(R, P, pf, pooled, fr, kr, imp, gi, main)
    return main, roll

def figures(R, P, pf, pooled, fr, kr, imp, gi, main):
    r4 = R[4]; idx = r4["idx"]; yact = np.exp(D.ly)
    colors = {"Naive": "#9AA5A8", "ARIMA": GREY, "SARIMA": "#8FA3A6", "Holt-Winters": "#B8C2C4", "Ridge": "#D9A441", "Random Forest": "#8B6B3E", "XGBoost": ORANGE, "LightGBM": RED, "Prophet": PURPLE, "LSTM": NAVY, "GRU": "#3C78B4", "Ensemble": TEAL}
    for p, nm, ttl in ((1, "11_forecast_h1.png", "Next-day (h = 1) forecasts, 90-day hold-out"), (7, "12_forecast_h7.png", "Week-ahead forecasts re-issued every 7 days, 90-day hold-out")):
        fig, ax = plt.subplots(figsize=(10, 4.2)); ax.plot(yact.loc[idx], color="black", lw=1.6, label="Actual")
        for m in ("ARIMA", "XGBoost", "GRU", "Ensemble"): ax.plot(np.exp(r4["pred"][p][m]), color=colors[m], lw=1.2, label=m, alpha=.95 if m == "Ensemble" else .8)
        ax.set_ylabel("PM2.5 (µg/m³)"); ax.set_xlabel("Date"); ax.set_title(ttl, weight="bold"); ax.legend(ncol=5, fontsize=8)
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y")); fig.autofmt_xdate(rotation=30, ha="right")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, nm)); plt.close(fig)
    # RMSE M2 vs M3 bars on hold-out
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for a_, p, t in ((ax[0], 1, "Next-day (h = 1)"), (ax[1], 7, "Week-ahead (h = 7 protocol)")):
        d = pf[(pf.fold == 4) & (pf.horizon == p) & pf.Model.isin(ORDER)].set_index("Model").reindex(ORDER)
        a_.bar(range(len(d)), d.RMSE, color=[colors[m] for m in d.index]); a_.set_xticks(range(len(d)), d.index, rotation=60, ha="right", fontsize=8)
        for i, v in enumerate(d.RMSE): a_.text(i, v + .2, f"{v:.1f}", ha="center", fontsize=7)
        a_.set_title(t, weight="bold"); a_.set_ylabel("RMSE (µg/m³)"); a_.axhline(d.RMSE["Naive"], color=RED, ls="--", lw=.8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "13_rmse_classical_vs_advanced.png")); plt.close(fig)
    # RMSE per fold
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a_, p in zip(ax, (1, 7)):
        d = pf[(pf.horizon == p)].pivot(index="fold", columns="Model", values="RMSE")
        for m in ("Naive", "ARIMA", "XGBoost", "GRU", "Ensemble"): a_.plot(d.index, d[m], marker="o", color=colors[m], label=m, lw=2 if m == "Ensemble" else 1.2)
        a_.set_xticks([1, 2, 3, 4], [f"Fold {f}\n{FOLDS[f][0]:%b %y}-{FOLDS[f][1]:%b %y}" for f in (1, 2, 3, 4)], fontsize=8); a_.set_title(f"RMSE by fold, {'h = 1' if p == 1 else 'week-ahead'}", weight="bold"); a_.set_ylabel("RMSE (µg/m³)")
    ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(FIG, "14_rmse_by_fold.png")); plt.close(fig)
    # day-ahead curve
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for m in kr.columns: ax.plot(kr.index, kr[m], marker="o", ms=3, color=colors[m], label=m, lw=2 if m == "Ensemble" else 1.1)
    ax.set_xlabel("Days ahead"); ax.set_ylabel("RMSE (µg/m³), 4 folds pooled"); ax.set_title("Error grows with horizon; ML keeps its edge", weight="bold"); ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(os.path.join(FIG, "15_rmse_by_day_ahead.png")); plt.close(fig)
    # SHAP
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.6, 1]))
    top = imp.head(15)[::-1]; ax[0].barh(top.index, top.values, color=TEAL); ax[0].set_title("XGBoost next-day model: mean |SHAP| (top 15)", weight="bold"); ax[0].set_xlabel("mean |SHAP| on log PM2.5")
    ax[1].bar(gi.index, gi.values, color=ORANGE); ax[1].set_xticks(range(len(gi)), gi.index, rotation=30, ha="right"); ax[1].set_ylabel("share of total |SHAP| (%)"); ax[1].set_title("By feature group", weight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "16_shap_importance.png")); plt.close(fig)
    # ablation
    fig, ax = plt.subplots(figsize=(7.5, 3.6)); d = pooled[(pooled.Model.isin(ABL))].pivot(index="Model", columns="horizon", values="RMSE").reindex(ABL)
    xs = np.arange(len(d)); ax.bar(xs - .2, d[1], .4, color=TEAL, label="h = 1"); ax.bar(xs + .2, d[7], .4, color=ORANGE, label="week-ahead")
    ax.set_xticks(xs, [m.replace("XGBoost ", "").strip("()") or "full" for m in d.index], fontsize=8); ax.set_ylabel("RMSE (µg/m³), 4 folds pooled"); ax.set_title("Ablation: what feature groups buy", weight="bold"); ax.legend()
    for i, (a1, a7) in enumerate(zip(d[1], d[7])): ax.text(i - .2, a1 + .2, f"{a1:.1f}", ha="center", fontsize=7); ax.text(i + .2, a7 + .2, f"{a7:.1f}", ha="center", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "17_ablation.png")); plt.close(fig)
    # scatter + category confusion (pooled h=1 ensemble)
    a = np.exp(np.concatenate([R[f]["y"].values for f in R])); q = np.exp(np.concatenate([R[f]["pred"][1]["Ensemble"].values for f in R])); qa = np.exp(np.concatenate([R[f]["pred"][1]["ARIMA"].values for f in R]))
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    for a_, qq, t, c in ((ax[0], qa, "ARIMA (M2)", GREY), (ax[1], q, "Ensemble (M3)", TEAL)):
        a_.scatter(a, qq, s=6, alpha=.45, color=c); lim = [0, max(a.max(), qq.max()) * 1.02]; a_.plot(lim, lim, "k--", lw=.8); a_.set_xlabel("Actual PM2.5"); a_.set_ylabel("Predicted"); a_.set_title(f"{t}: R² = {1 - ((a - qq) ** 2).sum() / ((a - a.mean()) ** 2).sum():.2f}", weight="bold")
    cats = ["Good", "Satisf.", "Moderate", "Poor", "V.poor", "Severe"]; cm = np.zeros((6, 6), int)
    for i, j in zip(band(a), band(q)): cm[i, j] += 1
    nz = [i for i in range(6) if cm[i].sum() or cm[:, i].sum()]; cmn = cm[np.ix_(nz, nz)]; rown = cmn / np.maximum(cmn.sum(1, keepdims=True), 1)
    ax[2].imshow(rown, cmap="Blues", vmin=0, vmax=1); ax[2].set_xticks(range(len(nz)), [cats[i] for i in nz], fontsize=7, rotation=30); ax[2].set_yticks(range(len(nz)), [cats[i] for i in nz], fontsize=7); ax[2].grid(False)
    for i in range(len(nz)):
        for j in range(len(nz)): ax[2].text(j, i, f"{cmn[i, j]}", ha="center", va="center", fontsize=8, color="white" if rown[i, j] > .5 else "black")
    ax[2].set_xlabel("Predicted category"); ax[2].set_ylabel("Actual category"); ax[2].set_title(f"CPCB category: {np.mean(band(a) == band(q)) * 100:.0f}% exact", weight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "18_scatter_and_categories.png")); plt.close(fig)
    # timeline of all four folds on a real date axis + rolling error
    ix = pd.DatetimeIndex(np.concatenate([R[f]["idx"].values for f in (1, 2, 3, 4)]))
    act = pd.Series(np.exp(np.concatenate([R[f]["y"].values for f in (1, 2, 3, 4)])), index=ix)
    pr = lambda m: pd.Series(np.exp(np.concatenate([R[f]["pred"][1][m].values for f in (1, 2, 3, 4)])), index=ix)
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True, gridspec_kw=dict(height_ratios=[1.5, 1]))
    ax[0].plot(act, color="black", lw=1.3, label="Actual"); ax[0].plot(pr("ARIMA"), color=GREY, lw=.9, label="ARIMA (Milestone 2)"); ax[0].plot(pr("Ensemble"), color=TEAL, lw=1.1, label="Ensemble (Milestone 3)")
    ax[0].axhline(60, color=RED, ls="--", lw=.8, label="60 µg/m³ standard"); ax[0].set_ylabel("PM2.5 (µg/m³)"); ax[0].legend(ncol=4, fontsize=8, loc="upper right")
    for f, (s_, e_) in FOLDS.items():
        for a_ in ax: a_.axvspan(s_, e_ + pd.Timedelta(days=1), color=GREY, alpha=.07 if f % 2 else .15)
        ax[0].text(s_ + pd.Timedelta(days=2), 0.03, f"Fold {f}", fontsize=8, color=GREY, transform=ax[0].get_xaxis_transform())
    for m, c, lb in (("Naive", "#9AA5A8", "Naive"), ("ARIMA", GREY, "ARIMA (M2)"), ("Ensemble", TEAL, "Ensemble (M3)")): ax[1].plot((act - pr(m)).abs().rolling(14).mean(), color=c, lw=1.6 if m == "Ensemble" else 1.1, label=lb)
    ax[1].set_ylabel("14-day rolling MAE (µg/m³)"); ax[1].set_xlabel("Date"); ax[1].legend(ncol=3, fontsize=8)
    ax[1].xaxis.set_major_locator(mdates.MonthLocator()); ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y")); fig.autofmt_xdate(rotation=30, ha="right")
    ax[0].set_title("Next-day forecasts over four seasons (Oct 2025 to Oct 2026) and rolling error", weight="bold"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "20_timeline_all_folds.png")); plt.close(fig)
    # workflow diagram
    fig, ax = plt.subplots(figsize=(11, 1.9)); ax.axis("off"); ax.grid(False)
    boxes = ["Open-Meteo\nCAMS air\nERA5 weather\nECMWF forecasts", "Daily table\n+ intraday stats\n(evening, max,\nmorning)", "79 features\nlags, intraday,\nweather,\nforecast", "4 rolling folds\n90 days each\nexpanding\ntraining window", "Tune on data\nbefore fold 1\ntrees, LSTM,\nGRU grid", "12 models\none per horizon\n1 to 7 days", "Score\nRMSE, MAE, MAPE\nR², category,\nDiebold-Mariano", "Insights\nSHAP, ablation,\nerrors by\nregime"]
    for i, b in enumerate(boxes):
        x = i * 1.0; ax.add_patch(plt.Rectangle((x + .04, .06), .88, .88, color=[TEAL, TEAL, ORANGE, ORANGE, NAVY, NAVY, PURPLE, PURPLE][i], alpha=.92)); ax.text(x + .48, .5, b, ha="center", va="center", color="white", fontsize=7.7, weight="bold", linespacing=1.25)
        if i < len(boxes) - 1: ax.annotate("", xy=(x + 1.04, .5), xytext=(x + .93, .5), arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    ax.set_xlim(0, 8.05); ax.set_ylim(0, 1); fig.subplots_adjust(left=.005, right=.995, top=.99, bottom=.01); fig.savefig(os.path.join(FIG, "10_workflow.png")); plt.close(fig)
    # data snapshot figure: PM2.5 + weather drivers
    fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(D.pm, color=TEAL, lw=.8); ax[0].set_ylabel("PM2.5 (µg/m³)"); ax[0].axhline(60, color=RED, ls="--", lw=.8)
    ax[1].plot(D.temp, color=ORANGE, lw=.8); ax[1].set_ylabel("Temperature (°C)")
    ax[2].plot(D.wind, color=NAVY, lw=.8); ax[2].set_ylabel("Wind (km/h)")
    for f, (s, e) in FOLDS.items():
        for a_ in ax: a_.axvspan(s, e, color=GREY, alpha=.08 if f % 2 else .16)
    ax[0].set_title("Delhi daily PM2.5 with weather drivers; shaded = four 90-day test folds", weight="bold"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "19_data_overview.png")); plt.close(fig)

if __name__ == "__main__":
    log("daily table", D.shape, D.index.min().date(), D.index.max().date(), "| features", len(COLS_ALL), "| folds", {k: (v[0].date().isoformat(), v[1].date().isoformat()) for k, v in FOLDS.items()})
    P = stage_tune(); ensure_dl()
    R = {fi: run_fold(fi, P) for fi in (4, 3, 2, 1)}
    main, roll = report(R, P)
    json.dump(dict(rows=len(D), start=str(D.index.min().date()), end=str(D.index.max().date()), n_features=len(COLS_ALL), features_dl=len(DL_FEATS) + 2,
                   folds={k: [str(v[0].date()), str(v[1].date())] for k, v in FOLDS.items()}, tune_end=str(TUNE_END.date()), best_params=P,
                   dl_best_epochs={k: r["extra"].get("epochs") for k, r in R.items()}), open(os.path.join(RES, "summary_m3.json"), "w"), indent=1, default=str)
    log("ALL DONE")
