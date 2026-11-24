"""Shared data preparation for Milestone 3 (no torch / xgboost imports, so it is safe to load from either process)."""
import os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
FAST = "--fast" in sys.argv
FC = True                           # archived ECMWF weather forecasts are always used as inputs (the final model)
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results_fast" if FAST else "results")           # --fast is a quick smoke test, not for reporting
FIG = os.path.join(RES, "figures")
CACHE = os.path.join(RES, "cache"); [os.makedirs(p, exist_ok=True) for p in (FIG, RES, CACHE)]
SEED = 42
TEST_DAYS, HMAX = 90, 7
LAST = pd.Timestamp("2026-10-20")
FOLDS = {4: (LAST - pd.Timedelta(days=89), LAST)}                       # fold 4 = Milestone 2 hold-out (23 Jul - 20 Oct 2026)
for f in (3, 2, 1):
    e = FOLDS[f + 1][0] - pd.Timedelta(days=1); FOLDS[f] = (e - pd.Timedelta(days=89), e)
TUNE_END = FOLDS[1][0] - pd.Timedelta(days=1)                            # tuning only sees data before fold 1 -> no leakage
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)

# =========================================================== 1. data: hourly -> daily feature table
def make_daily():
    air = pd.read_csv(os.path.join(DATA, "delhi_hourly_raw.csv"), parse_dates=["time"]).set_index("time")
    air.columns = ["pm", "pm10", "no2", "co", "co2", "aqi"]; air = air.drop(columns="co2")
    air = air.loc[air.pm.first_valid_index():].interpolate(limit=6)           # drops the 3 back-filled days of M2
    wx = pd.read_csv(os.path.join(DATA, "delhi_weather_raw.csv"), skiprows=3, parse_dates=["time"]).set_index("time")
    wx.columns = ["temp", "rh", "dew", "pres", "rain", "wind", "wdir", "cloud"]
    wx["u"] = -wx.wind * np.sin(np.deg2rad(wx.wdir)); wx["v"] = -wx.wind * np.cos(np.deg2rad(wx.wdir))
    g = air.resample("D"); D = pd.DataFrame(index=g.mean().index)
    lg = lambda s: np.log(s.clip(lower=1.0))
    D["pm"] = g["pm"].mean(); D["ly"] = lg(D.pm)
    D["lpm_max"] = lg(g["pm"].max()); D["lpm_min"] = lg(g["pm"].min()); D["lpm_last"] = lg(g["pm"].last())
    D["lpm_eve"] = lg(air.pm.between_time("18:00", "23:00").resample("D").mean())
    D["lpm_morn"] = lg(air.pm.between_time("05:00", "10:00").resample("D").mean())
    D["lpm10"] = lg(g["pm10"].mean()); D["lno2"] = lg(g["no2"].mean()); D["lco"] = lg(g["co"].mean()); D["aqi"] = g["aqi"].mean() / 100
    w = wx.resample("D")
    D["temp"] = w.temp.mean(); D["trange"] = w.temp.max() - w.temp.min(); D["rh"] = w.rh.mean(); D["pres"] = w.pres.mean()
    D["rain"] = w.rain.sum(); D["wind"] = w.wind.mean(); D["u"] = w.u.mean(); D["v"] = w.v.mean(); D["cloud"] = w.cloud.mean()
    D = D.loc[:LAST]; D = D.interpolate(limit=3).dropna()
    return D
D = make_daily()
INTRA = ["lpm_max", "lpm_min", "lpm_last", "lpm_eve", "lpm_morn"]
COPOL = ["lpm10", "lno2", "lco", "aqi"]
WCOLS = ["temp", "trange", "rh", "pres", "rain", "wind", "u", "v", "cloud"]

DL_FEATS = ["ly"] + INTRA + COPOL + ["temp", "rh", "pres", "rain", "wind", "u", "v", "cloud", "trange"]
SDL = D[DL_FEATS].copy(); _doy = D.index.dayofyear.values
SDL["s1"] = np.sin(2 * np.pi * _doy / 365.25); SDL["c1"] = np.cos(2 * np.pi * _doy / 365.25)

# ---- archived weather forecasts (Open-Meteo Previous Runs API, ECMWF IFS), built by m3_forecasts.py
FC_NAMES = [f"fc_{c}" for c in WCOLS]
FCD = {}                                   # FCD[k] row t = forecast of day t issued k days earlier (known at the end of day t-k)
if FC:
    _f = pd.read_csv(os.path.join(DATA, "forecasts_ecmwf_ifs025_daily.csv"), parse_dates=["date"]).set_index("date")
    _idx = D.index.union(_f.index); _act = D[WCOLS].rename(columns=lambda c: f"fc_{c}").reindex(_idx)
    for _k in range(1, 8):
        _d = pd.DataFrame({f"fc_{c}": _f[f"{c}__{_k}"] for c in WCOLS}).reindex(_idx)
        FCD[_k] = _d.fillna(_act)          # archive starts in Feb 2024: earlier days use the actual weather (perfect-prognosis fill)
    FC_FIRST = _f.dropna().index.min()
