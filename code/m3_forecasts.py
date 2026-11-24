"""
Milestone 3b: archived numerical weather FORECASTS for Delhi (Open-Meteo Previous Runs API).
For a target day t and horizon h the model may only use the forecast of day t issued h days earlier ("previous_day<h>"),
so nothing newer than the forecast origin is ever used. Coverage starts in early 2024.
  python m3_forecasts.py pilot     # compare GFS and ECMWF against ERA5 on Jan-Jun 2025 (before the first test fold)
  python m3_forecasts.py download <model>   # fetch leads 1..7 for the whole period into data/forecasts_<model>.csv
"""
import os, sys, json, time, urllib.request, urllib.error
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); DATA = os.path.join(ROOT, "data")
VARS = ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m", "precipitation", "cloud_cover"]
URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
def fetch(model, lead, start, end, tries=6):
    hv = ",".join(f"{v}_previous_day{lead}" for v in VARS)
    q = f"{URL}?latitude=28.60&longitude=77.20&start_date={start}&end_date={end}&hourly={hv}&models={model}&timezone=Asia%2FKolkata"
    for k in range(tries):
        try:
            d = json.load(urllib.request.urlopen(q, timeout=120)); break
        except urllib.error.HTTPError as e:
            if e.code == 429 and k < tries - 1: print("  rate limited, waiting 65 s", flush=True); time.sleep(65); continue
            raise
    h = d["hourly"]; df = pd.DataFrame({"time": pd.to_datetime(h["time"])}).set_index("time")
    for v in VARS:
        col = next(c for c in h if c.startswith(v + "_previous_day"))
        df[v] = h[col]
    return df
def daily(df):
    """hourly forecast of one lead -> daily features with the same definitions as the actual-weather features"""
    u = -df.wind_speed_10m * np.sin(np.deg2rad(df.wind_direction_10m)); v = -df.wind_speed_10m * np.cos(np.deg2rad(df.wind_direction_10m))
    g = df.resample("D"); out = pd.DataFrame(index=g.mean().index)
    out["temp"] = g.temperature_2m.mean(); out["trange"] = g.temperature_2m.max() - g.temperature_2m.min(); out["rh"] = g.relative_humidity_2m.mean()
    out["pres"] = g.surface_pressure.mean(); out["rain"] = g.precipitation.sum(min_count=1); out["wind"] = g.wind_speed_10m.mean()
    out["u"] = u.resample("D").mean(); out["v"] = v.resample("D").mean(); out["cloud"] = g.cloud_cover.mean()
    return out
if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "pilot":
        import m3_data as M
        act = M.D[["temp", "rh", "pres", "rain", "wind", "u", "v", "cloud", "trange"]]
        rows = []
        for model in ("gfs_seamless", "ecmwf_ifs025"):
            for lead in (1, 3, 7):
                f = daily(fetch(model, lead, "2025-01-01", "2025-06-30")); j = f.join(act, rsuffix="_act", how="inner").dropna()
                for c in ("temp", "wind", "u", "v", "rh", "cloud", "pres"):
                    e = j[c] - j[c + "_act"]; rows.append(dict(model=model, lead=lead, var=c, rmse=float(np.sqrt((e ** 2).mean())), corr=float(j[c].corr(j[c + "_act"])), n=len(j)))
                print(model, lead, "done", flush=True); time.sleep(8)
        r = pd.DataFrame(rows); r.round(3).to_csv(os.path.join(ROOT, "results", "forecast_pilot_vs_era5.csv"), index=False)
        print(r.pivot_table(index=["var", "lead"], columns="model", values="corr").round(3).to_string()); print(r.pivot_table(index=["var", "lead"], columns="model", values="rmse").round(2).to_string())
    elif mode == "download":
        model = sys.argv[2]; parts = []
        for lead in range(1, 8):
            f = daily(fetch(model, lead, "2024-01-01", "2026-10-27")); f.columns = [f"{c}__{lead}" for c in f.columns]; parts.append(f); print("lead", lead, "ok", len(f), "days; first non-null", f.dropna().index.min(), flush=True); time.sleep(20)
        out = pd.concat(parts, axis=1); out.index.name = "date"; out.round(4).to_csv(os.path.join(DATA, f"forecasts_{model}_daily.csv")); print("saved", out.shape)
