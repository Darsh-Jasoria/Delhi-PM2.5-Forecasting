"""
Milestone 3 - deep-learning part: LSTM and GRU for PM2.5 (run as its own process by milestone3_analysis.py).
Multi-horizon sequence models: a window of the last L days of 21 standardised daily features (PM2.5 and its intraday
statistics, PM10, NO2, CO, AQI, weather, season) -> change in log PM2.5 for days +1..+7. Five-seed average.
Hyper-parameters (hidden size, window, dropout) are tuned on two 90-day blocks that end before fold 1 starts.
"""
import json, pickle
from m3_data import *
import torch, torch.nn as nn
torch.manual_seed(SEED); torch.set_num_threads(4)

class RNN(nn.Module):
    def __init__(s, nf, hid, kind, drop):
        super().__init__(); s.rnn = (nn.LSTM if kind == "lstm" else nn.GRU)(nf, hid, batch_first=True); s.drop = nn.Dropout(drop); s.fc = nn.Linear(hid, HMAX)
    def forward(s, x): o, _ = s.rnn(x); return s.fc(s.drop(o[:, -1]))
def dl_arrays(L, tr_end):
    """standardise on train period; windows ending at every origin; target = ly(d+k) - ly(d), k=1..7"""
    pos = {d: i for i, d in enumerate(D.index)}; te = pos[tr_end]
    mu, sd = SDL.iloc[:te + 1].mean(), SDL.iloc[:te + 1].std() + 1e-9
    S = ((SDL - mu) / sd).values.astype("float32"); ly = D.ly.values
    X = np.stack([S[i - L + 1:i + 1] for i in range(L - 1, len(S))]); origins = np.arange(L - 1, len(S))
    Y = np.full((len(origins), HMAX), np.nan, "float32")
    for j, i in enumerate(origins):
        if i + HMAX < len(ly): Y[j] = ly[i + 1:i + HMAX + 1] - ly[i]
    if FC:                                  # 7-day weather-forecast vector issued at the origin, repeated at every time step
        Fr = np.concatenate([FCD[k][FC_NAMES].reindex(D.index + pd.Timedelta(days=k)).values for k in range(1, HMAX + 1)], axis=1)
        fm, fs = np.nanmean(Fr[:te + 1], 0), np.nanstd(Fr[:te + 1], 0) + 1e-9
        Fs = np.nan_to_num(((Fr - fm) / fs).astype("float32"))
        X = np.concatenate([X, np.repeat(Fs[origins][:, None, :], L, axis=1)], axis=2)
    return X, Y, origins, te
def train_rnn(kind, hid, L, drop, tr_end, seed, epochs=120, patience=12, lr=2e-3, n_epochs=None, return_best_epoch=False):
    X, Y, origins, te = dl_arrays(L, tr_end)
    okm = (origins + HMAX <= te) & ~np.isnan(Y).any(1); Xt, Yt = X[okm], Y[okm]
    torch.manual_seed(seed); np.random.seed(seed)
    net = RNN(X.shape[2], hid, kind, drop); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3); lossf = nn.MSELoss()
    nv = 60 if n_epochs is None else 0; Xtr, Ytr = (Xt[:-nv], Yt[:-nv]) if nv else (Xt, Yt)
    Xv, Yv = (torch.tensor(Xt[-nv:]), torch.tensor(Yt[-nv:])) if nv else (None, None)
    Xtr_t, Ytr_t = torch.tensor(Xtr), torch.tensor(Ytr); best, best_ep, bad, best_state = 1e9, 0, 0, None
    for ep in range(1, (n_epochs or epochs) + 1):
        net.train(); perm = torch.randperm(len(Xtr_t))
        for b in range(0, len(perm), 64):
            idx = perm[b:b + 64]; opt.zero_grad(); loss = lossf(net(Xtr_t[idx]), Ytr_t[idx]); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if nv:
            net.eval()
            with torch.no_grad(): v = lossf(net(Xv), Yv).item()
            if v < best - 1e-5: best, best_ep, bad = v, ep, 0; best_state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience: break
    if nv: net.load_state_dict(best_state)
    net.eval()
    if return_best_epoch: return net, best_ep, best
    return net
def rnn_predict(net, L, tr_end, first_origin, last_origin):
    X, Y, origins, te = dl_arrays(L, tr_end); m = (origins >= first_origin) & (origins <= last_origin)
    with torch.no_grad(): out = net(torch.tensor(X[m])).numpy()
    return out, origins[m]


def tune_dl():
    path = os.path.join(RES, "dl_params.json")
    if os.path.exists(path): return json.load(open(path))
    grid = [(hid, L, dr) for hid in (32, 64) for L in (7, 14, 28) for dr in (0.1, 0.3)] if not FAST else [(32, 14, .1), (64, 14, .3)]
    blocks = [TUNE_END - pd.Timedelta(days=90), TUNE_END]; P = {"grid": {}}; pos = {d: i for i, d in enumerate(D.index)}; ly = D.ly.values
    for kind in ("lstm", "gru"):
        best = None; rows = []
        for hid, L, dr in grid:
            errs = []
            for bend in blocks:
                ts = bend - pd.Timedelta(days=89); tr_end = ts - pd.Timedelta(days=1); i0 = pos[ts] - 1
                net = train_rnn(kind, hid, L, dr, tr_end, seed=0, epochs=40 if FAST else 100)
                out, org = rnn_predict(net, L, tr_end, i0, i0 + 89); e = []
                for o, i in zip(out, org):
                    for k in range(HMAX):
                        if i + 1 + k <= i0 + 90: e.append(ly[i] + o[k] - ly[i + 1 + k])
                errs.append(float(np.sqrt(np.mean(np.square(e)))))
            sc = float(np.mean(errs)); rows.append(dict(hidden=hid, window=L, dropout=dr, val_rmse_log=round(sc, 4)))
            if best is None or sc < best[0]: best = (sc, dict(hidden=hid, window=L, dropout=dr))
            log(" ", kind, hid, L, dr, round(sc, 4))
        P[kind] = dict(params=best[1], val_rmse_log=round(best[0], 4)); P["grid"][kind] = rows
    json.dump(P, open(path, "w"), indent=1); return P

def run_fold_dl(fi, P):
    path = os.path.join(CACHE, f"dl_fold{fi}.pkl")
    if os.path.exists(path): return
    ts, te = FOLDS[fi]; tr_end = ts - pd.Timedelta(days=1); idx = pd.date_range(ts, te); n = len(idx)
    pos = {d: i for i, d in enumerate(D.index)}; i0 = pos[ts] - 1; ly = D.ly.values; pred = {1: {}, 7: {}}; hist = {}
    for kind in ("lstm", "gru"):
        pr = P[kind]["params"]; hid, L, dr = pr["hidden"], pr["window"], pr["dropout"]; outs = []; eps = []
        for sd in range(2 if FAST else 5):     # epochs found by early stopping on the last 60 training days, then refit on all training data
            _, ep, _ = train_rnn(kind, hid, L, dr, tr_end, SEED + sd, return_best_epoch=True, epochs=40 if FAST else 120); eps.append(ep)
            net = train_rnn(kind, hid, L, dr, tr_end, SEED + sd, n_epochs=max(ep, 5))
            o, org = rnn_predict(net, L, tr_end, i0, i0 + n - 1); outs.append(o)
        o = np.mean(outs, 0); hist[kind] = eps
        pred[1][kind.upper()] = pd.Series([ly[i] + o[j, 0] for j, i in enumerate(org)], index=idx)
        pred[7][kind.upper()] = pd.Series([ly[i0 + (j // HMAX) * HMAX] + o[(j // HMAX) * HMAX, j % HMAX] for j in range(n)], index=idx)
        log(f"fold {fi}", kind, "done; best epochs", eps)
    pickle.dump(dict(pred=pred, epochs=hist), open(path, "wb"))

if __name__ == "__main__":
    P = tune_dl()
    for fi in (4, 3, 2, 1): run_fold_dl(fi, P)
    log("DL DONE")
