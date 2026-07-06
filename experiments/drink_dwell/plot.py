"""THE GRAPH: worst-N reps by proxy21 error, one panel each — cup→head distance (tracked +
mocap), both models' P(drink), and the dwell bars (truth / proxy21 / base17). LOOK before
modelling more. Reads cache/results.json; retrains only the worst reps' folds (cached to
cache/worst_preds.json) so replotting is GPU-free.

    python experiments/drink_dwell/plot.py [--n 9] [--epochs 250] [--refresh]
Writes ../drink_study/slides/worst_proxy21_grid.png
"""
from __future__ import annotations
import sys as _s, pathlib as _p
_s.path.insert(0, str(_p.Path(__file__).resolve().parent))
import argparse, glob, json, os
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import features as F
from model import TCN, resample, span_from_prob, SEQ, DEV, HZ

HERE = _p.Path(__file__).resolve().parent
RESULTS = HERE / "cache" / "results.json"
PRED_CACHE = HERE / "cache" / "worst_preds.json"
OUT = HERE.parents[0] / "drink_study" / "slides" / "worst_proxy21_grid.png"
FEATS = [("proxy21", "fx21"), ("base17", "fx17")]


def worst_videos(n):
    d = json.load(open(RESULTS))
    ip = d["perrep_cols"].index("proxy21"); ib = d["perrep_cols"].index("base17")
    rows = sorted(((v[ip], v[ib], k) for k, v in d["perrep"].items()), reverse=True)
    return [k for _, _, k in rows[:n]], {k: (p, b) for p, b, k in rows}


def preds_for(reps, held, key, epochs):
    import torch.nn as nn
    trn = [r for r in reps if r["pid"] != held]; te = [r for r in reps if r["pid"] == held]
    Xtr = torch.tensor(np.stack([r[key + "_seq"] for r in trn])).transpose(1, 2).to(DEV)
    Mtr = torch.tensor(np.stack([r["mx"] for r in trn])).to(DEV)
    clf = TCN(Xtr.shape[1]).to(DEV)
    opt = torch.optim.Adam(clf.parameters(), lr=2e-3, weight_decay=1e-4)
    pos = Mtr.mean().clamp(1e-3, 1 - 1e-3); w = ((1 - pos) / pos).item()
    lf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w, device=DEV)); clf.train()
    for _ in range(epochs):
        opt.zero_grad(); lf(clf(Xtr).squeeze(1), Mtr).backward(); opt.step()
    clf.eval()
    with torch.no_grad():
        ptr = torch.sigmoid(clf(Xtr).squeeze(1)).cpu().numpy()
        from model import errs
        best_thr, best = 0.5, 1e9
        for thr in np.arange(0.15, 0.96, 0.05):
            de = [errs(r["tsp"], span_from_prob(resample(ptr[k], r["T"]), thr)) for k, r in enumerate(trn)]
            if np.mean(de) < best:
                best, best_thr = np.mean(de), thr
        out = {}
        for r in te:
            x = torch.tensor(r[key + "_seq"][None]).transpose(1, 2).to(DEV)
            pr = resample(torch.sigmoid(clf(x).squeeze(1))[0].cpu().numpy(), r["T"])
            out[r["video"]] = ([round(float(x), 4) for x in pr], float(best_thr))
    return out


def load_preds(reps, pids, byv, refresh):
    cache = json.load(open(PRED_CACHE)) if (PRED_CACHE.exists() and not refresh) else {}
    have = lambda v: v in cache and all(m in cache[v] for m, _ in FEATS)
    need = sorted({byv[v]["pid"] for v in byv if not have(v) and byv[v]["pid"] in pids})
    if need:
        print(f"training {len(need)} fold(s) x {len(FEATS)} models: {need}", flush=True)
        for held in need:
            for m, key in FEATS:
                for v, (pr, thr) in preds_for(reps, held, key, EPOCHS).items():
                    cache.setdefault(v, {})[m] = {"prob": pr, "thr": thr}
        PRED_CACHE.parent.mkdir(exist_ok=True)
        json.dump(cache, open(PRED_CACHE, "w"))
        print(f"cached -> {PRED_CACHE}", flush=True)
    else:
        print(f"all served from {PRED_CACHE} (no training)", flush=True)
    return {v: {m: (np.array(cache[v][m]["prob"]), cache[v][m]["thr"]) for m, _ in FEATS}
            for v in byv if have(v) and byv[v]["pid"] in pids}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    global EPOCHS; EPOCHS = a.epochs
    want, errmap = worst_videos(a.n)
    reps = []
    for f in sorted(glob.glob(str(F.FUSED_DIR / "*.npz"))):
        r = F.build_rep(np.load(f, allow_pickle=True))
        if r is None:
            continue
        for _, key in FEATS:
            r[key + "_seq"] = resample(r[key], SEQ).astype(np.float32)
        s, e = r["tsp"]; m = np.zeros(r["T"]); m[s:e] = 1.0
        r["mx"] = resample(m, SEQ).astype(np.float32)
        reps.append(r)
    byv = {r["video"]: r for r in reps}
    want = [v for v in want if v in byv]
    pids = sorted({byv[v]["pid"] for v in want})
    probs = load_preds(reps, pids, byv, a.refresh)

    ncol = 3; nrow = int(np.ceil(len(want) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 3.4 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, v in enumerate(want):
        r = byv[v]; T = r["T"]; t = np.arange(T) / HZ
        pr_p, thr_p = probs[v]["proxy21"]; sp_p = span_from_prob(pr_p, thr_p)
        pr_b, thr_b = probs[v]["base17"]; sp_b = span_from_prob(pr_b, thr_b)
        dist = r["fx21"][:, 17] * 600.0                       # de-normalise head dist channel
        ax = axes.flat[i]; ax.axis("on")
        ax.plot(t, dist, color="#1a8a3a", lw=1.8)
        dmax = max(np.nanmax(dist) * 1.08, 1)
        ax.set_ylim(0, dmax * 1.30); ax.set_ylabel("cup→head (mm)", color="#1a8a3a", fontsize=10)
        ax2 = ax.twinx()
        ax2.plot(t, pr_p, color="#c0392b", lw=1.6)
        ax2.plot(t, pr_b, color="#888", lw=1.2, ls="--")
        ax2.set_ylim(0, 1.30); ax2.set_ylabel("P(drink)", color="#c0392b", fontsize=10)
        for bi, (nm, sp, col) in enumerate([("truth", r["tsp"], "#e8930a"),
                                            ("proxy21", sp_p, "#c0392b"), ("base17", sp_b, "#777")]):
            y = 1.05 + bi * 0.075
            ax2.axhline(y, color="#ddd", lw=0.6)
            if sp:
                ax2.plot([sp[0] / HZ, sp[1] / HZ], [y, y], color=col, lw=5,
                         solid_capstyle="butt", clip_on=False)
            ax2.text(-0.01, y, nm, transform=ax2.get_yaxis_transform(), ha="right",
                     va="center", fontsize=7.5, color=col)
        p = v.split("_")[0]; side = "R" if "right" in v else "L"
        ax.set_title(f"{p} {side}   proxy21 {errmap[v][0]:.0f} / base17 {errmap[v][1]:.0f} ms",
                     fontsize=11, fontweight="bold", pad=24)
        ax.set_xlabel("time (s)", fontsize=10)
    fig.suptitle("Worst reps — proxy21 (cup→head) vs base17 — dwell bars: truth / proxy21 / base17",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
