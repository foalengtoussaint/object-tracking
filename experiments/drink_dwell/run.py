"""THE ENTRY POINT: reproduce the drink-dwell numbers, LOPO (leave-one-participant-out).

Trains a per-frame TCN for base17 (video-only) and proxy21 (+head distance) against the
mocap cup→head dwell truth, scores |duration error| in ms. Reproduces the headline:
    proxy21 ~77ms   vs   base17 ~118ms   (n≈666)

    python experiments/drink_dwell/run.py [--epochs 250]
Writes cache/results.json (per-rep errors + summary). Cache-first for FEATURES via the
lopo_fused/track3d caches (no GPU for detection/tracking); trains the tiny TCN (CPU ok).
"""
from __future__ import annotations
import sys as _s, pathlib as _p
_s.path.insert(0, str(_p.Path(__file__).resolve().parent))
import argparse, glob, json, time
import numpy as np
import torch, torch.nn as nn

import features as F
from model import TCN, resample, span_from_prob, errs, SEQ, DEV, HZ

HERE = _p.Path(__file__).resolve().parent
OUT = HERE / "cache" / "results.json"
METHODS = [("base17", "fx17"), ("proxy21", "fx21")]


def build_all():
    reps = []
    files = sorted(glob.glob(str(F.FUSED_DIR / "*.npz")))
    print(f"building {len(files)} reps from cache (no GPU)...", flush=True)
    dropped = 0
    for i, f in enumerate(files):
        r = F.build_rep(np.load(f, allow_pickle=True))
        if r is None:
            dropped += 1
        else:
            for nm, key in METHODS:
                r[key + "_seq"] = resample(r[key], SEQ).astype(np.float32)
            r["mx"] = np.zeros(SEQ, np.float32)
            s, e = r["tsp"]
            m = np.zeros(r["T"]); m[s:e] = 1.0
            r["mx"] = resample(m, SEQ).astype(np.float32)
            reps.append(r)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(files)}  (kept {len(reps)}, dropped {dropped})", flush=True)
    print(f"{len(reps)} reps usable ({dropped} dropped: no occ/head/pairing/dwell)", flush=True)
    return reps


def train_fold(reps, held, key, epochs):
    trn = [r for r in reps if r["pid"] != held]
    te = [r for r in reps if r["pid"] == held]
    Xtr = torch.tensor(np.stack([r[key + "_seq"] for r in trn])).transpose(1, 2).to(DEV)
    Mtr = torch.tensor(np.stack([r["mx"] for r in trn])).to(DEV)
    clf = TCN(Xtr.shape[1]).to(DEV)
    opt = torch.optim.Adam(clf.parameters(), lr=2e-3, weight_decay=1e-4)
    pos = Mtr.mean().clamp(1e-3, 1 - 1e-3); w = ((1 - pos) / pos).item()
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w, device=DEV)); clf.train()
    for _ in range(epochs):
        opt.zero_grad(); lossf(clf(Xtr).squeeze(1), Mtr).backward(); opt.step()
    clf.eval()
    with torch.no_grad():
        ptr = torch.sigmoid(clf(Xtr).squeeze(1)).cpu().numpy()
        best_thr, best = 0.5, 1e9
        for thr in np.arange(0.15, 0.96, 0.05):
            de = [errs(r["tsp"], span_from_prob(resample(ptr[k], r["T"]), thr))
                  for k, r in enumerate(trn)]
            if np.mean(de) < best:
                best, best_thr = np.mean(de), thr
        out = {}
        for r in te:
            x = torch.tensor(r[key + "_seq"][None]).transpose(1, 2).to(DEV)
            pr = resample(torch.sigmoid(clf(x).squeeze(1))[0].cpu().numpy(), r["T"])
            out[r["video"]] = span_from_prob(pr, best_thr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=250)
    a = ap.parse_args()
    reps = build_all()
    pids = sorted({r["pid"] for r in reps})
    print(f"device {DEV}; {len(pids)} folds; training {', '.join(m for m,_ in METHODS)}", flush=True)
    P = {nm: {} for nm, _ in METHODS}
    t0 = time.time()
    for fi, held in enumerate(pids):
        for nm, key in METHODS:
            P[nm].update(train_fold(reps, held, key, a.epochs))
        el = time.time() - t0
        print(f"  [{fi+1}/{len(pids)}] {held}  ({el:.0f}s, ~{el/(fi+1)*(len(pids)-fi-1):.0f}s left)",
              flush=True)

    tsp = {r["video"]: r["tsp"] for r in reps}
    common = [k for k in P["base17"] if k in P["proxy21"]]
    D = {nm: np.array([errs(tsp[k], P[nm][k]) for k in common]) for nm, _ in METHODS}
    print(f"\n=== drink-dwell vs mocap cup→head truth, LOPO {len(common)} reps ===")
    print(f"  {'method':<9}{'mean':>7}{'p50':>6}{'p90':>6}{'p99':>7}{'max':>7}")
    for nm, _ in METHODS:
        de = D[nm]
        print(f"  {nm:<9}{de.mean():>7.0f}{np.percentile(de,50):>6.0f}{np.percentile(de,90):>6.0f}"
              f"{np.percentile(de,99):>7.0f}{de.max():>7.0f}", flush=True)
    OUT.parent.mkdir(exist_ok=True)
    js = {"n": len(common),
          "summary": {nm: {k: float(np.percentile(D[nm], p) if k != "mean" else D[nm].mean())
                           for k, p in [("mean", 0), ("p50", 50), ("p90", 90),
                                        ("p99", 99), ("max", 100)]}
                      for nm, _ in METHODS},
          "perrep_cols": [nm for nm, _ in METHODS],
          "perrep": {common[i]: [float(D[nm][i]) for nm, _ in METHODS] for i in range(len(common))}}
    json.dump(js, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
