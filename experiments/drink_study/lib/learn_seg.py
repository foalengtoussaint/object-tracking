"""Learned drink-dwell segmenter vs the tuned geometric gate -- 3 heads, LOPO.

Motivation: the tuned gate (DRINK_SPEED/disp 150/150, tune_seg.py) fixed the median +
bias but leaves a heavy TAIL (|dur|err p95 517ms, max 3617; boundary p95 275). Those
tail reps are where a SCALAR speed gate structurally can't win -- P20 wants a LOWER gate
(slow place-down read as dwell), P10/P16 want HIGHER (noisy in-dwell track). A model that
reads the local SHAPE can make that call per-rep. This tests whether it actually does,
LEAVE-ONE-PARTICIPANT-OUT (the only trustworthy eval here -- every single-split win in this
project evaporated on LOPO: see [[project_velocity_fill_gap]]).

Three heads (all trained on the SAME features / truth, LOPO):
  A clf      : per-frame P(drink) TCN; span = longest run of P>thr (thr LOPO-tuned).
  B bound    : regress dwell (onset,offset) as fractions of the movement window.
  C resid    : predict a per-rep gate ADJUSTMENT delta; span = geometric gate at 150+delta.

Truth = mocap-track dwell = segment_cup_only(true) drink span (same target tune_seg used).
Baseline = tuned geometric gate on the fused track. Cache-only reload of cache/lopo_fused,
NO GPU needed (tiny TCN, CPU ok). Per-rep predictions cached to cache/learn_seg.json.

    python experiments/drink_study/learn_seg.py [--epochs 250]
Run from repo root. Per-fold prints (flush). ~few min on CPU.
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import argparse, glob, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "experiments" / "drink_study"))
import segment_cup_only as S

from _paths import CACHE as _C
CACHE = _C / "lopo_fused"
OUT = _C / "learn_seg.json"
HZ = 60.0
SEQ = 256                       # resample grid
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DEF = dict(drink_speed=120, drink_disp_pad=90, fwd_on=15, back_off=10)     # truth gate
TUN = dict(drink_speed=150, drink_disp_pad=150, fwd_on=15, back_off=10)    # baseline gate


def geo_span(xyz, **kw):
    r = S.segment_cup_only(xyz, fps=HZ, **kw)
    rr = r["drink_runs"]
    return (rr[0][0], rr[-1][1]) if rr else None


def resample(a, n_out):
    """(T,) or (T,k) linear resample to (n_out, ...)."""
    a = np.asarray(a, float)
    x0 = np.linspace(0, 1, len(a)); x1 = np.linspace(0, 1, n_out)
    if a.ndim == 1:
        return np.interp(x1, x0, a)
    return np.stack([np.interp(x1, x0, a[:, k]) for k in range(a.shape[1])], 1)


def build_rep(d):
    """Per-frame features from the fused track: the filtered scalars the segmenter reads
    PLUS raw (unfiltered) speed and the 3D velocity / disp-from-rest VECTORS in the rep-local
    basis (direction, not just magnitude). Truth dwell mask + span, movement window, native T.

    Why 3D + raw: the scalar filtered speed is exactly the gate's blurred view -- it can't
    tell a genuine dwell (hover near mouth) from a slow place-down (moving DOWN/away) since
    both have low speed MAGNITUDE. The velocity direction can. And raw speed keeps the high-
    freq structure the 6Hz filter removes before the gate sees it."""
    fused = d["fused"]; cons = d["cons"]; valid = d["valid"]
    basis = d["basis"]; rest = d["rest"]
    T = len(fused)
    # filtered speed + disp exactly like the segmenter (kept for continuity)
    seg = S.segment_cup_only(fused, fps=HZ, **TUN)
    speed = seg["speed"]; disp = seg["disp"]
    peak = disp.max() if T else 1.0
    dtp = peak - disp                                    # distance below peak (0 at apex)
    accel = np.r_[0.0, np.diff(speed)]
    gap = (~valid).astype(float)
    win = np.zeros(T); gs, ge = seg["grasp"]
    win[gs:ge] = 1.0
    # --- rep-local 3D vectors (direction-aware, RAW, no 6Hz filter) ---
    loc = (fused - rest) @ basis.T                       # (T,3) position rel rest, local axes
    vel = np.vstack([np.zeros(3), np.diff(loc, axis=0)]) * HZ   # (T,3) mm/s raw local velocity
    raw_speed = np.linalg.norm(vel, axis=1)              # unfiltered speed magnitude
    P = peak + 1e-6; V = 200.0
    feats = np.stack([
        speed / V, disp / P, dtp / P, accel / V, gap, win,   # 0-5: original filtered scalars
        raw_speed / V,                                        # 6: RAW speed magnitude
        vel[:, 0] / V, vel[:, 1] / V, vel[:, 2] / V,          # 7-9: RAW 3D velocity (local)
        loc[:, 0] / P, loc[:, 1] / P, loc[:, 2] / P,          # 10-12: 3D disp-from-rest (local)
    ], 1).astype(np.float32)                              # (T, 13)
    # truth
    tsp = geo_span(d["true"])
    tmask = np.zeros(T, np.float32)
    if tsp is not None:
        tmask[tsp[0]:tsp[1]] = 1.0
    return dict(feats=feats, tmask=tmask, tsp=tsp, T=T, win=(gs, ge),
                fused=fused, true=d["true"], pid=str(d["pid"]), video=str(d["video"]))


class TCN(nn.Module):
    def __init__(self, cin, ch=48, layers=5, nout=1, p=0.1):
        super().__init__()
        L = []
        c = cin
        for i in range(layers):
            dl = 2 ** i
            L += [nn.Conv1d(c, ch, 3, padding=dl, dilation=dl), nn.ReLU(), nn.Dropout(p)]
            c = ch
        self.body = nn.Sequential(*L)
        self.head = nn.Conv1d(ch, nout, 1)
        self.pool_head = nn.Linear(ch, nout)             # for global (bound/resid) outputs
        self.nout = nout

    def forward(self, x, mode="frame"):
        h = self.body(x)                                 # (B,ch,T)
        if mode == "frame":
            return self.head(h)                          # (B,nout,T)
        g = h.mean(-1) + h.max(-1).values                # (B,ch) global pool
        return self.pool_head(g)                          # (B,nout)


def span_from_prob(prob, thr):
    """Longest run of prob>thr -> (s,e) or None."""
    m = prob > thr
    best = None; i = 0; T = len(m)
    while i < T:
        if m[i]:
            j = i
            while j < T and m[j]:
                j += 1
            if best is None or (j - i) > (best[1] - best[0]):
                best = (i, j)
            i = j
        else:
            i += 1
    return best


def errs(tsp, vsp, T):
    """(|durErr|ms, boundary ms). Miss => whole true dwell as error."""
    td = (tsp[1] - tsp[0]) / HZ * 1000
    if vsp is None:
        return td, td
    vd = (vsp[1] - vsp[0]) / HZ * 1000
    de = abs(vd - td)
    bn = (abs(vsp[0] - tsp[0]) + abs(vsp[1] - tsp[1])) / 2 / HZ * 1000
    return de, bn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=250)
    args = ap.parse_args()
    print(f"device {DEV}; loading cache...", flush=True)
    reps = []
    for f in sorted(glob.glob(str(CACHE / "*.npz"))):
        d = np.load(f, allow_pickle=True)
        r = build_rep(d)
        if r["tsp"] is not None:                          # need a truth dwell to score
            reps.append(r)
    pids = sorted({r["pid"] for r in reps})
    print(f"{len(reps)} reps (with truth dwell), {len(pids)} participants", flush=True)

    # precompute resampled tensors
    for r in reps:
        r["fx"] = resample(r["feats"], SEQ).astype(np.float32)       # (SEQ,6)
        r["mx"] = resample(r["tmask"], SEQ).astype(np.float32)       # (SEQ,)
        gs, ge = r["win"]; wl = max(ge - gs, 1)
        # bound target = onset/offset as fraction of the movement window
        s, e = r["tsp"]
        r["bt"] = np.array([(s - gs) / wl, (e - gs) / wl], np.float32).clip(0, 1)

    cin = reps[0]["fx"].shape[1]
    # storage of per-rep predictions per method
    P = {"tuned": {}, "clf": {}, "bound": {}, "resid": {}}
    t0 = time.time()
    for fi, held in enumerate(pids):
        trn = [r for r in reps if r["pid"] != held]
        te = [r for r in reps if r["pid"] == held]
        Xtr = torch.tensor(np.stack([r["fx"] for r in trn])).transpose(1, 2).to(DEV)
        Mtr = torch.tensor(np.stack([r["mx"] for r in trn])).to(DEV)         # (N,SEQ)
        Btr = torch.tensor(np.stack([r["bt"] for r in trn])).to(DEV)         # (N,2)

        # --- A: per-frame classifier ---
        clf = TCN(cin, nout=1).to(DEV)
        opt = torch.optim.Adam(clf.parameters(), lr=2e-3, weight_decay=1e-4)
        pos = Mtr.mean().clamp(1e-3, 1 - 1e-3); w = ((1 - pos) / pos).item()
        lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w, device=DEV))
        clf.train()
        for _ in range(args.epochs):
            opt.zero_grad()
            out = clf(Xtr, "frame").squeeze(1)                              # (N,SEQ)
            lossf(out, Mtr).backward(); opt.step()

        # --- B: boundary regressor ---
        bnd = TCN(cin, nout=2).to(DEV)
        ob = torch.optim.Adam(bnd.parameters(), lr=2e-3, weight_decay=1e-4)
        bnd.train()
        for _ in range(args.epochs):
            ob.zero_grad()
            pb = torch.sigmoid(bnd(Xtr, "global"))                         # (N,2)
            ((pb - Btr) ** 2).mean().backward(); ob.step()

        # --- C: residual gate corrector (predict delta on drink_speed) ---
        res = TCN(cin, nout=1).to(DEV)
        orc = torch.optim.Adam(res.parameters(), lr=2e-3, weight_decay=1e-4)
        # target delta: per rep, the gate offset (in mm/s from 150) that best matches
        # truth -- precompute by a tiny 1-D search on train reps
        for r in trn:
            if "dstar" not in r:
                best = 0; bestg = 150
                for g in range(90, 320, 10):
                    vsp = geo_span(r["fused"], **{**TUN, "drink_speed": g})
                    de, _ = errs(r["tsp"], vsp, r["T"])
                    if bestg == 150 or de < best:
                        best = de; bestg = g
                r["dstar"] = (bestg - 150) / 100.0        # scaled
        Dtr = torch.tensor(np.array([[r["dstar"]] for r in trn], np.float32)).to(DEV)
        res.train()
        for _ in range(args.epochs):
            orc.zero_grad()
            pd = res(Xtr, "global")                                        # (N,1)
            ((pd - Dtr) ** 2).mean().backward(); orc.step()

        # LOPO-select the classifier threshold on TRAIN reps
        clf.eval(); bnd.eval(); res.eval()
        with torch.no_grad():
            ptr = torch.sigmoid(clf(Xtr, "frame").squeeze(1)).cpu().numpy()
            best_thr, best_de = 0.5, 1e9
            for thr in np.arange(0.15, 0.96, 0.05):
                des = []
                for k, r in enumerate(trn):
                    pr = resample(ptr[k], r["T"])
                    vsp = span_from_prob(pr, thr)
                    de, _ = errs(r["tsp"], vsp, r["T"]); des.append(de)
                m = np.mean(des)
                if m < best_de:
                    best_de, best_thr = m, thr

            # evaluate held-out
            for r in te:
                x = torch.tensor(r["fx"][None]).transpose(1, 2).to(DEV)
                pr = resample(torch.sigmoid(clf(x, "frame").squeeze(1))[0].cpu().numpy(), r["T"])
                P["clf"][r["video"]] = span_from_prob(pr, best_thr)
                pb = torch.sigmoid(bnd(x, "global"))[0].cpu().numpy()
                gs, ge = r["win"]; wl = max(ge - gs, 1)
                s = int(gs + pb[0] * wl); e = int(gs + pb[1] * wl)
                P["bound"][r["video"]] = (s, e) if e > s else None
                pd = float(res(x, "global")[0, 0].cpu().numpy()) * 100.0
                g = float(np.clip(150 + pd, 90, 320))
                P["resid"][r["video"]] = geo_span(r["fused"], **{**TUN, "drink_speed": g})
                P["tuned"][r["video"]] = geo_span(r["fused"], **TUN)
        el = time.time() - t0
        print(f"  [{fi+1}/{len(pids)}] held {held}: {len(te)} reps  thr={best_thr:.2f}  "
              f"({el:.0f}s, ~{el/(fi+1)*(len(pids)-fi-1):.0f}s left)", flush=True)

    # score
    tsp = {r["video"]: r["tsp"] for r in reps}; Tn = {r["video"]: r["T"] for r in reps}
    common = sorted(set.intersection(*[set(P[m]) for m in P]) & set(tsp))
    print(f"\n=== LEARNED SEGMENTER (LOPO), {len(common)} reps ===")
    print(f"  {'method':<8}{'|dur|mean':>10}{'|dur|p50':>9}{'|dur|p90':>9}{'|dur|p95':>9}"
          f"{'bnd_mean':>9}{'bnd_p90':>9}")
    summ = {}
    for m in ("tuned", "clf", "bound", "resid"):
        de = []; bn = []
        for k in common:
            d, b = errs(tsp[k], P[m][k], Tn[k]); de.append(d); bn.append(b)
        de = np.array(de); bn = np.array(bn)
        summ[m] = dict(dur_mean=float(de.mean()), dur_p50=float(np.percentile(de, 50)),
                       dur_p90=float(np.percentile(de, 90)), dur_p95=float(np.percentile(de, 95)),
                       bnd_mean=float(bn.mean()), bnd_p90=float(np.percentile(bn, 90)))
        print(f"  {m:<8}{de.mean():>10.0f}{np.percentile(de,50):>9.0f}{np.percentile(de,90):>9.0f}"
              f"{np.percentile(de,95):>9.0f}{bn.mean():>9.0f}{np.percentile(bn,90):>9.0f}", flush=True)
    json.dump({"n": len(common), "summary": summ,
               "pred": {m: {k: (list(P[m][k]) if P[m][k] else None) for k in common} for m in P}},
              open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
