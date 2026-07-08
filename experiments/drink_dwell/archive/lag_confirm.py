"""CONFIRM the velocity-direction lag fix is real, not the metric fooling itself.

The trap: if we CHOOSE the lag to minimise the drink angle and then REPORT that same angle,
it looks better by construction. So we separate selection from evaluation three ways, none
gameable by the selection:

  (1) HELD-OUT: choose lag on the FIRST half of moving frames, evaluate angle on the SECOND
      half. Improvement that survives on unseen frames = real timing fix, not overfitting.
  (2) POSITION RMS: the lag is chosen on velocity DIRECTION; the Kabsch position RMS (absolute
      mm) is an independent quantity the selection never touches. If it ALSO drops, corroborated.
  (3) CURVE SHAPE: print angle-vs-lag. A true timing error is a smooth single valley at the
      shifted lag; noise is jagged. Negative-control (already-good) reps should be flat, shift~0.

Runs on a named set of reps (P16 cluster + P24/P23 + good controls). Cache-only.
"""
from __future__ import annotations
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # drink_dwell root (archived)
import features as F
from mocap import load_trial, resample as resample3d, VIDEO_FPS
from truth import dwell_truth
from velfit import fit_source

SPEED_MM_S = 80.0
HZ = 60.0
SEARCH = 6

# reps to inspect: bad (P16/P24/P23 from the lag-check worst list) + good controls
WANT = ["105728", "105518", "110110", "105716",       # P16 bad
        "105702", "105744",                           # P24 bad
        "151411",                                      # P23 bad
        "P02", "P05"]                                  # good controls (any rep from these pids)


def _synced(cup_world, mocap_centroid, rate, lag):
    vr = resample3d(cup_world, VIDEO_FPS)
    mr = resample3d(mocap_centroid, rate)
    if lag >= 0:
        v = vr[lag:]; mo = mr[:len(v)]
    else:
        mo = mr[-lag:]; v = vr[:len(mo)]
    L = min(len(v), len(mo))
    return v[:L], mo[:L]


GOOD_ANG = 20.0     # a frame's motion direction "agrees" if within this many degrees

def _fit_and_eval(src, fused, cent, rate, lag, dw, half=None):
    """Return (GOOD-FRAME FRACTION on the eval half, median angle, position rms) at this lag.
    good-frame frac = fraction of eval frames with velocity angle < GOOD_ANG -- the metric the
    user cares about (how many frames the truth is usable), NOT the average. half: None=all,
    0=first, 1=second of the MOVING drink frames (train/test split)."""
    fit = F.mocap_to_w0(src, cent, rate, lag)
    if fit is None:
        return np.nan, np.nan, np.nan
    R, t, rms = fit
    mmc, omc_lab = _synced(fused, cent, rate, lag)
    omc = omc_lab @ R.T + t
    vm = np.diff(mmc, axis=0) * HZ; vo = np.diff(omc, axis=0) * HZ
    sm = np.linalg.norm(vm, axis=1); so = np.linalg.norm(vo, axis=1)
    moving = (sm > SPEED_MM_S) & (so > SPEED_MM_S) & np.isfinite(sm) & np.isfinite(so)
    drink = np.zeros(len(vm), bool)
    sp = dw.span_at(len(vm)) if dw.span else None
    if sp:
        drink[sp[0]:sp[1]] = True
    idx = np.where(moving & drink)[0]
    if len(idx) < 4:
        idx = np.where(moving)[0]
    if len(idx) < 4:
        return np.nan, np.nan, rms
    if half is not None:
        mid = len(idx) // 2
        idx = idx[:mid] if half == 0 else idx[mid:]
    if len(idx) < 2:
        return np.nan, np.nan, rms
    cos = np.sum(vm[idx] * vo[idx], axis=1) / (sm[idx] * so[idx] + 1e-9)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    good = float(np.mean(ang < GOOD_ANG))
    return good, float(np.median(ang)), rms


def inspect(npz):
    video = str(npz["video"])
    idx = F.align_index()
    if video not in idx:
        return None
    r = idx[video]; lag0 = r["lag"]
    tr = load_trial(r["c3d"])
    fused = np.asarray(npz["fused"], float)
    src = fit_source(npz)
    cent, rate = tr.centroid(), tr.rate
    dw = dwell_truth(tr)
    lags = list(range(lag0 - SEARCH, lag0 + SEARCH + 1))
    # full-rep curve (good-frac, angle, rms) for shape inspection
    curve = [(L,) + _fit_and_eval(src, fused, cent, rate, L, dw, half=None) for L in lags]
    # (1) HELD-OUT: choose lag on first half by MAX GOOD-FRAME FRAC, evaluate good-frac on 2nd half
    train = [(L, _fit_and_eval(src, fused, cent, rate, L, dw, half=0)[0]) for L in lags]
    train = [(L, g) for L, g in train if np.isfinite(g)]
    if not train:
        return None
    best_train_lag = max(train, key=lambda z: z[1])[0]   # MAX good frac
    good_chosen = _fit_and_eval(src, fused, cent, rate, lag0, dw, half=1)[0]
    good_best = _fit_and_eval(src, fused, cent, rate, best_train_lag, dw, half=1)[0]
    # (2) position rms at chosen vs full-rep-max-good lag
    full = [(L, g, a, m) for L, g, a, m in curve if np.isfinite(g)]
    best_full_lag = max(full, key=lambda z: z[1])[0]
    rms0 = dict((L, m) for L, g, a, m in curve).get(lag0, np.nan)
    rms_best = dict((L, m) for L, g, a, m in curve).get(best_full_lag, np.nan)
    return dict(video=video, lag0=lag0, curve=curve,
                best_train_lag=best_train_lag, best_full_lag=best_full_lag,
                good_chosen=good_chosen, good_best=good_best,
                rms0=rms0, rms_best=rms_best)


def main():
    files = sorted(glob.glob(str(F.FUSED_DIR / "*.npz")))
    picked = {}
    for f in files:
        stem = Path(f).stem
        for w in WANT:
            if w in stem and w not in picked:
                picked[w] = f
    print(f"confirming lag fix on {len(picked)} reps: {list(picked)}", flush=True)
    print(f"metric = GOOD-FRAME FRACTION (velocity angle < {GOOD_ANG:.0f}deg), held out on 2nd half\n",
          flush=True)
    deltas = []
    for w, f in picked.items():
        try:
            m = inspect(np.load(f, allow_pickle=True))
        except Exception as e:
            print(f"{w}: ERR {e}"); continue
        if m is None:
            print(f"{w}: no pair"); continue
        print(f"=== {m['video'][:44]}  (chosen lag {m['lag0']}) ===", flush=True)
        # (1) held-out good-frame fraction
        dg = ((m['good_best'] - m['good_chosen']) * 100
              if np.isfinite(m['good_chosen']) and np.isfinite(m['good_best']) else np.nan)
        if np.isfinite(dg):
            deltas.append(dg)
        print(f"  HELD-OUT good-frame%: chosen-lag {m['good_chosen']*100:.0f}% -> "
              f"best-lag(={m['best_train_lag']}) {m['good_best']*100:.0f}%   "
              f"delta {dg:+.0f} pts", flush=True)
        # (2) position rms (independent)
        print(f"  POSITION RMS (independent): chosen {m['rms0']:.2f}mm -> "
              f"max-good-lag(={m['best_full_lag']}) {m['rms_best']:.2f}mm", flush=True)
        # (3) curve of good-frac (full rep)
        s = "  good%-vs-lag: " + "  ".join(
            f"{L:+d}:{g*100:.0f}" if np.isfinite(g) else f"{L:+d}:--" for L, g, a, mm in m['curve'])
        print(s + "\n", flush=True)
    if deltas:
        d = np.array(deltas)
        print(f"HELD-OUT good-frame delta: mean {d.mean():+.1f} pts  median {np.median(d):+.1f} pts  "
              f"positive {int((d>0).sum())}/{len(d)}", flush=True)


if __name__ == "__main__":
    main()
