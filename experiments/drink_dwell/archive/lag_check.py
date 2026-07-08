"""Is the TEMPORAL fit (lag) actually good, or is the residual velocity-angle a LAG error?

The pairing lag was chosen by cross-correlating cup-SPEED magnitude. That confirms same tempo
but not the exact frame offset -- and a few-frame lag error inflates the velocity-DIRECTION
angle even when positions overlap. So for each rep we search lag in [chosen-6 .. chosen+6],
recompute the position fit + drink-phase velocity angle at each, and report:

  ang@chosen   the drink angle at the pairing's lag
  ang_best     the minimum over the search
  lag_shift    how far the best lag sits from the chosen one

If ang_best << ang@chosen and lag_shift != 0 for the bad reps -> the temporal fit was OFF and
the "rotation ambiguity" story is (partly) wrong. If the angle is FLAT across nearby lags -> the
temporal fit is genuinely good and the residual is spatial/rotation, not timing.  Cache-only.
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
SEARCH = 6          # +/- frames around the chosen lag


def _synced(cup_world, mocap_centroid, rate, lag):
    vr = resample3d(cup_world, VIDEO_FPS)
    mr = resample3d(mocap_centroid, rate)
    if lag >= 0:
        v = vr[lag:]; mo = mr[:len(v)]
    else:
        mo = mr[-lag:]; v = vr[:len(mo)]
    L = min(len(v), len(mo))
    return v[:L], mo[:L]


def _drink_angle_at(raw, fused, cent, rate, lag, dw):
    fit = F.mocap_to_w0(raw, cent, rate, lag)     # position fit at THIS lag
    if fit is None:
        return np.nan
    R, t, _ = fit
    mmc, omc_lab = _synced(fused, cent, rate, lag)
    omc = omc_lab @ R.T + t
    vm = np.diff(mmc, axis=0) * HZ; vo = np.diff(omc, axis=0) * HZ
    sm = np.linalg.norm(vm, axis=1); so = np.linalg.norm(vo, axis=1)
    moving = (sm > SPEED_MM_S) & (so > SPEED_MM_S) & np.isfinite(sm) & np.isfinite(so)
    if moving.sum() < 5:
        return np.nan
    cos = np.sum(vm * vo, axis=1) / (sm * so + 1e-9)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    drink = np.zeros(len(ang), bool)
    sp = dw.span_at(len(ang)) if dw.span else None
    if sp:
        drink[sp[0]:sp[1]] = True
    dm = drink & moving
    return float(np.median(ang[dm])) if dm.any() else float(np.median(ang[moving]))


def rep_lagcheck(npz):
    video = str(npz["video"])
    idx = F.align_index()
    if video not in idx:
        return None
    r = idx[video]
    lag0 = r["lag"]
    tr = load_trial(r["c3d"])
    fused = np.asarray(npz["fused"], float)
    raw = fit_source(npz)      # KF-RTS masked to detected frames
    cent, rate = tr.centroid(), tr.rate
    dw = dwell_truth(tr)
    curve = {}
    for lag in range(lag0 - SEARCH, lag0 + SEARCH + 1):
        curve[lag] = _drink_angle_at(raw, fused, cent, rate, lag, dw)
    at = curve.get(lag0, np.nan)
    valid = {k: v for k, v in curve.items() if np.isfinite(v)}
    if not valid:
        return None
    best_lag = min(valid, key=valid.get)
    return dict(video=video, lag0=lag0, at=at, best=valid[best_lag],
                best_lag=best_lag, shift=best_lag - lag0)


def main():
    files = sorted(glob.glob(str(F.FUSED_DIR / "*.npz")))
    print(f"lag-robustness of the drink-angle over {len(files)} reps "
          f"(search +/-{SEARCH} frames)\n", flush=True)
    rows = []
    for i, f in enumerate(files):
        try:
            m = rep_lagcheck(np.load(f, allow_pickle=True))
        except Exception:
            m = None
        if m is not None:
            rows.append(m)
        if (i + 1) % 150 == 0 or i == len(files) - 1:
            print(f"  [{i+1}/{len(files)}] {len(rows)} done", flush=True)

    at = np.array([r["at"] for r in rows if np.isfinite(r["at"])])
    best = np.array([r["best"] for r in rows if np.isfinite(r["at"])])
    shift = np.array([r["shift"] for r in rows if np.isfinite(r["at"])])
    print(f"\n  reps: {len(at)}")
    print(f"  drink-angle at CHOSEN lag: median {np.median(at):.1f}deg")
    print(f"  drink-angle at BEST  lag:  median {np.median(best):.1f}deg")
    print(f"  improvement from re-lagging: median {np.median(at-best):.1f}deg")
    print(f"  reps whose best lag == chosen (shift 0): {(shift==0).sum()}/{len(shift)}"
          f"  ({100*(shift==0).mean():.0f}%)")
    print(f"  reps needing shift >=2 frames: {(np.abs(shift)>=2).sum()}")
    print(f"  reps where re-lagging drops angle >10deg: {((at-best)>10).sum()}", flush=True)
    print("\n  reps most improved by re-lagging (candidate LAG errors):")
    print(f"  {'':<34}{'lag0':>5}{'ang@0':>7}{'best':>6}{'@lag':>6}{'shift':>7}")
    for r in sorted([r for r in rows if np.isfinite(r['at'])],
                    key=lambda z: z['best'] - z['at'])[:12]:
        print(f"  {r['video'][:32]:<34}{r['lag0']:5d}{r['at']:7.0f}{r['best']:6.0f}"
              f"{r['best_lag']:6d}{r['shift']:7d}", flush=True)


if __name__ == "__main__":
    main()
