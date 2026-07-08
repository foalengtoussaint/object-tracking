"""AFFINE time-warp alignment: t_omc = a*t_vid + b, vs the current lag-only alignment.

The current alignment resamples each stream to 60Hz by its NOMINAL rate then applies one integer
LAG. That absorbs pure lead-in dead-time, but NOT the "one recording starts before AND ends after
the other" case (different lead-in vs lead-out) -- that needs an OFFSET + SCALE. Total durations
being ~equal does NOT rule this out (equal length, different anchoring still needs a warp).

We fit (a,b) by matching the cup-SPEED profile (rotation-invariant -> can't be confounded by the
rotational-symmetry degeneracy, unlike the velocity-angle). Objective = speed-curve correlation,
which is INDEPENDENT of the good-frame metric we score with (non-circular). Then resample OMC onto
the warped video clock, refit Kabsch on the KF-RTS-masked source, and score HELD-OUT good-frame
fraction on DRINK frames vs lag-only.

Grid: a in [0.90..1.10], b in seconds; coarse grid then local refine.  Cache-only.
"""
from __future__ import annotations
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # drink_dwell root (archived)
import features as F
from mocap import load_trial, resample as resample3d, VIDEO_FPS, kabsch
from truth import dwell_truth
from velfit import fit_source

HZ = 60.0
SPEED_MM_S = 80.0
GOOD_ANG = 20.0
A_RANGE = np.linspace(0.90, 1.10, 21)     # scale search
POS_GATE_MM = 15.0


def _speed(xyz):
    d = np.linalg.norm(np.diff(xyz, axis=0), axis=1) * HZ
    return np.r_[d, d[-1]]


def _warp_omc(omc60, a, b_frames, n_out):
    """Sample OMC (already at 60Hz) at video-frame k -> omc time index a*k + b_frames."""
    src_t = a * np.arange(n_out) + b_frames
    out = np.full((n_out, omc60.shape[1]), np.nan)
    lo = np.floor(src_t).astype(int); frac = src_t - lo
    ok = (lo >= 0) & (lo < len(omc60) - 1)
    for c in range(omc60.shape[1]):
        out[ok, c] = (1 - frac[ok]) * omc60[lo[ok], c] + frac[ok] * omc60[lo[ok] + 1, c]
    return out


def _fit_affine(vid_sp, omc_sp, lag0):
    """(a, b_frames, corr) maximizing speed-curve correlation under t_omc = a*t_vid + b."""
    n = len(vid_sp)
    best = (1.0, float(lag0), -2.0)
    b_grid = np.arange(lag0 - 20, lag0 + 21)
    for a in A_RANGE:
        for b in b_grid:
            src_t = a * np.arange(n) + b
            lo = np.floor(src_t).astype(int); frac = src_t - lo
            ok = (lo >= 0) & (lo < len(omc_sp) - 1)
            if ok.sum() < 30:
                continue
            os = (1 - frac[ok]) * omc_sp[lo[ok]] + frac[ok] * omc_sp[lo[ok] + 1]
            vs = vid_sp[ok]
            if vs.std() < 1e-6 or os.std() < 1e-6:
                continue
            c = float(np.corrcoef(vs, os)[0, 1])
            if c > best[2]:
                best = (a, float(b), c)
    return best


def _good_frac(mmc, omc_w0, drink, half=None):
    vm = np.diff(mmc, axis=0) * HZ; vo = np.diff(omc_w0, axis=0) * HZ
    sm = np.linalg.norm(vm, axis=1); so = np.linalg.norm(vo, axis=1)
    moving = (sm > SPEED_MM_S) & (so > SPEED_MM_S) & np.isfinite(sm) & np.isfinite(so)
    idx = np.where(moving & drink[:len(vm)])[0]
    if len(idx) < 4:
        idx = np.where(moving)[0]
    if len(idx) < 2:
        return np.nan
    if half is not None:
        mid = len(idx) // 2
        idx = idx[:mid] if half == 0 else idx[mid:]
    if len(idx) < 2:
        return np.nan
    cos = np.sum(vm[idx] * vo[idx], axis=1) / (sm[idx] * so[idx] + 1e-9)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return float(np.mean(ang < GOOD_ANG))


def _kabsch_on(mmc, omc_lab):
    """Kabsch mmc<-omc on spatially-close frames (robust), return R,t or None."""
    ok = ~(np.isnan(mmc).any(1) | np.isnan(omc_lab).any(1))
    if ok.sum() < 10:
        return None
    R, t, _ = kabsch(omc_lab[ok], mmc[ok], robust=True)
    for _ in range(4):
        r = np.linalg.norm(mmc[ok] - (omc_lab[ok] @ R.T + t), axis=1)
        keep = r < POS_GATE_MM
        if keep.sum() < 10:
            break
        R, t, _ = kabsch(omc_lab[ok][keep], mmc[ok][keep], robust=False)
    return R, t


def rep_warp(npz):
    video = str(npz["video"])
    idx = F.align_index()
    if video not in idx:
        return None
    r = idx[video]; lag0 = r["lag"]
    tr = load_trial(r["c3d"])
    fused = np.asarray(npz["fused"], float)
    src = fit_source(npz)
    cent, rate = tr.centroid(), tr.rate
    # both to 60Hz
    vid60 = resample3d(src, VIDEO_FPS); vidf60 = resample3d(fused, VIDEO_FPS)
    omc60 = resample3d(cent, rate)
    n = len(vid60)
    vid_sp = _speed(resample3d(fused, VIDEO_FPS)); omc_sp = _speed(omc60)
    a, b, corr = _fit_affine(vid_sp, omc_sp, lag0)
    dw = dwell_truth(tr)
    drink = np.zeros(n, bool)
    sp = dw.span_at(n) if dw.span else None
    if sp:
        drink[sp[0]:sp[1]] = True

    # LAG-ONLY baseline (a=1, b=lag0), fit + eval on fused
    def eval_alignment(aa, bb):
        omc_lab = _warp_omc(omc60, aa, bb, n)         # warp OMC-lab onto video clock
        src_lab = _warp_omc(omc60, aa, bb, n)         # same, for fit (fit source uses src track)
        fit = _kabsch_on(vid60, src_lab)              # fit on KF-RTS-masked src vs warped omc
        if fit is None:
            return None
        R, t = fit
        omc_w0 = omc_lab @ R.T + t
        return dict(
            all=_good_frac(vidf60, omc_w0, drink, None),
            train=_good_frac(vidf60, omc_w0, drink, 0),
            test=_good_frac(vidf60, omc_w0, drink, 1))

    base = eval_alignment(1.0, float(lag0))
    warp = eval_alignment(a, b)
    if base is None or warp is None:
        return None
    return dict(video=video, a=a, b=b, corr=corr, lag0=lag0,
                base_all=base["all"], warp_all=warp["all"],
                base_test=base["test"], warp_test=warp["test"])


def main():
    files = sorted(glob.glob(str(F.FUSED_DIR / "*.npz")))
    print(f"affine time-warp vs lag-only over {len(files)} reps "
          f"(good-frame%<{GOOD_ANG:.0f}deg on drink frames)\n", flush=True)
    rows = []
    for i, f in enumerate(files):
        try:
            m = rep_warp(np.load(f, allow_pickle=True))
        except Exception:
            m = None
        if m is not None:
            rows.append(m)
        if (i + 1) % 100 == 0 or i == len(files) - 1:
            print(f"  [{i+1}/{len(files)}] paired {len(rows)}", flush=True)

    a = np.array([r["a"] for r in rows])
    d_all = np.array([(r["warp_all"] - r["base_all"]) * 100 for r in rows
                      if np.isfinite(r["warp_all"]) and np.isfinite(r["base_all"])])
    d_test = np.array([(r["warp_test"] - r["base_test"]) * 100 for r in rows
                       if np.isfinite(r["warp_test"]) and np.isfinite(r["base_test"])])
    print(f"\n  reps: {len(rows)}")
    print(f"  scale a: median {np.median(a):.3f}  |a-1|>0.03 in {(np.abs(a-1)>0.03).sum()} reps"
          f"  |a-1|>0.05 in {(np.abs(a-1)>0.05).sum()}")
    print(f"  good-frame delta (full):     mean {d_all.mean():+.1f} pts  median {np.median(d_all):+.1f}"
          f"  positive {(d_all>0).sum()}/{len(d_all)}")
    print(f"  good-frame delta (HELD-OUT): mean {d_test.mean():+.1f} pts  median {np.median(d_test):+.1f}"
          f"  positive {(d_test>0).sum()}/{len(d_test)}", flush=True)
    print("\n  reps most helped by warp (held-out good-frame gain):")
    print(f"  {'':<40}{'a':>6}{'b':>6}{'base%':>7}{'warp%':>7}{'d':>6}")
    for r in sorted([r for r in rows if np.isfinite(r['warp_test']) and np.isfinite(r['base_test'])],
                    key=lambda z: -(z['warp_test'] - z['base_test']))[:12]:
        print(f"  {r['video'][:38]:<40}{r['a']:6.3f}{r['b']:6.1f}"
              f"{r['base_test']*100:7.0f}{r['warp_test']*100:7.0f}"
              f"{(r['warp_test']-r['base_test'])*100:+6.0f}", flush=True)


if __name__ == "__main__":
    main()
