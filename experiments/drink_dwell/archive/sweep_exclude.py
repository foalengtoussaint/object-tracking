"""Does TIGHTENING the Kabsch exclude threshold (15mm -> 5mm) sharpen the fit?

The alignment keeps frames with residual < exclude_mm and refits. Looser = more frames
(incl. tilt-contaminated ones that pull the rotation); tighter = only near-coincident
upright frames, but risks too-few-frames / degeneracy. We sweep exclude_mm and, per rep,
measure the DRINK-PHASE velocity angle (the rotation-sensitive metric) and how many frames
survive the fit. If tightening helps, drink-angle drops without the fit collapsing.

Cache-only.  Prints a sweep table; no plot.
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

THRESHOLDS = [5.0, 8.0, 12.0, 15.0, 25.0]
SPEED_MM_S = 80.0
HZ = 60.0


def _synced_pair(cup_world, mocap_centroid, rate, lag, R, t):
    vr = resample3d(cup_world, VIDEO_FPS)
    mr = resample3d(mocap_centroid, rate) @ R.T + t
    if lag >= 0:
        v = vr[lag:]; mo = mr[:len(v)]
    else:
        mo = mr[-lag:]; v = vr[:len(mo)]
    L = min(len(v), len(mo))
    return v[:L], mo[:L]


def _vel(xyz):
    d = np.diff(xyz, axis=0) * HZ
    d = np.vstack([d, d[-1:]])
    return d, np.linalg.norm(d, axis=1)


def _kept_count(cup_world, mocap_centroid, rate, lag, R, t, thr):
    """how many synced frames land within thr of the fit (fit support)."""
    mmc, omc = _synced_pair(cup_world, mocap_centroid, rate, lag, R, t)
    d = np.linalg.norm(mmc - omc, axis=1)
    ok = np.isfinite(d)
    return int((d[ok] < thr).sum()), int(ok.sum())


def rep_metrics(npz, thr):
    video = str(npz["video"])
    idx = F.align_index()
    if video not in idx:
        return None
    r = idx[video]
    tr = load_trial(r["c3d"])
    fused = np.asarray(npz["fused"], float)
    raw = np.asarray(npz["cons"], float) if "cons" in npz else fused
    fit = F.mocap_to_w0(raw, tr.centroid(), tr.rate, r["lag"], exclude=True, exclude_mm=thr)
    if fit is None:
        return None
    R, t, rms = fit
    kept, total = _kept_count(raw, tr.centroid(), tr.rate, r["lag"], R, t, thr)
    mmc, omc = _synced_pair(fused, tr.centroid(), tr.rate, r["lag"], R, t)
    vm, sm = _vel(mmc); vo, so = _vel(omc)
    moving = (sm > SPEED_MM_S) & (so > SPEED_MM_S) & np.isfinite(sm) & np.isfinite(so)
    if moving.sum() < 5:
        return None
    cos = np.sum(vm * vo, axis=1) / (sm * so + 1e-9)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    dm = np.zeros(len(ang), bool)
    dw = dwell_truth(tr)
    sp = dw.span_at(len(ang)) if dw.span else None
    if sp:
        dm[sp[0]:sp[1]] = True
    md = moving & dm
    return dict(all_ang=float(np.median(ang[moving])),
                drink_ang=float(np.median(ang[md])) if md.any() else np.nan,
                rms=rms, kept=kept, total=total)


def main():
    files = sorted(glob.glob(str(F.FUSED_DIR / "*.npz")))
    npzs = []
    print(f"loading {len(files)} reps...", flush=True)
    for f in files:
        try:
            npzs.append(np.load(f, allow_pickle=True))
        except Exception:
            pass
    print(f"loaded {len(npzs)}. sweeping exclude_mm={THRESHOLDS}\n", flush=True)

    print(f"{'thr':>5}{'reps':>6}{'all-ang':>9}{'drink-ang':>11}{'fit-rms':>9}"
          f"{'kept-frac':>11}{'few(<15)':>10}")
    for thr in THRESHOLDS:
        aa, da, rms, kf, few = [], [], [], [], 0
        for npz in npzs:
            m = rep_metrics(npz, thr)
            if m is None:
                continue
            aa.append(m["all_ang"])
            if np.isfinite(m["drink_ang"]):
                da.append(m["drink_ang"])
            rms.append(m["rms"])
            kf.append(m["kept"] / max(m["total"], 1))
            if m["kept"] < 15:
                few += 1
        print(f"{thr:5.0f}{len(aa):6d}{np.median(aa):9.1f}{np.median(da):11.1f}"
              f"{np.median(rms):9.2f}{np.median(kf)*100:10.0f}%{few:10d}", flush=True)
    print("\nall-ang / drink-ang = median per-rep MMC-OMC velocity angle (deg, moving frames)")
    print("kept-frac = median fraction of frames inside the fit; few = reps with <15 fit frames")


if __name__ == "__main__":
    main()
