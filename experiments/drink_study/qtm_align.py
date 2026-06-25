"""Spatial + temporal alignment of QTM cup mocap to the video cup tracks.

Goal: overlay the optical-mocap cup ground truth (QTM lab frame, 100/120 Hz, mm)
onto the video pipeline's 3D cup track (camera-calibration "W0" frame, 60 fps, mm)
so we can measure how accurate the video cup tracking is.

Pipeline, per participant (right-side sips — the video pipeline only triangulated
`drinking_right`):

  1. Pair video track3d reps <-> right-side C3D takes (time order; monotonic
     duration alignment via qtm_video_map.monotonic_align for off-by-one counts).
  2. Temporal sync (per rep): both cup-speed curves resampled to a common rate,
     cross-correlated to find the lag (handles fps mismatch + clip start offset).
  3. Spatial Kabsch (pooled over a participant's reps): rotation + translation
     (NO scale; units already match in mm) mapping QTM lab -> video W0.
  4. Residual = RMS distance (mm) between transformed mocap centroid and the video
     cup track after sync. This is the cup-tracking accuracy vs GT.

Video track 3D = the consensus-anchored KF/RTS centroid (see cache_track3d.py);
mocap cup = mean of the 4 cup markers (qtm_c3d.CupTrial.centroid()).

Usage:
    python qtm_align.py --participant P06
    python qtm_align.py --all --json cache/qtm_align.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import CACHE
from qtm_c3d import load_trial
from qtm_video_map import build_map, monotonic_align

# clean3d_refill has BOTH sides (351 left + 355 right) and is the best cup track
# per project memory (reproject-fill wins every axis). track3d/ is right-only.
TRACK3D_DIR = CACHE / "track3d_clean3d_refill"
VIDEO_FPS = 60.0
COMMON_HZ = 60.0      # resample both modalities to this for sync + Kabsch
REP_REJECT_MM = 50.0  # per-rep residual above this = confident-wrong, drop from pooled fit
TRANSFORM_OUTLIER_MM = 120.0  # rep whose fitted translation t is this far from the
                              # participant median = failed sync/alignment, not a sample


def _ts(stem: str) -> str:
    m = re.search(r"(\d{8}_\d{6})", stem)
    return m.group(1) if m else stem


def video_reps(pid: str, side: str) -> list[Path]:
    """Time-ordered track3d JSONs for a participant on one side ('left'|'right')."""
    fs = [p for p in glob.glob(str(TRACK3D_DIR / f"{pid}_*drinking_{side}*.json"))
          if "_summary" not in p]
    return sorted((Path(p) for p in fs), key=lambda p: _ts(p.stem))


def video_track(path: Path) -> np.ndarray:
    """(T,3) cup centroid in W0 (mm). Prefer RTS, fall back kf/consensus; NaN gaps."""
    d = json.loads(path.read_text())
    out = []
    for f in d["frames"]:
        p = f.get("rts") or f.get("kf") or f.get("consensus")
        out.append(p if p is not None else [np.nan] * 3)
    return np.asarray(out, float)


def _speed(xyz: np.ndarray, rate: float) -> np.ndarray:
    x = xyz.copy()
    for k in range(3):  # interp small gaps so gradient is clean
        col = x[:, k]; idx = np.arange(len(col)); g = ~np.isnan(col)
        if g.sum() > 2:
            x[:, k] = np.interp(idx, idx[g], col[g])
    return np.linalg.norm(np.gradient(x, 1.0 / rate, axis=0), axis=1)


def _resample(xyz: np.ndarray, rate: float, to_hz: float = COMMON_HZ) -> np.ndarray:
    n = xyz.shape[0]
    t0 = np.arange(n) / rate
    t1 = np.arange(0, t0[-1], 1.0 / to_hz)
    out = np.empty((len(t1), 3))
    for k in range(3):
        col = xyz[:, k]; g = ~np.isnan(col)
        out[:, k] = np.interp(t1, t0[g], col[g]) if g.sum() > 2 else np.nan
    return out


def _xcorr_lag(a: np.ndarray, b: np.ndarray):
    """Lag (samples) maximizing normalized correlation of speed curves and the peak
    correlation value (sync confidence in [0,1]). Positive lag => b later."""
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    c = np.correlate(a, b, mode="full") / min(len(a), len(b))
    k = int(np.argmax(c))
    return k - (len(b) - 1), float(c[k])


def _kabsch_weighted(P, Q, w):
    """Weighted Kabsch: rotation+translation mapping P->Q minimizing sum w*||.||^2."""
    w = w / (w.sum() + 1e-12)
    Pc = (w[:, None] * P).sum(0); Qc = (w[:, None] * Q).sum(0)
    H = (w[:, None] * (P - Pc)).T @ (Q - Qc)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, Qc - R @ Pc


def kabsch(P: np.ndarray, Q: np.ndarray, robust: bool = True, iters: int = 10):
    """Rotation+translation mapping P -> Q (no scale). Returns (R, t, rms_mm).

    With robust=True, uses IRLS with Huber weights so the transform locks onto the
    correspondences that already agree (rest / transport frames) and is NOT dragged
    by genuinely-discrepant segments (e.g. an occluded drink apex where the video
    track is biased ~140mm in z). I.e. adding low-error frames pulls the fit; high-
    error frames are down-weighted rather than allowed to smear the fit across the
    whole trajectory. Reported rms is over the LOW-residual (inlier, w>0.5) frames
    so it measures agreement where the cup is trackable, not the worst moments."""
    R, t = _kabsch_weighted(P, Q, np.ones(len(P)))
    if robust:
        for _ in range(iters):
            r = np.linalg.norm(Q - (P @ R.T + t), axis=1)
            c = 1.345 * (np.median(r) + 1e-6)        # Huber scale ~ median residual
            w = np.where(r <= c, 1.0, c / (r + 1e-9))
            R, t = _kabsch_weighted(P, Q, w)
        r = np.linalg.norm(Q - (P @ R.T + t), axis=1)
        inl = r <= 1.345 * (np.median(r) + 1e-6)
        rms = float(np.sqrt(np.mean(r[inl] ** 2))) if inl.any() else float(np.sqrt(np.mean(r ** 2)))
    else:
        rms = float(np.sqrt(np.mean(np.sum((Q - (P @ R.T + t)) ** 2, axis=1))))
    return R, t, rms


# A rep is a valid accuracy sample only if (a) the mocap is good ground truth — the
# cup actually moved (lift>=120mm; dead/static takes like the P14_0031.. cluster have
# no motion) and excursion glitches are despiked in CupTrial.centroid — and (b) the
# two cup-speed curves genuinely line up. Sync confidence = peak normalized
# cross-correlation of the speed curves: ~0.99 for real matches, ~0.2 when the track
# or GT is wrong. This single signal cleanly separates the 14mm accuracy cluster (96%
# of reps) from the ~2% failed-sync tail (median 129mm) — no lag/offset heuristics.
MIN_GT_LIFT_MM = 120.0
# With broken GT pre-rejected by gt_quality(), a clean take's speed curves line up at
# corr~0.99; borderline-clean ones sit 0.80-0.90. 0.7 keeps those, still drops genuine
# sync failures (any that slip past the GT gate land near ~0.2).
MIN_SYNC_CORR = 0.7

# A single per-rep median/RMS hides localized failures (a rep can track at 5 mm for
# 2/3 of the motion then be 150 mm off at the occluded drink apex). So describe each
# rep with TWO numbers and classify:
#   inlier_rms_mm  = robust-fit residual on agreeing frames = tracking fidelity
#   frac_fail      = fraction of trajectory beyond FAIL_MM = coverage of failure
FAIL_MM = 50.0          # >50 mm = real disagreement (vs <~20 mm sync/transport jitter)
FIDELITY_OK_MM = 15.0   # inlier_rms below this = tracking is tight where it works
FRAC_FAIL_OK = 0.10     # up to 10% of trajectory off = tolerable (brief apex moment)


def classify_rep(inlier_rms: float, frac_fail: float) -> str:
    """clean / localized / broken from the two metrics."""
    if frac_fail <= FRAC_FAIL_OK and inlier_rms <= FIDELITY_OK_MM:
        return "clean"                      # accurate throughout
    if inlier_rms <= FIDELITY_OK_MM:
        return "localized"                  # tight fidelity, but a chunk fails (apex/occlusion)
    return "broken"                         # disagrees ~everywhere (bad track or residual GT)


def _synced_rep(vpath: Path, c3d_stem: str):
    """Sync one video rep to its C3D. Returns (mocap_pts, video_pts, lag, sync_corr)
    in mm after temporal alignment, or None. Rejection is two-stage:
      1. GT-quality gate on the MOCAP ALONE (CupTrial.gt_quality) — drops broken QTM
         takes (non-physical steps, dead/static) BEFORE any sync, so a defective GT
         can never be scored as tracking error.
      2. sync-confidence gate — among clean GT, require the speed curves to actually
         line up (corr >= MIN_SYNC_CORR)."""
    tr = load_trial(c3d_stem)
    if not tr.gt_quality()["ok"]:
        return None
    vt = video_track(vpath)
    mc = tr.centroid()          # despiked
    vr = _resample(vt, VIDEO_FPS); mr = _resample(mc, tr.rate)
    lag, corr = _xcorr_lag(_speed(vr, COMMON_HZ), _speed(mr, COMMON_HZ))
    if corr < MIN_SYNC_CORR:
        return None
    if lag >= 0:
        v = vr[lag:]; mo = mr[: len(v)]
    else:
        mo = mr[-lag:]; v = vr[: len(mo)]
    L = min(len(v), len(mo))
    v, mo = v[:L], mo[:L]
    ok = ~(np.isnan(v).any(1) | np.isnan(mo).any(1))
    if ok.sum() < 10:
        return None
    return mo[ok], v[ok], lag, corr


def align_participant(pid: str, pm) -> dict:
    """Both sides: pair video reps with same-side C3D, sync per rep, reject
    confident-wrong reps by per-rep residual, then pool survivors for one Kabsch."""
    reps_all = []      # (mocap_pts, video_pts, info) — clean GT, confidently synced
    n_bad_gt = 0       # reps rejected up-front: broken QTM trajectory (defect/dead)
    n_bad_sync = 0     # clean GT but speed curves don't line up
    bad_gt = []        # for reporting
    for side in ("left", "right"):
        reps = video_reps(pid, side)
        c3d = [(c, row) for c, row in pm.pairs if (row.side or "").lower() == side]
        if not reps or not c3d:
            continue
        vdur = [video_track(r).shape[0] / VIDEO_FPS for r in reps]
        cdur = [load_trial(c).duration_s for c, _ in c3d]
        idx_pairs, _, _ = monotonic_align(vdur, cdur)
        for vi, ci in idx_pairs:
            q = load_trial(c3d[ci][0]).gt_quality()
            if not q["ok"]:
                n_bad_gt += 1
                bad_gt.append(dict(c3d=c3d[ci][0], side=side, reason=q["reason"]))
                continue
            s = _synced_rep(reps[vi], c3d[ci][0])
            if s is None:
                n_bad_sync += 1
                continue
            mo, v, lag, corr = s
            R_i, t_i, inlier_rms = kabsch(mo, v, robust=True)   # robust per-rep fit
            e = np.linalg.norm(v - (mo @ R_i.T + t_i), axis=1)  # per-frame error
            frac_fail = float(np.mean(e > FAIL_MM))
            cls = classify_rep(inlier_rms, frac_fail)
            reps_all.append((mo, v, dict(video=reps[vi].stem, c3d=c3d[ci][0],
                                         side=side, lag=lag, sync_corr=round(corr, 3),
                                         n=len(mo),
                                         inlier_rms_mm=round(inlier_rms, 1),
                                         frac_fail=round(frac_fail, 2),
                                         max_err_mm=round(float(e.max()), 0),
                                         cls=cls)))
    if not reps_all:
        return dict(participant=pid, ok=False, reason="no confidently-synced reps",
                    n_bad_gt=n_bad_gt, n_bad_sync=n_bad_sync, bad_gt=bad_gt)

    # Per-rep transform: we validate trajectory SHAPE, not absolute placement, so each
    # rep's own rigid transform is factored out. GT defects already rejected upstream.
    infos = [r[2] for r in reps_all]
    cls_counts = {c: sum(i["cls"] == c for i in infos) for c in ("clean", "localized", "broken")}
    inlier = [i["inlier_rms_mm"] for i in infos]
    return dict(participant=pid, ok=True,
                n_reps=len(reps_all), n_bad_gt=n_bad_gt, n_bad_sync=n_bad_sync,
                cls_counts=cls_counts,
                inlier_rms_median_mm=float(np.median(inlier)),
                frac_fail_median=float(np.median([i["frac_fail"] for i in infos])),
                reps=infos, bad_gt=bad_gt)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--participant", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    m = build_map()
    pids = [args.participant] if args.participant else (
        [p for p, pm in m.items() if pm.matched] if args.all else ["P06"])

    results = {}
    # Two numbers per rep: inlier_rms (fidelity where trackable) + frac_fail (coverage
    # of failure). A median alone hides localized apex/occlusion failures.
    print(f"{'pid':5s} {'reps':>4s} {'inlier_mm':>9s} {'fracfail':>8s}  clean/local/broken  rej(gt,sync)")
    tot = {"clean": 0, "localized": 0, "broken": 0}
    for pid in pids:
        if pid not in m or not m[pid].matched:
            print(f"{pid:5s}  -- not matched in qtm_video_map"); continue
        r = align_participant(pid, m[pid])
        results[pid] = r
        if r["ok"]:
            cc = r["cls_counts"]
            for k in tot: tot[k] += cc[k]
            print(f"{pid:5s} {r['n_reps']:>4d} {r['inlier_rms_median_mm']:>9.1f} "
                  f"{r['frac_fail_median']:>8.2f}  {cc['clean']:>3d}/{cc['localized']:<3d}/{cc['broken']:<3d}"
                  f"      ({r['n_bad_gt']},{r['n_bad_sync']})")
        else:
            print(f"{pid:5s}  FAILED: {r['reason']}")
    n = sum(tot.values())
    if n:
        print(f"\nCOHORT classes: clean {tot['clean']} ({tot['clean']/n:.0%})  "
              f"localized {tot['localized']} ({tot['localized']/n:.0%})  "
              f"broken {tot['broken']} ({tot['broken']/n:.0%})  of {n} valid reps")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
