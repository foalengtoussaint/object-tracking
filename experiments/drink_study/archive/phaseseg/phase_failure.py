"""Per-phase tracking-failure analysis: attribute each mocap-vs-video error frame
to a drink-task phase, so we can show frac_fail BY phase and WHERE within a phase
the failure concentrates.

Phases are segmented from the OMC (QTM mocap) cup trajectory, NOT the video track.
This keeps the phase timeline independent of the thing we are scoring: if we cut
phases from the video track and the video is wrong during drinking, the phase
boundary would move with the error (circular). The mocap is sub-mm ground truth,
so its phase boundaries are trustworthy regardless of video tracking quality.

For every scored rep in cache/qtm_align.json we:
  1. segment the MOCAP cup centroid into 5 phases (segment_cup_only on the GT),
  2. recompute the per-frame mocap-video distance WITH its mocap-frame index
     (so error frames map onto the GT phase timeline through the sync lag),
  3. bin each failing frame (>FAIL_MM) by phase and by normalised position
     within that phase.

Writes cache/phase_failure.json (per-phase totals + within-phase failure profile).

    python phase_failure.py            # compute + cache + print summary
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import json
from pathlib import Path
import numpy as np

import qtm_align as Q
import segment_cup_only as S
from qtm_video_map import ParticipantMap, DrinkRow

CACHE = Q.CACHE
TRACKDIR = CACHE / "track3d_clean3d_refill"
PHASES = S.PHASE_NAMES
NBINS = 10  # within-phase position bins (0=phase start .. 1=phase end)


def _synced_error_with_mocap_phase(vpath: Path, c3d_stem: str):
    """Per-frame mocap-video distance plus the OMC phase id at each frame.

    Phases are segmented from the RESAMPLED MOCAP centroid (the ground-truth cup
    trajectory on the COMMON_HZ grid), so the phase array shares the error array's
    index grid. Returns (mocap_frame_idx, err_mm, phase_at_frame, corr) or None on
    the same GT/sync rejections as qtm_align._synced_rep."""
    tr = Q.load_trial(c3d_stem)
    if not tr.gt_quality()["ok"]:
        return None
    vt = Q.video_track(vpath)
    mc = tr.centroid()
    vr = Q._resample(vt, Q.VIDEO_FPS)              # both -> COMMON_HZ
    mr = Q._resample(mc, tr.rate)
    lag, corr = Q._xcorr_lag(Q._speed(vr, Q.COMMON_HZ), Q._speed(mr, Q.COMMON_HZ))
    if corr < Q.MIN_SYNC_CORR:
        return None
    # phase timeline from the GT (mocap), on the resampled COMMON_HZ grid
    seg_full = S.segment_cup_only(mr, fps=Q.COMMON_HZ)["phase"]   # (len(mr),)
    if lag >= 0:
        v = vr[lag:]; mo = mr[:len(v)]; mfi = np.arange(len(v))   # mocap idx 0..
    else:
        mo = mr[-lag:]; v = vr[:len(mo)]; mfi = np.arange(-lag, -lag + len(mo))
    L = min(len(v), len(mo))
    v, mo, mfi = v[:L], mo[:L], mfi[:L]
    ok = ~(np.isnan(v).any(1) | np.isnan(mo).any(1))
    if ok.sum() < 10:
        return None
    v, mo, mfi = v[ok], mo[ok], mfi[ok]
    R, t, _ = Q.kabsch(mo, v, robust=True)
    err = np.linalg.norm(v - (mo @ R.T + t), axis=1)
    mfi = np.clip(mfi, 0, len(seg_full) - 1)
    return mfi, err, seg_full[mfi], seg_full, corr


def analyse():
    align = json.load(open(CACHE / "qtm_align.json"))
    mp = json.load(open(CACHE / "qtm_video_map.json"))["participants"]

    # per-phase frame tallies
    tot = {p: 0 for p in PHASES}          # total frames seen in phase
    fail = {p: 0 for p in PHASES}         # frames >FAIL_MM in phase
    err_by_phase = {p: [] for p in PHASES}
    within = {p: np.zeros(NBINS) for p in PHASES}        # fail count per within-phase bin
    within_tot = {p: np.zeros(NBINS) for p in PHASES}    # total frames per within-phase bin
    # whole-movement continuous profile: forward_transport -> drinking -> back_transport
    # mapped onto [0,1] (rest excluded), so failure reads as one continuous arc.
    MBINS = 30
    mv_fail = np.zeros(MBINS)
    mv_tot = np.zeros(MBINS)
    bnd_fd, bnd_db = [], []   # per-rep fwd->drink and drink->back boundary positions in [0,1]
    n_reps = 0

    for pid, pr in align.items():
        if not (isinstance(pr, dict) and pr.get("ok")):
            continue
        for rep in pr["reps"]:
            vpath = TRACKDIR / (rep["video"] + ".json")
            if not vpath.exists():
                continue
            r = _synced_error_with_mocap_phase(vpath, rep["c3d"])
            if r is None:
                continue
            mfi, err, ph, seg, _ = r       # seg = full OMC phase timeline (mocap grid)
            n_reps += 1
            failed = err > Q.FAIL_MM
            for pidx, pname in enumerate(PHASES):
                m = ph == pidx
                if not m.any():
                    continue
                tot[pname] += int(m.sum())
                fail[pname] += int((m & failed).sum())
                err_by_phase[pname].extend(err[m].tolist())
                # within-phase normalised position of these frames (in the OMC timeline)
                fr = mfi[m]
                pos = _within_phase_pos(seg, fr, pidx)
                b = np.clip((pos * NBINS).astype(int), 0, NBINS - 1)
                ff = failed[m]
                for bi, fl in zip(b, ff):
                    within_tot[pname][bi] += 1
                    if fl:
                        within[pname][bi] += 1

            # ---- whole-movement continuous profile (rest excluded) ----
            move = np.isin(ph, [S.P_FWD, S.P_DRINK, S.P_BACK])
            if move.any():
                fr = mfi[move]
                lo, hi = fr.min(), fr.max()
                span = max(hi - lo, 1)
                pos = (fr - lo) / span                       # 0..1 across the movement
                mb = np.clip((pos * MBINS).astype(int), 0, MBINS - 1)
                fm = failed[move]
                for bi, fl in zip(mb, fm):
                    mv_tot[bi] += 1
                    if fl:
                        mv_fail[bi] += 1
                # phase boundaries as fraction of the movement span (for marking)
                dfr = mfi[ph == S.P_DRINK]
                bfr = mfi[ph == S.P_BACK]
                if dfr.size:
                    bnd_fd.append((dfr.min() - lo) / span)
                if bfr.size:
                    bnd_db.append((bfr.min() - lo) / span)

    out = {
        "n_reps": n_reps,
        "fail_mm": Q.FAIL_MM,
        "phases": PHASES,
        "frac_fail_by_phase": {p: (fail[p] / tot[p] if tot[p] else 0.0) for p in PHASES},
        "frames_by_phase": tot,
        "fail_frames_by_phase": fail,
        "median_err_by_phase": {p: (float(np.median(err_by_phase[p])) if err_by_phase[p] else 0.0)
                                for p in PHASES},
        "p90_err_by_phase": {p: (float(np.percentile(err_by_phase[p], 90)) if err_by_phase[p] else 0.0)
                             for p in PHASES},
        "within_phase_fail_rate": {p: (within[p] / np.maximum(within_tot[p], 1)).tolist()
                                   for p in PHASES},
        "nbins": NBINS,
        # whole-movement continuous failure profile (rest excluded), [0,1] across
        # forward_transport -> drinking -> back_transport
        "movement_fail_rate": (mv_fail / np.maximum(mv_tot, 1)).tolist(),
        "movement_nbins": int(MBINS),
        "movement_bound_fwd_drink": float(np.median(bnd_fd)) if bnd_fd else None,
        "movement_bound_drink_back": float(np.median(bnd_db)) if bnd_db else None,
    }
    json.dump(out, open(CACHE / "phase_failure.json", "w"), indent=2)
    print(f"analysed {n_reps} reps -> cache/phase_failure.json\n")
    print(f"{'phase':18s} {'frac_fail':>9s} {'med_mm':>7s} {'p90_mm':>7s} {'frames':>8s}")
    for p in PHASES:
        print(f"{p:18s} {out['frac_fail_by_phase'][p]:>9.3f} "
              f"{out['median_err_by_phase'][p]:>7.1f} {out['p90_err_by_phase'][p]:>7.1f} "
              f"{tot[p]:>8d}")
    return out


def _within_phase_pos(seg: np.ndarray, frames: np.ndarray, pidx: int) -> np.ndarray:
    """For each frame (already known to be in phase pidx), its position in [0,1)
    within the contiguous run of that phase containing it."""
    # precompute run bounds for this phase id
    runs = []
    i, T = 0, len(seg)
    while i < T:
        if seg[i] == pidx:
            j = i + 1
            while j < T and seg[j] == pidx:
                j += 1
            runs.append((i, j)); i = j
        else:
            i += 1
    pos = np.zeros(len(frames))
    for k, f in enumerate(frames):
        for s, e in runs:
            if s <= f < e:
                pos[k] = (f - s) / max(e - s, 1)
                break
    return pos


if __name__ == "__main__":
    analyse()
