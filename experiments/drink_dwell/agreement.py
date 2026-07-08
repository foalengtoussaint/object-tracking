"""THE shared MMC<->OMC agreement math — one definition, imported by every probe/plot/render,
so they can never drift (the copy-pasted `_synced`/angle/good-frame in ~14 scripts was the bug
this module fixes; see feedback_shared_code_metric_and_render memory).

    sync_tracks(video_track, mocap_lab, rate, lag) -> (v, mo)  paired 60Hz, lag-shifted
    velocity_angles(v_w0, omc_w0)                  -> (angle_deg, moving_mask)  per frame
    good_frac(v_w0, omc_w0, window=None)           -> fraction of (moving[&window]) frames < GOOD
    mmc_quality(video)                             -> (ncams, med_px) per track frame
    drink_mask(dw, n)                              -> bool mask of the dwell span on an n-grid

All velocity-DIRECTION based (translation/scale-invariant). Frame grid = 60Hz video/track grid.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

import features as F
from mocap import resample as resample3d, VIDEO_FPS

HZ = 60.0
SPEED_MM_S = 80.0        # a frame's velocity direction is only trustworthy above this speed
GOOD_ANG = 20.0          # velocity angle < this = the two cups "agree" this frame
MIN_CAMS = 4             # MMC-quality gate: at least this many cameras
MAX_PX = 8.0             # MMC-quality gate: reprojection spread at most this
TRACK_DIR = F._DS / "cache" / "track3d_clean3d_refill"


def sync_tracks(video_track, mocap_lab, rate, lag):
    """Resample both to 60Hz, shift by lag, truncate to overlap. Returns (v, mo), NOT yet in W0."""
    vr = resample3d(video_track, VIDEO_FPS); mr = resample3d(mocap_lab, rate)
    if lag >= 0:
        v = vr[lag:]; mo = mr[:len(v)]
    else:
        mo = mr[-lag:]; v = vr[:len(mo)]
    L = min(len(v), len(mo))
    return v[:L], mo[:L]


def velocity_angles(v_w0, omc_w0):
    """Per-frame angle (deg) between video & mocap cup velocity + a moving mask (both > SPEED).
    Both inputs already in W0 (mocap transformed by the fit). Length = len-1 of the inputs."""
    a = np.diff(v_w0, axis=0) * HZ; b = np.diff(omc_w0, axis=0) * HZ
    sa = np.linalg.norm(a, axis=1); sb = np.linalg.norm(b, axis=1)
    moving = (sa > SPEED_MM_S) & (sb > SPEED_MM_S) & np.isfinite(sa) & np.isfinite(sb)
    ang = np.degrees(np.arccos(np.clip(np.sum(a * b, 1) / (sa * sb + 1e-9), -1, 1)))
    return ang, moving


def good_frac(v_w0, omc_w0, window=None):
    """Fraction of moving frames (optionally & window) with velocity angle < GOOD_ANG."""
    ang, moving = velocity_angles(v_w0, omc_w0)
    ev = moving & window[:len(moving)] if window is not None else moving
    if ev.sum() < 2:
        ev = moving
    if ev.sum() < 2:
        return np.nan
    return float(np.mean(ang[ev] < GOOD_ANG))


def mmc_quality(video):
    """(ncams, med_px) per track frame from the track3d JSON — INDEPENDENT of mocap. None if absent."""
    p = TRACK_DIR / f"{video}.json"
    if not p.exists():
        return None
    fr = json.loads(p.read_text())["frames"]
    ncams = np.array([len(f.get("kept", [])) for f in fr], float)
    mpx = np.array([f.get("median_px") if f.get("median_px") is not None else np.nan for f in fr], float)
    return ncams, mpx


def align_quality_to_grid(arr, lag, n):
    """Slice a per-track-frame array (e.g. ncams) onto the n-frame velocity grid at the given lag."""
    return arr[lag:lag + n] if lag >= 0 else arr[:n]


def drink_mask(dw, n):
    """Bool mask (len n-1, velocity grid) of the dwell span, or all-False."""
    m = np.zeros(max(n - 1, 1), bool)
    sp = dw.span_at(n) if dw.span else None
    if sp:
        m[sp[0]:min(sp[1], n - 1)] = True
    return m
