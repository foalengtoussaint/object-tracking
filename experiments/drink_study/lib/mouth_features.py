"""Bring the head-mocap signals (cup->mouth distance + mouth dwell truth) onto the 60Hz
VIDEO-TRACK grid for a video rep, using the video<->c3d pairing + sync lag in qtm_align.json.

The mouth signals live in the MOCAP frame per C3D (100/120 Hz). qtm_align.json already stores,
for each validated video rep, which C3D it paired to and the temporal `lag` (video leads mocap
by `lag` video-frames). So: resample the mocap-rate signal to 60Hz, shift by +lag, and scatter
onto the T-frame track index. Used by learn_seg_mouth.py (mouth as FEATURE + mouth as TRUTH).

    from mouth_features import mouth_channels, mouth_truth_mask, align_index
    idx = align_index()                       # video-stem -> {c3d,lag,sync_corr,n}
    dist, dvel, mask = mouth_channels(video, T), mouth_truth_mask(video, T)
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mouth_dwell import dwell_truth

HZ = 60.0
_AL = None


def align_index() -> dict:
    """video-stem -> rep dict {c3d, lag, sync_corr, n} from qtm_align.json (cached)."""
    global _AL
    if _AL is None:
        from _paths import CACHE as _C
        al = json.load(open(_C / "qtm_align.json"))
        _AL = {}
        for p in al.values():
            if isinstance(p, dict) and p.get("ok"):
                for r in p["reps"]:
                    _AL[r["video"]] = r
    return _AL


def _mocap_to_track(sig_mocap, rate, lag, T):
    """Resample a mocap-rate (T_m,) or (T_m,k) signal to 60Hz and shift by +lag onto a
    T-frame track index. NaN where the mocap doesn't cover the frame."""
    sig = np.asarray(sig_mocap, float)
    n_v = int(round(len(sig) / rate * HZ))                 # mocap length in video frames
    x0 = np.linspace(0, 1, len(sig)); x1 = np.linspace(0, 1, max(n_v, 1))
    if sig.ndim == 1:
        vv = np.interp(x1, x0, sig)
    else:
        vv = np.stack([np.interp(x1, x0, sig[:, k]) for k in range(sig.shape[1])], 1)
    out = np.full((T,) + sig.shape[1:], np.nan)
    for i in range(len(vv)):
        j = i + lag
        if 0 <= j < T:
            out[j] = vv[i]
    return out


def _mocap_to_w0(cup_world, mocap_centroid, rate, lag):
    """Robust Kabsch mocap(lab)->W0 from the synced cup-centroid vs the TRACKED cup track.
    Returns (R, t, rms) with X_w0 = X_mocap @ R.T + t, or None if too few overlap frames."""
    import qtm_align as Q
    vr = Q._resample(cup_world, Q.VIDEO_FPS)
    mr = Q._resample(mocap_centroid, rate)
    if lag >= 0:
        v = vr[lag:]; mo = mr[:len(v)]
    else:
        mo = mr[-lag:]; v = vr[:len(mo)]
    L = min(len(v), len(mo)); v, mo = v[:L], mo[:L]
    ok = ~(np.isnan(v).any(1) | np.isnan(mo).any(1))
    if ok.sum() < 10:
        return None
    R, t, rms = Q.kabsch(mo[ok], v[ok], robust=True)
    return R, t, rms


def tracked_cup_to_head_channels(video, T, cup_world):
    """(T,4) head-distance channels using the TRACKED cup (what the video pipeline actually
    estimates) -> the mocap HEAD-centroid, both in W0. This is the DEPLOYMENT-realistic
    feature: only the head is a mocap stand-in (for the future video head landmark); the cup
    is the real noisy track. Channels: [0] dist mm, [1] approach velocity, [2] normalised
    0..1, [3] present-flag. None if no pairing/head/alignment.

    Distance is computed in the W0 frame: the mocap head-centroid is mapped into W0 by the
    per-rep robust Kabsch (fit on the synced cup-centroids), then dist to `cup_world`."""
    from qtm_c3d import load_trial
    from _paths import QTM_C3D_HEAD
    idx = align_index()
    if video not in idx:
        return None
    r = idx[video]
    tr = load_trial(r["c3d"], root=QTM_C3D_HEAD)
    if not tr.has_head():
        return None
    fit = _mocap_to_w0(np.asarray(cup_world, float), tr.centroid(), tr.rate, r["lag"])
    if fit is None:
        return None
    R, t, _ = fit
    head_w0_mocapgrid = tr.head_centroid() @ R.T + t          # (T_m,3) mocap head in W0
    head_w0 = _mocap_to_track(head_w0_mocapgrid, tr.rate, r["lag"], T)   # (T,3) on track grid
    cup = np.asarray(cup_world, float)
    d = np.linalg.norm(head_w0 - cup, axis=1)                 # tracked cup -> mocap head (W0)
    present = np.isfinite(d).astype(np.float32)
    dd = d.copy()
    if present.sum() >= 2:
        good = present > 0.5
        dd = np.interp(np.arange(T), np.where(good)[0], d[good])
    dvel = np.r_[0.0, -np.diff(dd)]
    lo, hi = np.nanmin(dd), np.nanmax(dd)
    dn = (dd - lo) / (hi - lo + 1e-6)
    return np.stack([dd, dvel, dn, present], 1).astype(np.float32)


def mouth_channels(video, T):
    """(T,4) head feature channels on the track grid, for video rep `video`:
      [0] cup->mouth distance (mm),
      [1] its 1st derivative (mm/frame, approach speed; +approaching),
      [2] distance normalised 0..1 within this rep (0=at mouth, 1=far),
      [3] present-flag (1 where mocap covers the frame, else 0).
    Returns None if the rep has no pairing / no head markers."""
    idx = align_index()
    if video not in idx:
        return None
    r = idx[video]
    dw = dwell_truth(r["c3d"])
    if not np.isfinite(dw.dist).any():
        return None
    dist = _mocap_to_track(dw.dist, dw.rate, r["lag"], T)   # (T,)
    present = np.isfinite(dist).astype(np.float32)
    d = dist.copy()
    # fill gaps for the derivative / normalisation, but keep `present` honest
    if present.sum() >= 2:
        good = present > 0.5
        d = np.interp(np.arange(T), np.where(good)[0], dist[good])
    dvel = np.r_[0.0, -np.diff(d)]                          # +ve = approaching mouth
    lo, hi = np.nanmin(d), np.nanmax(d)
    dn = (d - lo) / (hi - lo + 1e-6)
    return np.stack([d, dvel, dn, present], 1).astype(np.float32)


def headframe_channels(video, T):
    """(T,4) RAW head-geometry channels — the mouth-proxy-free alternative:
      [0..2] cup position in the HEAD frame (right/fwd/down of face, mm), and
      [3]    present-flag (mocap covers the frame).
    Tilt- and person-invariant; the model learns 'at the mouth' itself. None if no
    pairing / no head markers."""
    idx = align_index()
    if video not in idx:
        return None
    r = idx[video]
    from qtm_c3d import load_trial
    from _paths import QTM_C3D_HEAD
    tr = load_trial(r["c3d"], root=QTM_C3D_HEAD)
    if not tr.has_head():
        return None
    chf = tr.cup_in_head_frame()                           # (T_m,3) mocap rate
    chf_t = _mocap_to_track(chf, tr.rate, r["lag"], T)     # (T,3)
    present = np.isfinite(chf_t).all(1).astype(np.float32)
    good = present > 0.5
    if good.sum() >= 2:
        for k in range(3):
            chf_t[:, k] = np.interp(np.arange(T), np.where(good)[0], chf_t[good, k])
    return np.concatenate([chf_t, present[:, None]], 1).astype(np.float32)


def _head_pts_on_track(video, T):
    """(T,5,3) lab-frame head marker points on the 60Hz track grid, or None."""
    idx = align_index()
    if video not in idx:
        return None
    r = idx[video]
    from qtm_c3d import load_trial
    from _paths import QTM_C3D_HEAD
    tr = load_trial(r["c3d"], root=QTM_C3D_HEAD)
    if not tr.has_head():
        return None
    pts = tr.head_marker_pts()                              # (T_m,5,3) mocap rate
    flat = _mocap_to_track(pts.reshape(len(pts), -1), tr.rate, r["lag"], T)  # (T,15)
    return flat.reshape(T, 5, 3)


def _fill_cols(a):
    """Interp-fill NaNs independently per column of (T,k); leave a col all-NaN as 0."""
    a = a.copy()
    for k in range(a.shape[1]):
        g = np.isfinite(a[:, k])
        if g.sum() >= 2:
            a[:, k] = np.interp(np.arange(len(a)), np.where(g)[0], a[g, k])
        elif g.sum() == 1:
            a[:, k] = a[g, k][0]
        else:
            a[:, k] = 0.0
    return a


def points_channels(video, T, rest, basis):
    """(T,16) 5 head markers in the SHARED rest-anchored, basis-projected space (same
    transform the fused cup track uses: (p - rest) @ basis.T) + present-flag. No mouth
    proxy, no head-frame rotation — head points sit in the SAME coordinate space as the
    cup (the model already gets the local cup track), and it learns the relationship.
    Filled PER-MARKER (a marker present on most frames is kept even if another is missing).
    None if no pairing/head."""
    hp = _head_pts_on_track(video, T)                      # (T,5,3) lab frame
    if hp is None:
        return None
    present = np.isfinite(hp).any((1, 2)).astype(np.float32)   # any head data this frame
    loc = ((hp - rest) @ basis.T).reshape(T, -1)           # (T,15) shared local space
    loc = _fill_cols(loc)
    return np.concatenate([loc / 600.0, present[:, None]], 1).astype(np.float32)  # (T,16)


def dist_channels(video, T, cup_world):
    """(T,6) cup-to-each-head-marker distances (5) + present-flag. Rotation-invariant by
    construction (a distance ignores head tilt — the property that broke the mouth proxy).
    `cup_world` is the (T,3) cup track in the SAME frame the head points are loaded in
    (lab/world). Filled per-marker. None if no pairing/head."""
    hp = _head_pts_on_track(video, T)                      # (T,5,3) lab frame
    if hp is None:
        return None
    present = np.isfinite(hp).any((1, 2)).astype(np.float32)
    d = np.linalg.norm(hp - cup_world[:, None, :], axis=2)  # (T,5) mm, NaN where marker/cup missing
    d = _fill_cols(d)
    return np.concatenate([d / 600.0, present[:, None]], 1).astype(np.float32)  # (T,6)


def mouth_truth_mask(video, T):
    """(T,) 0/1 mouth-dwell truth on the track grid. None if no pairing/dwell."""
    idx = align_index()
    if video not in idx:
        return None
    r = idx[video]
    dw = dwell_truth(r["c3d"])
    if dw.span is None:
        return None
    m = np.zeros(len(dw.dist), np.float32)
    m[dw.span[0]:dw.span[1]] = 1.0
    mv = _mocap_to_track(m, dw.rate, r["lag"], T)
    mv = np.nan_to_num(mv, nan=0.0)
    return (mv >= 0.5).astype(np.float32)


def mouth_span(video, T):
    """(s,e) mouth dwell in track frames, or None."""
    m = mouth_truth_mask(video, T)
    if m is None:
        return None
    on = np.where(m > 0.5)[0]
    return (int(on[0]), int(on[-1] + 1)) if len(on) else None
