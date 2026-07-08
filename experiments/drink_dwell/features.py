"""STAGE: build the proxy21 feature (and base17) for one rep — as THREE named channel
groups, so each is inspectable on its own instead of buried in a 96-line blob:

    kinematics(npz)         -> (T,13)  fused-track speed/disp/velocity in the rep-local frame
    occlusion(track_json,T) -> (T,4)   detection health: present, median_px, ncams, occ
    head_distance(npz,trial,lag) -> (T,4)  TRACKED cup → mocap head-centroid distance

    base17  = kinematics + occlusion              (video-only ceiling)
    proxy21 = kinematics + occlusion + head_distance   (the headline: 77ms)

Data source: drink_study's caches (shared, not copied) —
    cache/lopo_fused/<rep>.npz           fused 3D track + rep-local basis/rest + truth
    cache/track3d_clean3d_refill/<rep>.json   per-frame detection health (kept cams, px)
    cache/qtm_align.json                 which mocap C3D each rep paired to + sync lag
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from mocap import load_trial, kabsch, resample as resample3d, VIDEO_FPS

_DS = Path(__file__).resolve().parents[1] / "drink_study"
FUSED_DIR = _DS / "cache" / "lopo_fused"
TRACK_DIR = _DS / "cache" / "track3d_clean3d_refill"
ALIGN_JSON = _DS / "cache" / "qtm_align.json"
HZ = 60.0

_ALIGN = None


def align_index() -> dict:
    """video-stem -> {c3d, lag, sync_corr} from qtm_align.json (which mocap trial paired)."""
    global _ALIGN
    if _ALIGN is None:
        al = json.load(open(ALIGN_JSON))
        _ALIGN = {}
        for p in al.values():
            if isinstance(p, dict) and p.get("ok"):
                for r in p["reps"]:
                    _ALIGN[r["video"]] = r
    return _ALIGN


# =========================================================================================
# STAGE 1 — kinematics (13ch): the filtered scalars the segmenter reads + RAW 3D velocity /
# displacement vectors in the rep-local basis (direction, not just magnitude).
# =========================================================================================
def _segment(xyz):
    """Minimal van-Andel speed/disp (6Hz Butterworth on position, then differentiate)."""
    from scipy.signal import butter, filtfilt
    T = len(xyz); valid = np.isfinite(xyz).all(1)
    if valid.sum() < 10:
        return np.zeros(T), np.zeros(T), (0, 0)
    idx = np.flatnonzero(valid); filled = xyz.copy()
    for ax in range(3):
        filled[:, ax] = np.interp(np.arange(T), idx, xyz[idx, ax])
    b, a = butter(4, 6.0 / (HZ / 2.0))
    filled = filtfilt(b, a, filled, axis=0)
    speed = np.r_[0.0, np.linalg.norm(np.diff(filled, axis=0), axis=1)] * HZ
    rest = np.median(filled[:max(int(0.5 * HZ), 10)], axis=0)
    disp = np.linalg.norm(filled - rest, axis=1)
    onset = [(s, e) for s, e in _runs(speed > 15) if e - s >= 3]
    grasp = (onset[0][0], onset[-1][1]) if onset else (0, 0)
    return speed, disp, grasp


def _runs(mask):
    out = []; i = 0; T = len(mask)
    while i < T:
        if mask[i]:
            j = i
            while j < T and mask[j]:
                j += 1
            out.append((i, j)); i = j
        else:
            i += 1
    return out


def kinematics(npz) -> np.ndarray:
    """(T,13) fused-track kinematics in the rep-local basis. See drink_study build_rep."""
    fused = np.asarray(npz["fused"], float); valid = np.asarray(npz["valid"], bool)
    basis = np.asarray(npz["basis"], float); rest = np.asarray(npz["rest"], float)
    T = len(fused)
    speed, disp, (gs, ge) = _segment(fused)
    peak = disp.max() if T else 1.0
    dtp = peak - disp
    accel = np.r_[0.0, np.diff(speed)]
    gap = (~valid).astype(float)
    win = np.zeros(T); win[gs:ge] = 1.0
    loc = (fused - rest) @ basis.T
    vel = np.vstack([np.zeros(3), np.diff(loc, axis=0)]) * HZ
    raw_speed = np.linalg.norm(vel, axis=1)
    P = peak + 1e-6; V = 200.0
    return np.stack([
        speed / V, disp / P, dtp / P, accel / V, gap, win,
        raw_speed / V,
        vel[:, 0] / V, vel[:, 1] / V, vel[:, 2] / V,
        loc[:, 0] / P, loc[:, 1] / P, loc[:, 2] / P,
    ], 1).astype(np.float32)


# =========================================================================================
# STAGE 2 — occlusion (4ch): detection health per frame, from the track3d JSON.
# =========================================================================================
CAMS = [f"cam_{i}" for i in range(1, 11)]


def occlusion(video: str, T: int) -> np.ndarray | None:
    """(T,4) [present, median_px/30, ncams/10, occ(ncams<2)] on the track-frame grid."""
    tj = TRACK_DIR / f"{video}.json"
    if not tj.exists():
        return None
    fr = json.loads(tj.read_text())["frames"]
    present = np.array([1.0 if f.get("consensus") else 0.0 for f in fr])
    mpx = np.array([f.get("median_px") if f.get("median_px") is not None else 30.0 for f in fr], float)
    ncams = np.array([len(f.get("kept", [])) for f in fr], float)
    occ = (ncams < 2).astype(float)
    out = np.stack([present, mpx / 30.0, ncams / 10.0, occ], 1).astype(np.float32)
    return out[:T] if len(out) >= T else np.pad(out, ((0, T - len(out)), (0, 0)))


# =========================================================================================
# STAGE 3 — head distance (4ch): TRACKED cup → mocap head-centroid, both in W0.
# The mocap head is a stand-in for the future VIDEO head landmark; the cup is the real track.
# =========================================================================================
def _mocap_to_track(sig, rate, lag, T):
    """Resample a mocap-rate (T_m,) or (T_m,k) signal to 60Hz and shift by +lag onto a
    T-frame track index (NaN where the mocap doesn't cover the frame)."""
    sig = np.asarray(sig, float)
    n_v = int(round(len(sig) / rate * HZ))
    x0 = np.linspace(0, 1, len(sig)); x1 = np.linspace(0, 1, max(n_v, 1))
    vv = (np.interp(x1, x0, sig) if sig.ndim == 1
          else np.stack([np.interp(x1, x0, sig[:, k]) for k in range(sig.shape[1])], 1))
    out = np.full((T,) + sig.shape[1:], np.nan)
    for i in range(len(vv)):
        j = i + lag
        if 0 <= j < T:
            out[j] = vv[i]
    return out


ALIGN_EXCLUDE_MM = 15.0        # drop a frame from the Kabsch fit if it aligns worse than this


def mocap_to_w0(cup_world, mocap_centroid, rate, lag, exclude=True, exclude_mm=ALIGN_EXCLUDE_MM):
    """THE single mocap(lab)->W0 alignment used everywhere in drink_dwell (features AND the
    overlay) — one function so they never disagree. Kabsch on the synced cup-centroids at the
    KNOWN-GOOD lag from qtm_align.json (never re-estimate sync here). Returns (R, t, rms_mm) or
    None if too few overlapping frames.

    With exclude=True: HARD-EXCLUDE by residual, with RE-ADMISSION. The drink phase tilts the
    cup so the mocap 4-marker centroid and the video track measure different points (~50mm apart,
    correlates with tilt r=0.54); robust Huber only LINEARISES those frames but a coherent cluster
    of them still pulls the rotation, leaving the trustworthy (upright) frames misaligned (up to
    8mm). So we iterate: keep frames with residual < exclude_mm, refit on the kept set; a frame
    excluded early REJOINS once the fit tightens (fixed physical threshold, recomputed each round).
    This tightens the good frames to ~1-3mm (P19 8.4->2.9). Falls back to the all-frame robust fit
    if <10 frames survive (rep tilted throughout). rms reported over the KEPT (trustworthy) frames."""
    vr = resample3d(cup_world, VIDEO_FPS)
    mr = resample3d(mocap_centroid, rate)
    if lag >= 0:
        v = vr[lag:]; mo = mr[:len(v)]
    else:
        mo = mr[-lag:]; v = vr[:len(mo)]
    L = min(len(v), len(mo)); v, mo = v[:L], mo[:L]
    ok = ~(np.isnan(v).any(1) | np.isnan(mo).any(1))
    if ok.sum() < 10:
        return None
    mo, v = mo[ok], v[ok]
    R, t, rms = kabsch(mo, v, robust=True)
    if exclude:
        keep = np.ones(len(mo), bool)
        for _ in range(6):
            r = np.linalg.norm(v - (mo @ R.T + t), axis=1)
            nk = r < exclude_mm
            if nk.sum() < 10 or np.array_equal(nk, keep):
                keep = nk if nk.sum() >= 10 else keep
                break
            keep = nk
            R, t, _ = kabsch(mo[keep], v[keep], robust=False)
        r = np.linalg.norm((v - (mo @ R.T + t))[keep], axis=1)
        rms = float(np.sqrt(np.mean(r ** 2))) if keep.any() else rms
    return R, t, rms


_mocap_to_w0 = mocap_to_w0     # back-compat alias


def head_distance(video: str, cup_world: np.ndarray, T: int,
                  fit_cup: np.ndarray | None = None) -> np.ndarray | None:
    """(T,4) [dist mm, approach vel, norm 0..1, present] — tracked cup → mocap head in W0.

    The ALIGNMENT (mocap->W0 Kabsch) is fit on `fit_cup` if given, else `cup_world`. Pass the
    RAW consensus track as fit_cup: the KF-smoothed 'fused' track overshoots at the fast drink
    motion and pulls the Kabsch rotation by up to 17deg / 86mm on the worst reps (P10_153316,
    P19_120423) — visible in the overlay as the mocap markers sliding ~2cm off the real cup.
    The raw track doesn't overshoot, so it gives the correct rotation. The DISTANCE is still
    measured to `cup_world` (the tracked cup the pipeline actually outputs)."""
    idx = align_index()
    if video not in idx:
        return None
    r = idx[video]
    tr = load_trial(r["c3d"])
    if not tr.has_head():
        return None
    fit = mocap_to_w0(np.asarray(fit_cup if fit_cup is not None else cup_world, float),
                      tr.centroid(), tr.rate, r["lag"])
    if fit is None:
        return None
    R, t, _ = fit
    head_w0 = _mocap_to_track(tr.head_centroid() @ R.T + t, tr.rate, r["lag"], T)   # (T,3)
    d = np.linalg.norm(head_w0 - cup_world, axis=1)                                 # (T,)
    present = np.isfinite(d).astype(np.float32)
    dd = d.copy()
    if present.sum() >= 2:
        good = present > 0.5
        dd = np.interp(np.arange(T), np.where(good)[0], d[good])
    dvel = np.r_[0.0, -np.diff(dd)]
    lo, hi = np.nanmin(dd), np.nanmax(dd)
    dn = (dd - lo) / (hi - lo + 1e-6)
    return np.stack([dd, dvel, dn, present], 1).astype(np.float32)


# =========================================================================================
# ASSEMBLE: proxy21 / base17 for one rep. Returns None if any required stage is missing.
# =========================================================================================
def build_rep(npz, include_bad_head=False):
    """-> dict with fx17, fx21, truth span (track frames), pid, video, T, head_ok. None if
    occlusion/head/pairing/dwell missing OR (unless include_bad_head) the head is too broken
    for a trustworthy truth. Truth = dwell_truth mapped onto the rep."""
    from truth import dwell_truth
    video = str(npz["video"])
    fused = np.asarray(npz["fused"], float); T = len(fused)
    raw = np.asarray(npz["cons"], float) if "cons" in npz else fused   # unfiltered cup for the FIT
    kin = kinematics(npz)                                   # (T,13)
    occ = occlusion(video, T)                               # (T,4)
    hd = head_distance(video, fused, T, fit_cup=raw)        # (T,4) fit on RAW, distance on fused
    if occ is None or hd is None:
        return None
    r = align_index()[video]
    tr = load_trial(r["c3d"])
    dw = dwell_truth(tr)
    tsp = dw.span_at(T) if dw.span else None
    if tsp is None:
        return None
    # EXCLUDE reps whose head is too broken for the truth to be trustworthy, even after
    # despiking (mocap.head_centroid already despikes; head_quality flags what it can't save).
    from truth import head_quality
    hq = head_quality(tr, dw)
    if not hq["ok"] and not include_bad_head:
        return None
    # normalise the raw head-distance mm channels to ~unit scale (as the original did)
    hd = hd.copy(); hd[:, 0] /= 600.0; hd[:, 1] /= 20.0
    base = np.concatenate([kin, occ], 1)                    # (T,17)
    return dict(video=video, pid=video.split("_")[0], T=T, tsp=tsp, head_ok=hq["ok"],
                fx17=base.astype(np.float32),
                fx21=np.concatenate([base, hd], 1).astype(np.float32))


if __name__ == "__main__":   # smoke: build one rep, print channel shapes + ranges
    import glob
    f = sorted(glob.glob(str(FUSED_DIR / "*.npz")))[0]
    d = np.load(f, allow_pickle=True)
    r = build_rep(d)
    if r is None:
        print(f"{str(d['video'])}: build_rep -> None (missing occ/head/pairing)")
    else:
        print(f"{r['video']}  T={r['T']}  truth span={r['tsp']}")
        print(f"  fx17 {r['fx17'].shape}  fx21 {r['fx21'].shape}")
        print(f"  head dist ch0 (norm) range [{r['fx21'][:,17].min():.2f},{r['fx21'][:,17].max():.2f}]")
