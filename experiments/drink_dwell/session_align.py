"""SESSION-constant rotation + per-trial translation alignment, vs the per-trial fit.

Validated by session_fit_check.py: the mocap->W0 ROTATION is a session constant (1-3deg spread on
clean sessions); the degenerate reps (P16/P19) are trials whose per-trial Kabsch picked the WRONG
rotation branch (rotational-symmetry of the round cup). TRANSLATION is mostly-but-not-perfectly
constant (~30-50mm real per-trial residual even when R is fixed), so we keep t per-trial.

METHOD:
  1. per session, fit each trial's ORIGINAL position Kabsch (R_i, t_i).
  2. ROBUST session rotation R_sess = chordal mean of the R_i that agree (drop >20deg outliers,
     re-average) -> the branch the majority of trials vote for.
  3. per trial, re-fit ONLY the translation given R_sess:  t_i = median(video_cup - R_sess @ mocap_cup)
     over the spatially-close frames (so the degenerate reps get the session R + their own t).
  4. score drink good-frame% (velocity angle <20deg) per trial: per-trial fit vs session-R fit.

Cache-only.  Prints per-session + overall + the reps most rescued.  -> slides/session_align.png
"""
from __future__ import annotations
import glob
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F
from mocap import load_trial, resample as resample3d, VIDEO_FPS
from truth import dwell_truth

HZ = 60.0
SPEED_MM_S = 80.0
GOOD_ANG = 20.0
OUT = Path(__file__).resolve().parent / "slides" / "session_align.png"


def _sk(v):
    m = re.search(r"(20\d{6})", v)
    return f"{v.split('_')[0]}@{m.group(1) if m else '?'}"


def _rotang(A, B):
    return float(np.degrees(np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1))))


def _chordal_mean(Rs):
    Rbar = np.mean(Rs, axis=0)
    U, _, Vt = np.linalg.svd(Rbar)
    return U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt


_ROT_CACHE = {}

def session_rotation(video, exclude_deg=20.0):
    """Robust session rotation R_sess for the SESSION containing `video`: chordal mean of the
    per-trial position-Kabsch rotations that agree (drop >exclude_deg outliers, re-average).
    Returns (R_sess, n_trials, n_inliers, self_dev_deg) or None. Importable by the graph/overlay.
    Cached per session (the scan over the session's trials is expensive)."""
    idx = F.align_index()
    if video not in idx:
        return None
    key = _sk(video)
    if key in _ROT_CACHE:
        return _ROT_CACHE[key]
    Rs = []
    for f in glob.glob(str(F.FUSED_DIR / "*.npz")):
        vv = Path(f).stem
        if _sk(vv) != key:
            continue
        try:
            npz = np.load(f, allow_pickle=True)
        except Exception:
            continue
        v2 = str(npz["video"])
        if v2 not in idx:
            continue
        r = idx[v2]; tr = load_trial(r["c3d"])
        raw = np.asarray(npz["cons"], float) if "cons" in npz else np.asarray(npz["fused"], float)
        fit = F.mocap_to_w0(raw, tr.centroid(), tr.rate, r["lag"])
        if fit is not None:
            Rs.append(fit[0])
    if len(Rs) < 3:
        _ROT_CACHE[key] = None
        return None
    R0 = _chordal_mean(Rs)
    inl = [R for R in Rs if _rotang(R0, R) < exclude_deg]
    R_sess = _chordal_mean(inl) if len(inl) >= 3 else R0
    dev = float(np.median([_rotang(R_sess, R) for R in Rs]))
    out = (R_sess, len(Rs), len(inl), dev)
    _ROT_CACHE[key] = out
    return out


_SIM_CACHE = {}

def session_similarity(video, exclude_mm=25.0):
    """ONE global (R, s, t) for the whole session, fit JOINTLY by Umeyama over ALL the session's
    (video-cup, mocap-cup) frame-pairs pooled together. Unlike session_rotation (which averages
    per-trial rotations then fits scale separately), here rotation AND scale are solved together on
    one big cloud — the scale can pull the rotation to a different optimum. Robust: fit, drop
    frame-pairs with residual > exclude_mm, refit (few iters). Returns (R, s, t, n_frames, n_kept)
    or None. Cached per session."""
    key = _sk(video)
    if key in _SIM_CACHE:
        return _SIM_CACHE[key]
    idx = F.align_index()
    P_all, Q_all = [], []
    for f in glob.glob(str(F.FUSED_DIR / "*.npz")):
        if _sk(Path(f).stem) != key:
            continue
        try:
            npz = np.load(f, allow_pickle=True)
        except Exception:
            continue
        v2 = str(npz["video"])
        if v2 not in idx:
            continue
        r = idx[v2]; tr = load_trial(r["c3d"])
        raw = np.asarray(npz["cons"], float) if "cons" in npz else np.asarray(npz["fused"], float)
        v, mo = _synced(raw, tr.centroid(), tr.rate, r["lag"])
        ok = ~(np.isnan(v).any(1) | np.isnan(mo).any(1))
        if ok.sum() >= 5:
            Q_all.append(v[ok]); P_all.append(mo[ok])   # P=mocap -> Q=video
    if not P_all:
        _SIM_CACHE[key] = None; return None
    P = np.vstack(P_all); Q = np.vstack(Q_all)
    R, t, s, _ = umeyama_sim(P, Q)
    keep = np.ones(len(P), bool)
    for _ in range(4):
        res = np.linalg.norm(Q - (s * (P @ R.T) + t), axis=1)
        nk = res < exclude_mm
        if nk.sum() < 20 or np.array_equal(nk, keep):
            keep = nk if nk.sum() >= 20 else keep
            break
        keep = nk
        R, t, s, _ = umeyama_sim(P[keep], Q[keep])
    out = (R, s, t, len(P), int(keep.sum()))
    _SIM_CACHE[key] = out
    return out


def _synced(vid_track, omc_lab, rate, lag):
    """video cup + omc-lab centroid at 60Hz, synced at lag (both NOT yet in W0)."""
    vr = resample3d(vid_track, VIDEO_FPS); mr = resample3d(omc_lab, rate)
    if lag >= 0:
        v = vr[lag:]; mo = mr[:len(v)]
    else:
        mo = mr[-lag:]; v = vr[:len(mo)]
    L = min(len(v), len(mo))
    return v[:L], mo[:L]


def translation_for_R(vid_track, omc_lab, rate, lag, R, exclude_mm=15.0):
    """t mapping mocap-lab -> W0 given a FIXED rotation R: robust median of (video - R*mocap) over
    synced, spatially-close frames (iterate to <exclude_mm). This is THE translation both the
    metric scoring and the overlay must use for a session/velocity R, so the number and the render
    never disagree (the overlay bug was a length-mismatched fallback that skipped the sync)."""
    v, mo = _synced(vid_track, omc_lab, rate, lag)
    ok = ~(np.isnan(v).any(1) | np.isnan(mo).any(1))
    if ok.sum() < 3:
        return np.zeros(3)
    t = np.nanmedian((v - mo @ R.T)[ok], axis=0)
    for _ in range(5):
        res = np.linalg.norm(v - (mo @ R.T + t), axis=1)
        keep = ok & (res < exclude_mm)
        if keep.sum() < 10:
            break
        t = np.mean((v - mo @ R.T)[keep], axis=0)
    return t


def umeyama_sim(P, Q):
    """Similarity transform mapping P -> Q as s*R*P + t (Umeyama 1991). Returns (R, t, s, rms).
    Rigid Kabsch is the special case s=1. A stable s!=1 across a session's trials indicates a
    real metric-scale mismatch (calibration), NOT rotation-degeneracy noise (which gives a
    variable s that absorbs residual)."""
    muP = P.mean(0); muQ = Q.mean(0)
    Pc = P - muP; Qc = Q - muQ
    H = Pc.T @ Qc / len(P)
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = Vt.T @ S @ U.T
    s = float(np.trace(np.diag(D) @ S) / ((Pc ** 2).sum() / len(P)))
    t = muQ - s * R @ muP
    rms = float(np.sqrt(((Q - (s * (P @ R.T) + t)) ** 2).sum(1).mean()))
    return R, t, s, rms


def alignment_for(video, mode, npz=None, tr=None):
    """THE single source of the alignment for a rep, used by BOTH metric scoring AND the overlay.
    mode: 'position' (per-trial rigid Kabsch) | 'session' (session R + per-trial t) |
          'velocity' (velocity-vector R) | 'scale' (per-trial similarity: R, t, AND scale s).
    Returns (R, t, info) for rigid modes; for 'scale' returns (R, t, info) where info['s'] is the
    scale and the caller must apply s (overlay handles this). Loads npz/trial if not given."""
    idx = F.align_index()
    if video not in idx:
        return None
    r = idx[video]; lag = r["lag"]
    if npz is None:
        import glob as _g
        cand = _g.glob(str(F.FUSED_DIR / f"*{video}*.npz"))
        if not cand:
            return None
        npz = np.load(cand[0], allow_pickle=True)
    if tr is None:
        tr = load_trial(r["c3d"])
    raw = np.asarray(npz["cons"], float) if "cons" in npz else np.asarray(npz["fused"], float)
    cent = tr.centroid()
    pos = F.mocap_to_w0(raw, cent, tr.rate, lag)
    if pos is None:
        return None
    Rp, tp, rms = pos
    if mode == "position":
        return Rp, tp, dict(rms=rms)
    if mode == "session":
        sr = session_rotation(video)
        if sr is None:
            return Rp, tp, dict(rms=rms, note="no session R, fell back to position")
        R_sess, ntr, ninl, dev = sr
        t_sess = translation_for_R(raw, cent, tr.rate, lag, R_sess)
        return R_sess, t_sess, dict(ntr=ntr, ninl=ninl, dev=dev)
    if mode == "velocity":
        from overlay import _velocity_fit
        fused = np.asarray(npz["fused"], float)
        Rv, gvel, vlag = _velocity_fit(fused, tr, lag)
        if Rv is None:
            return Rp, tp, dict(rms=rms, note="no velocity R")
        t_v = translation_for_R(raw, cent, tr.rate, lag, Rv)
        return Rv, t_v, dict(good=gvel, vlag=vlag)
    if mode == "scale":
        # SESSION rotation held FIXED (the good branch); fit only scale s + translation t on top.
        # This isolates whether a metric-scale mismatch adds anything BEYOND the session-R fix,
        # instead of a fresh per-trial similarity that would re-introduce the degenerate rotation.
        sr = session_rotation(video)
        R_use = sr[0] if sr is not None else Rp
        v, mo = _synced(raw, cent, tr.rate, lag)
        ok = ~(np.isnan(v).any(1) | np.isnan(mo).any(1))
        if ok.sum() < 10:
            return R_use, translation_for_R(raw, cent, tr.rate, lag, R_use), dict(s=1.0, note="rigid")
        # with R fixed, optimal scale s = <(v-vbar)·(R m - Rmbar)> / |R m - Rmbar|^2, then t
        P = mo[ok] @ R_use.T; Q = v[ok]
        Pc = P - P.mean(0); Qc = Q - Q.mean(0)
        s = float((Pc * Qc).sum() / (Pc ** 2).sum())
        t = Q.mean(0) - s * P.mean(0)
        srms = float(np.sqrt(((Q - (s * P + t)) ** 2).sum(1).mean()))
        return R_use, t, dict(s=s, rms=srms, note="session-R + scale")
    if mode == "simrot":
        # ONE global (R, s, t) fit JOINTLY over the whole session (rotation AND scale together).
        sim = session_similarity(video)
        if sim is None:
            return Rp, tp, dict(s=1.0, note="no session similarity, rigid")
        R, s, t, ntot, nkept = sim
        return R, t, dict(s=s, ntot=ntot, nkept=nkept, note="session global R+s+t")
    raise ValueError(mode)


def _good_frac(vid_fused, omc_w0, drink):
    """Return (all_moving_good, drink_good): fraction of frames with velocity angle < GOOD_ANG,
    over ALL moving frames and over the DRINK-window moving frames. The user cares about BOTH:
    all-moving reflects whether the WHOLE reach/transport tracks (what you see in the video);
    drink-window is the hardest apex sub-window (where the tilt makes even a correct fit disagree)."""
    vm = np.diff(vid_fused, axis=0) * HZ; vo = np.diff(omc_w0, axis=0) * HZ
    sm = np.linalg.norm(vm, axis=1); so = np.linalg.norm(vo, axis=1)
    mv = (sm > SPEED_MM_S) & (so > SPEED_MM_S) & np.isfinite(sm) & np.isfinite(so)
    if mv.sum() < 2:
        return np.nan, np.nan
    ang = np.degrees(np.arccos(np.clip(np.sum(vm * vo, axis=1) / (sm * so + 1e-9), -1, 1)))
    all_good = float(np.mean(ang[mv] < GOOD_ANG))
    dm = mv & drink[:len(mv)]
    drink_good = float(np.mean(ang[dm] < GOOD_ANG)) if dm.sum() >= 2 else np.nan
    return all_good, drink_good


# back-compat: some callers expect the drink number
def _drink_good(vid_fused, omc_w0, drink):
    _, dg = _good_frac(vid_fused, omc_w0, drink)
    return dg


def main():
    files = sorted(glob.glob(str(F.FUSED_DIR / "*.npz")))
    idx = F.align_index()
    print(f"loading + per-trial fits over {len(files)} reps...", flush=True)
    # gather per-trial data
    trials = defaultdict(list)   # session -> list of dict
    for i, f in enumerate(files):
        try:
            npz = np.load(f, allow_pickle=True)
        except Exception:
            continue
        video = str(npz["video"])
        if video not in idx:
            continue
        r = idx[video]; tr = load_trial(r["c3d"])
        raw = np.asarray(npz["cons"], float) if "cons" in npz else np.asarray(npz["fused"], float)
        fused = np.asarray(npz["fused"], float)
        fit = F.mocap_to_w0(raw, tr.centroid(), tr.rate, r["lag"])
        if fit is None:
            continue
        R_i, t_i, _ = fit
        dw = dwell_truth(tr)
        trials[_sk(video)].append(dict(video=video, R=R_i, t=t_i, lag=r["lag"],
                                       raw=raw, fused=fused, cent=tr.centroid(),
                                       rate=tr.rate, dw=dw))
        if (i + 1) % 100 == 0 or i == len(files) - 1:
            print(f"  [{i+1}/{len(files)}] {len(trials)} sessions", flush=True)

    # track BOTH all-moving and drink-window good-frame%, for per-trial and session-R
    pt_all, ss_all, pt_dr, ss_dr = [], [], [], []
    rescued = []
    print(f"\n{'session':<14}{'n':>4}  {'per-trial all/drink':>20}  {'session-R all/drink':>20}",
          flush=True)
    for key in sorted(trials):
        reps = trials[key]
        if len(reps) < 3:
            continue
        Rs = [x["R"] for x in reps]
        R0 = _chordal_mean(Rs)
        inl = [R for R in Rs if _rotang(R0, R) < 20]
        R_sess = _chordal_mean(inl) if len(inl) >= 3 else R0   # robust session rotation
        s_pt_all, s_ss_all = [], []
        for x in reps:
            v_s, mo_s = _synced(x["fused"], x["cent"], x["rate"], x["lag"])
            n = len(v_s)
            drink = np.zeros(max(n - 1, 1), bool)
            sp = x["dw"].span_at(n) if x["dw"].span else None
            if sp:
                drink[sp[0]:min(sp[1], n - 1)] = True
            # per-trial: own R,t
            a_pt, d_pt = _good_frac(v_s, mo_s @ x["R"].T + x["t"], drink)
            # session-R: fix R=R_sess, per-trial t via the SHARED translation_for_R
            t_sess = translation_for_R(x["raw"], x["cent"], x["rate"], x["lag"], R_sess)
            a_ss, d_ss = _good_frac(v_s, mo_s @ R_sess.T + t_sess, drink)
            if np.isfinite(a_pt) and np.isfinite(a_ss):
                pt_all.append(a_pt); ss_all.append(a_ss)
                s_pt_all.append(a_pt); s_ss_all.append(a_ss)
                if np.isfinite(d_pt): pt_dr.append(d_pt)
                if np.isfinite(d_ss): ss_dr.append(d_ss)
                if a_ss - a_pt > 0.20:
                    rescued.append((x["video"], a_pt, a_ss))
        if s_pt_all:
            print(f"{key:<14}{len(reps):>4}  {np.median(s_pt_all)*100:18.0f}%  "
                  f"{np.median(s_ss_all)*100:18.0f}%", flush=True)

    pt = np.array(pt_all) * 100; ss = np.array(ss_all) * 100
    ptd = np.array(pt_dr) * 100; ssd = np.array(ss_dr) * 100
    print(f"\n  reps scored: {len(pt)}")
    print(f"  === ALL-MOVING window (whole reach/transport) ===")
    print(f"  PER-TRIAL fit  good-frame%: median {np.median(pt):.0f}%  mean {pt.mean():.0f}%")
    print(f"  SESSION-R  fit good-frame%: median {np.median(ss):.0f}%  mean {ss.mean():.0f}%")
    print(f"  improved (>5pts): {int((ss > pt + 5).sum())}   worsened (>5pts): {int((ss < pt - 5).sum())}")
    print(f"  === DRINK window (apex only) ===")
    print(f"  PER-TRIAL fit  good-frame%: median {np.median(ptd):.0f}%  mean {ptd.mean():.0f}%")
    print(f"  SESSION-R  fit good-frame%: median {np.median(ssd):.0f}%  mean {ssd.mean():.0f}%", flush=True)
    print(f"\n  most rescued reps by ALL-MOVING good-frame (session-R - per-trial > 20pts):")
    for v, a, b in sorted(rescued, key=lambda z: -(z[2] - z[1]))[:15]:
        print(f"    {v[:44]:<46} {a*100:3.0f}% -> {b*100:3.0f}%", flush=True)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.parent.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].scatter(pt, ss, s=9, alpha=0.4, color="#4477aa")
    ax[0].plot([0, 100], [0, 100], "k--", lw=0.8)
    ax[0].set_xlabel("per-trial fit good-frame %"); ax[0].set_ylabel("session-R fit good-frame %")
    ax[0].set_title(f"per-trial vs session-R (above line = better)\n"
                    f"improved {int((ss>pt+5).sum())}  worsened {int((ss<pt-5).sum())}")
    ax[1].hist(pt, bins=25, range=(0, 100), color="#cc6677", alpha=0.6, label=f"per-trial (med {np.median(pt):.0f})")
    ax[1].hist(ss, bins=25, range=(0, 100), color="#228833", alpha=0.6, label=f"session-R (med {np.median(ss):.0f})")
    ax[1].legend(); ax[1].set_xlabel("drink good-frame %"); ax[1].set_ylabel("reps")
    ax[1].set_title("distribution shift")
    fig.tight_layout(); fig.savefig(OUT, dpi=120)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
