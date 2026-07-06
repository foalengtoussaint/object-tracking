"""Improve the consensus->KF->RTS cup interpolator using OMC (mocap) as ground truth.

Now that mocap is aligned to every rep, we can score the smoother's REAL mm error
(not proxy metrics) and tune it. Three stages, each scored vs GT on all reps:

  1. baseline   : current params (Q=200^2, R=30^2), constant-velocity  (kf_consensus)
  2. tuned      : grid-search Q,R to minimise GT mm error
  3. occ-aware  : + position-hold when consensus coverage drops at the slow dwell
                  (the apex-drift fix), validated vs GT.

Error = robust-Kabsch inlier RMS of the smoothed track vs the synced mocap centroid
(same metric as qtm_align), reported overall and IN THE DRINKING PHASE (where the
filter has to interpolate through occlusion — that is where gains matter).

    python tune_interp.py --stage baseline
    python tune_interp.py --stage tune
    python tune_interp.py --stage occ
Run from repo root.
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "drink_study"))
import qtm_align as Q
import segment_cup_only as S
from kf_consensus import kf_rts_on_consensus

from _paths import CACHE
TRACKDIR = CACHE / "track3d_clean3d_refill"


def _load_rep(rep):
    """Return (consensus(T,3) NaN-gapped, mocap_resampled(Tm,3), lag, phase, coverage)
    for a rep, or None. Consensus + mocap are the inputs; mocap is GT."""
    vpath = TRACKDIR / (rep["video"] + ".json")
    if not vpath.exists():
        return None
    tr = Q.load_trial(rep["c3d"])
    if not tr.gt_quality()["ok"]:
        return None
    d = json.loads(vpath.read_text())["frames"]
    cons = np.array([f["consensus"] if f.get("consensus") else [np.nan] * 3 for f in d], float)
    ncams = np.array([len(f.get("kept", [])) for f in d])
    if np.isfinite(cons).all(1).sum() < 10:
        return None
    mr = Q._resample(tr.centroid(), tr.rate)
    seg = S.segment_cup_only(mr, fps=Q.COMMON_HZ)
    return cons, mr, seg["phase"], ncams


def _score(track, mr):
    """Robust-Kabsch inlier RMS of a video track vs mocap, with per-frame error +
    the sync lag (so we can attribute error to phase). Returns (rms, err, vfi)."""
    vr = Q._resample(track, Q.VIDEO_FPS)
    lag, corr = Q._xcorr_lag(Q._speed(vr, Q.COMMON_HZ), Q._speed(mr, Q.COMMON_HZ))
    if corr < Q.MIN_SYNC_CORR:
        return None
    if lag >= 0:
        v = vr[lag:]; mo = mr[:len(v)]; mfi = np.arange(len(v))
    else:
        mo = mr[-lag:]; v = vr[:len(mo)]; mfi = np.arange(-lag, -lag + len(mo))
    L = min(len(v), len(mo)); v, mo, mfi = v[:L], mo[:L], mfi[:L]
    ok = ~(np.isnan(v).any(1) | np.isnan(mo).any(1))
    v, mo, mfi = v[ok], mo[ok], mfi[ok]
    if len(v) < 10:
        return None
    R, t, rms = Q.kabsch(mo, v, robust=True)
    err = np.linalg.norm(v - (mo @ R.T + t), axis=1)
    return rms, err, mfi


def occ_aware(cons, ncams, fps=Q.COMMON_HZ, q=200.0**2, r=30.0**2,
              hold_cams=2, q_hold=20.0**2):
    """KF->RTS but with a POSITION-HOLD process model when consensus coverage is low
    (ncams < hold_cams): drop process noise so the state stops coasting and holds
    near the last good consensus through the occluded dwell, instead of drifting on
    a stale velocity. Validated vs GT."""
    T = len(cons); dt = 1.0 / fps
    Fc = np.eye(6); Fc[:3, 3:] = dt * np.eye(3)
    H = np.zeros((3, 6)); H[:, :3] = np.eye(3)
    Rm = r * np.eye(3)

    def Qm(qq):
        m = np.zeros((6, 6))
        m[:3, :3] = qq * dt**3 / 3 * np.eye(3); m[:3, 3:] = qq * dt**2 / 2 * np.eye(3)
        m[3:, :3] = qq * dt**2 / 2 * np.eye(3); m[3:, 3:] = qq * dt * np.eye(3)
        return m
    Qhi, Qlo = Qm(q), Qm(q_hold)
    valid = np.isfinite(cons).all(1); idx = np.flatnonzero(valid)
    if len(idx) < 2:
        return np.full((T, 3), np.nan)
    x = np.zeros(6); x[:3] = cons[idx[0]]
    P = np.diag([50, 50, 50, 500, 500, 500.0])**2
    xs_p, Ps_p, xs_u, Ps_u, Fs = [], [], [], [], []
    for t in range(T):
        low = ncams[t] < hold_cams
        F = Fc.copy()
        if low:
            F[:3, 3:] = 0.0          # hold position: ignore velocity coasting
        Qt = Qlo if low else Qhi
        x = F @ x; P = F @ P @ F.T + Qt
        xs_p.append(x.copy()); Ps_p.append(P.copy()); Fs.append(F)
        if valid[t]:
            z = cons[t]; y = z - H @ x
            Sm = H @ P @ H.T + Rm; K = P @ H.T @ np.linalg.inv(Sm)
            x = x + K @ y; P = (np.eye(6) - K @ H) @ P
        xs_u.append(x.copy()); Ps_u.append(P.copy())
    xs_s = [None] * T; xs_s[-1] = xs_u[-1]
    for t in range(T - 2, -1, -1):
        C = Ps_u[t] @ Fs[t + 1].T @ np.linalg.inv(Ps_p[t + 1])
        xs_s[t] = xs_u[t] + C @ (xs_s[t + 1] - xs_p[t + 1])
    return np.array([s[:3] for s in xs_s])


def gp_interp(cons, ncams, fps=Q.COMMON_HZ, length_s=0.25, noise_mm=30.0):
    """Gaussian-process interpolation of the consensus track. Unlike a constant-
    velocity KF, a GP REVERTS TO THE MEAN over long gaps instead of coasting on a
    stale velocity — so at the occluded dwell it relaxes toward the local data
    rather than drifting. RBF kernel (length scale ~ a fast cup move) + white noise
    (consensus DLT jitter). Fit per axis on the observed frames; predict all T."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as Ck
    T = len(cons); t = np.arange(T) / fps
    valid = np.isfinite(cons).all(1); idx = np.flatnonzero(valid)
    if len(idx) < 5:
        return np.full((T, 3), np.nan)
    out = np.empty((T, 3))
    # amplitude ~ trajectory spread; fixed kernel (no expensive per-rep optimisation)
    for a in range(3):
        y = cons[idx, a]
        amp = max(np.std(y), 1.0)
        k = Ck(amp**2, "fixed") * RBF(length_s, "fixed") + WhiteKernel(noise_mm**2, "fixed")
        gp = GaussianProcessRegressor(kernel=k, optimizer=None, normalize_y=True)
        gp.fit(t[idx, None], y)
        out[:, a] = gp.predict(t[:, None])
    return out


def _reps():
    align = json.load(open(CACHE / "qtm_align.json"))
    for pp in align.values():
        if isinstance(pp, dict) and pp.get("ok"):
            for rep in pp["reps"]:
                yield rep


# ---- learned mocap shape prior --------------------------------------------------
# The drink-reach displacement-from-rest follows a low-variance canonical profile
# (rise to a mouth dwell ~t=0.4-0.6, symmetric descent). We learn that profile from
# mocap and use it to fill the OCCLUDED apex of the consensus track, instead of a
# constant-velocity coast. Anchored to each rep's own rest position + peak reach
# direction from its good consensus frames. Trained LEAVE-ONE-PARTICIPANT-OUT.
SHAPE_T = np.linspace(0, 1, 100)


def learn_shapes(exclude_pid=None):
    """Mean normalised displacement profile over the movement window, from mocap.
    Returns (100,) canonical disp profile in [0,1] (frac of peak reach)."""
    prof = []
    for rep in _reps():
        if exclude_pid and rep["c3d"].startswith(exclude_pid):
            continue
        tr = Q.load_trial(rep["c3d"])
        if not tr.gt_quality()["ok"]:
            continue
        mr = Q._resample(tr.centroid(), tr.rate)
        seg = S.segment_cup_only(mr, fps=Q.COMMON_HZ)
        mv = np.isin(seg["phase"], [S.P_FWD, S.P_DRINK, S.P_BACK])
        if mv.sum() < 20:
            continue
        idx = np.flatnonzero(mv); d = seg["disp"][idx.min():idx.max() + 1]
        prof.append(np.interp(SHAPE_T, np.linspace(0, 1, len(d)), d) / (d.max() + 1e-9))
    return np.mean(prof, axis=0) if prof else None


def shape_prior_fill(cons, ncams, shape, fps=Q.COMMON_HZ, hold_cams=2):
    """Fill low-coverage (occluded) frames using the learned shape. Where coverage
    is good we keep an RTS-smoothed consensus; where it drops (ncams<hold_cams) we
    replace the position with: rest + shape(t)*peak_reach along the rest->peak
    direction, anchored to the rep's own good frames. Falls back to KF if no
    usable anchor."""
    base = kf_rts_on_consensus(cons)              # smoothed consensus baseline
    valid = np.isfinite(cons).all(1)
    if valid.sum() < 20:
        return base
    # rest position + peak reach from GOOD frames only
    rest = np.median(cons[np.flatnonzero(valid)[:int(0.5 * fps)]], axis=0)
    dgood = np.linalg.norm(cons[valid] - rest, axis=1)
    peak = np.percentile(dgood, 95)
    pk_idx = np.flatnonzero(valid)[np.argmax(np.linalg.norm(cons[valid] - rest, axis=1))]
    direction = cons[pk_idx] - rest
    direction = direction / (np.linalg.norm(direction) + 1e-9)
    # movement window from the baseline track's own segmentation
    seg = S.segment_cup_only(base, fps=fps)
    mv = np.isin(seg["phase"], [S.P_FWD, S.P_DRINK, S.P_BACK])
    if mv.sum() < 20:
        return base
    idx = np.flatnonzero(mv); lo, hi = idx.min(), idx.max()
    out = base.copy()
    low = ncams < hold_cams
    for t in range(lo, hi + 1):
        if low[t]:
            frac = (t - lo) / max(hi - lo, 1)
            mag = np.interp(frac, SHAPE_T, shape) * peak
            out[t] = rest + mag * direction
    return out


def shape_prior_fill_v2(cons, ncams, shape, fps=Q.COMMON_HZ, hold_cams=2):
    """Fair-shot shape prior. Fixes v1's three flaws:
      (1) NO straight-line assumption — fill each gap by interpolating the 3-D
          consensus endpoints, then RESHAPE only the radial magnitude toward the
          learned template (direction comes from the smooth endpoint interpolation,
          which curves);
      (2) time-ALIGN the template apex (frac where shape peaks) to the rep's own
          observed peak-displacement frame, so the dwell lines up;
      (3) BLEND template magnitude with the interpolated magnitude using a window
          that is 1 deep inside the gap and 0 at the good edges -> no discontinuity.
    Outside gaps: the normal RTS-smoothed consensus."""
    base = kf_rts_on_consensus(cons)
    valid = np.isfinite(cons).all(1)
    if valid.sum() < 20:
        return base
    vidx = np.flatnonzero(valid)
    rest = np.median(cons[vidx[:int(0.5 * fps)]], axis=0)
    # rep's own displacement-from-rest of the smoothed track + its peak frame
    disp = np.linalg.norm(base - rest, axis=1)
    seg = S.segment_cup_only(base, fps=fps)
    mv = np.isin(seg["phase"], [S.P_FWD, S.P_DRINK, S.P_BACK])
    if mv.sum() < 20:
        return base
    midx = np.flatnonzero(mv); lo, hi = midx.min(), midx.max()
    peak_frame = lo + int(np.argmax(disp[lo:hi + 1]))
    peak_val = disp[peak_frame]
    # template peak location (frac) -> piecewise-linear time warp so template apex
    # maps onto the rep's peak_frame
    t_pk = SHAPE_T[int(np.argmax(shape))]
    def template_mag(t):                       # t = frame index
        if t <= peak_frame:
            frac = t_pk * (t - lo) / max(peak_frame - lo, 1)
        else:
            frac = t_pk + (1 - t_pk) * (t - peak_frame) / max(hi - peak_frame, 1)
        return np.interp(np.clip(frac, 0, 1), SHAPE_T, shape) * peak_val

    out = base.copy()
    low = (ncams < hold_cams)
    # process each contiguous low-coverage RUN inside the movement window
    t = lo
    while t <= hi:
        if not low[t]:
            t += 1; continue
        s = t
        while t <= hi and low[t]:
            t += 1
        e = t - 1                              # gap [s..e]
        a = s - 1 if s - 1 >= 0 else s         # good anchors either side
        b = e + 1 if e + 1 < len(base) else e
        pa, pb = base[a], base[b]
        glen = max(e - s + 1, 1)
        for k, ti in enumerate(range(s, e + 1)):
            w = (k + 1) / (glen + 1)
            lin = (1 - w) * pa + w * pb        # smooth (curving) endpoint interp
            dlin = lin - rest
            rlin = np.linalg.norm(dlin) + 1e-9
            dirn = dlin / rlin                 # direction from the interpolation (curves)
            mag_t = template_mag(ti)           # radial magnitude from the warped template
            # blend: deep in the gap trust the template magnitude, at edges trust lin
            depth = min(k + 1, glen - k) / ((glen + 1) / 2)   # 0 at edges -> 1 mid-gap
            depth = min(depth, 1.0)
            mag = (1 - depth) * rlin + depth * mag_t
            out[ti] = rest + mag * dirn
    return out


def learn_shapes_pos_speed(exclude_pid=None):
    """Mean normalised DISPLACEMENT and SPEED profiles over the movement window,
    from mocap (leave-one-participant-out). disp in [0,1] of peak reach; speed in
    [0,1] of peak speed (bimodal: fast up, ~0 at the dwell, fast down)."""
    dprof, sprof = [], []
    for rep in _reps():
        if exclude_pid and rep["video"].startswith(exclude_pid):
            continue
        tr = Q.load_trial(rep["c3d"])
        if not tr.gt_quality()["ok"]:
            continue
        mr = Q._resample(tr.centroid(), tr.rate)
        seg = S.segment_cup_only(mr, fps=Q.COMMON_HZ)
        mv = np.isin(seg["phase"], [S.P_FWD, S.P_DRINK, S.P_BACK])
        if mv.sum() < 20:
            continue
        idx = np.flatnonzero(mv); lo, hi = idx.min(), idx.max()
        d = seg["disp"][lo:hi + 1]; s = seg["speed"][lo:hi + 1]
        dprof.append(np.interp(SHAPE_T, np.linspace(0, 1, len(d)), d) / (d.max() + 1e-9))
        sprof.append(np.interp(SHAPE_T, np.linspace(0, 1, len(s)), s) / (s.max() + 1e-9))
    if not dprof:
        return None, None
    return np.mean(dprof, axis=0), np.mean(sprof, axis=0)


def kf_shape_prior(cons, ncams, disp_shape, speed_shape, fps=Q.COMMON_HZ,
                   q=200.0**2, r=30.0**2, hold_cams=2,
                   r_pos_prior=80.0**2, r_spd_prior=60.0**2):
    """Consensus KF + RTS, but during occlusion (ncams<hold_cams) the learned shape
    enters as SOFT PSEUDO-MEASUREMENTS rather than overwriting the state:
      - position pseudo-meas = rest + disp_shape(t)*peak*direction   (soft, r_pos_prior)
      - SPEED  pseudo-meas   = speed_shape(t)*peak_speed  on |velocity|  (so the ~0
        dwell speed is enforced — the cup is known to be still at the mouth).
    The KF blends these with any real consensus + its own dynamics and RTS-smooths,
    so it KEEPS the filter (no hard overwrite, no edge discontinuity). Per @user:
    use the KF *with* the prior, and use speed too. No confidence gate.

    Anchoring (rest / peak reach / direction / peak speed / apex frame) is taken
    from a first KF pass on the consensus alone."""
    base = kf_rts_on_consensus(cons, fps=fps, q=q, r=r)
    valid = np.isfinite(cons).all(1)
    if valid.sum() < 20:
        return base
    vidx = np.flatnonzero(valid)
    rest = np.median(cons[vidx[:int(0.5 * fps)]], axis=0)
    disp = np.linalg.norm(base - rest, axis=1)
    seg = S.segment_cup_only(base, fps=fps)
    mv = np.isin(seg["phase"], [S.P_FWD, S.P_DRINK, S.P_BACK])
    if mv.sum() < 20:
        return base
    midx = np.flatnonzero(mv); lo, hi = midx.min(), midx.max()
    peak_frame = lo + int(np.argmax(disp[lo:hi + 1]))
    peak_val = disp[peak_frame]
    peak_speed = float(np.percentile(seg["speed"][lo:hi + 1], 95)) + 1e-6
    pk_dir = base[peak_frame] - rest
    pk_dir = pk_dir / (np.linalg.norm(pk_dir) + 1e-9)
    t_pk = SHAPE_T[int(np.argmax(disp_shape))]

    def warp(t):
        if t <= peak_frame:
            f = t_pk * (t - lo) / max(peak_frame - lo, 1)
        else:
            f = t_pk + (1 - t_pk) * (t - peak_frame) / max(hi - peak_frame, 1)
        return float(np.clip(f, 0, 1))

    T_ = len(cons); dt = 1.0 / fps
    F = np.eye(6); F[:3, 3:] = dt * np.eye(3)
    Hp = np.zeros((3, 6)); Hp[:, :3] = np.eye(3)        # position rows
    Qm = np.zeros((6, 6))
    Qm[:3, :3] = q * dt**3 / 3 * np.eye(3); Qm[:3, 3:] = q * dt**2 / 2 * np.eye(3)
    Qm[3:, :3] = q * dt**2 / 2 * np.eye(3); Qm[3:, 3:] = q * dt * np.eye(3)
    Rm = r * np.eye(3)
    x = np.zeros(6); x[:3] = cons[vidx[0]]
    P = np.diag([50, 50, 50, 500, 500, 500.0])**2
    xs_p, Ps_p, xs_u, Ps_u = [], [], [], []
    for t in range(T_):
        x = F @ x; P = F @ P @ F.T + Qm
        xs_p.append(x.copy()); Ps_p.append(P.copy())
        if valid[t]:                                    # real consensus
            y = cons[t] - Hp @ x; Sm = Hp @ P @ Hp.T + Rm
            K = P @ Hp.T @ np.linalg.inv(Sm)
            x = x + K @ y; P = (np.eye(6) - K @ Hp) @ P
        elif lo <= t <= hi and ncams[t] < hold_cams:    # occluded -> shape pseudo-meas
            f = warp(t)
            # position pseudo-measurement
            zp = rest + np.interp(f, SHAPE_T, disp_shape) * peak_val * pk_dir
            y = zp - Hp @ x; Sm = Hp @ P @ Hp.T + r_pos_prior * np.eye(3)
            K = P @ Hp.T @ np.linalg.inv(Sm); x = x + K @ y; P = (np.eye(6) - K @ Hp) @ P
            # speed pseudo-measurement: |velocity| should match the template speed
            spd_target = np.interp(f, SHAPE_T, speed_shape) * peak_speed
            v = x[3:]; vn = np.linalg.norm(v) + 1e-9
            Hv = np.zeros((1, 6)); Hv[0, 3:] = v / vn          # d|v|/dv = unit velocity
            yv = np.array([spd_target - vn]); Sv = Hv @ P @ Hv.T + r_spd_prior
            Kv = P @ Hv.T @ np.linalg.inv(Sv)
            x = x + (Kv * yv).ravel(); P = (np.eye(6) - Kv @ Hv) @ P
        xs_u.append(x.copy()); Ps_u.append(P.copy())
    xs_s = [None] * T_; xs_s[-1] = xs_u[-1]
    for t in range(T_ - 2, -1, -1):
        C = Ps_u[t] @ F.T @ np.linalg.inv(Ps_p[t + 1])
        xs_s[t] = xs_u[t] + C @ (xs_s[t + 1] - xs_p[t + 1])
    return np.array([s[:3] for s in xs_s])


def _reps_list():
    return list(_reps())


def evaluate(make_track, label):
    """Run a track-builder over all reps; report overall + drinking-phase RMS."""
    all_rms, drink_err, all_err = [], [], []
    n = 0
    for rep in _reps():
        L = _load_rep(rep)
        if L is None:
            continue
        cons, mr, ph, ncams = L
        track = make_track(cons, ncams)
        sc = _score(track, mr)
        if sc is None:
            continue
        rms, err, mfi = sc
        n += 1
        all_rms.append(rms)
        mfi = np.clip(mfi, 0, len(ph) - 1)
        dmask = ph[mfi] == S.P_DRINK
        if dmask.any():
            drink_err.append(float(np.median(err[dmask])))
        all_err.extend(err.tolist())
    all_err = np.array(all_err)
    print(f"{label:24s} n={n:3d}  median inlier RMS {np.median(all_rms):5.2f}mm  "
          f"drink-phase median err {np.median(drink_err):5.1f}mm  "
          f"p90 err {np.percentile(all_err,90):5.1f}mm")
    return dict(label=label, n=n, median_rms=float(np.median(all_rms)),
                drink_median=float(np.median(drink_err)),
                p90=float(np.percentile(all_err, 90)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["baseline", "tune", "occ", "gp", "shape", "all"],
                    default="all")
    a = ap.parse_args()
    # merge into existing results so a single-stage run doesn't clobber the others
    out_path = CACHE / "tune_interp.json"
    results = json.load(open(out_path)) if (a.stage != "all" and out_path.exists()) else {}

    if a.stage in ("baseline", "tune", "all"):
        results["baseline"] = evaluate(
            lambda c, nc: kf_rts_on_consensus(c, q=200.0**2, r=30.0**2),
            "baseline Q200/R30")

    if a.stage in ("tune", "all"):
        best = None
        for q in (50.0, 100.0, 200.0, 400.0, 800.0):
            for r in (10.0, 20.0, 30.0, 50.0):
                res = evaluate(lambda c, nc, q=q, r=r: kf_rts_on_consensus(c, q=q**2, r=r**2),
                               f"tune Q{q:.0f}/R{r:.0f}")
                if best is None or res["median_rms"] < best["median_rms"]:
                    best = res | dict(q=q, r=r)
        print(f"\nBEST tuned: Q{best['q']:.0f}/R{best['r']:.0f}  "
              f"median RMS {best['median_rms']:.2f}mm (baseline "
              f"{results.get('baseline',{}).get('median_rms',float('nan')):.2f})")
        results["tuned_best"] = best

    if a.stage in ("occ", "all"):
        q = results.get("tuned_best", {}).get("q", 200.0)
        r = results.get("tuned_best", {}).get("r", 30.0)
        results["occ_aware"] = evaluate(
            lambda c, nc: occ_aware(c, nc, q=q**2, r=r**2),
            f"occ-aware Q{q:.0f}/R{r:.0f}")

    if a.stage in ("gp", "all"):
        for ls in (0.15, 0.25, 0.40):
            res = evaluate(lambda c, nc, ls=ls: gp_interp(c, nc, length_s=ls),
                           f"GP len={ls:.2f}s")
            if results.get("gp_best") is None or res["drink_median"] < results["gp_best"]["drink_median"]:
                results["gp_best"] = res | dict(length_s=ls)

    if a.stage in ("shape", "all"):
        # leave-one-participant-out learned shape prior; cache priors per pid
        shape_cache = {}
        all_rms, drink_err, all_err, n = [], [], [], 0
        for rep in _reps():
            pid = rep["c3d"].split("_")[0][:3] if "_" in rep["c3d"] else rep["c3d"][:3]
            pid = rep["video"].split("_")[0]
            if pid not in shape_cache:
                shape_cache[pid] = learn_shapes(exclude_pid=pid)
            shp = shape_cache[pid]
            L = _load_rep(rep)
            if L is None or shp is None:
                continue
            cons, mr, ph, ncams = L
            track = shape_prior_fill(cons, ncams, shp)
            sc = _score(track, mr)
            if sc is None:
                continue
            rms, err, mfi = sc; n += 1; all_rms.append(rms)
            mfi = np.clip(mfi, 0, len(ph) - 1)
            dm = ph[mfi] == S.P_DRINK
            if dm.any():
                drink_err.append(float(np.median(err[dm])))
            all_err.extend(err.tolist())
        print(f"{'shape-prior (LOPO)':24s} n={n:3d}  median inlier RMS {np.median(all_rms):5.2f}mm  "
              f"drink-phase median err {np.median(drink_err):5.1f}mm  "
              f"p90 err {np.percentile(all_err,90):5.1f}mm")
        results["shape_prior"] = dict(label="shape-prior LOPO", n=n,
                                      median_rms=float(np.median(all_rms)),
                                      drink_median=float(np.median(drink_err)),
                                      p90=float(np.percentile(all_err, 90)))

    json.dump(results, open(CACHE / "tune_interp.json", "w"), indent=2)
    print("\nwrote cache/tune_interp.json")


if __name__ == "__main__":
    main()
