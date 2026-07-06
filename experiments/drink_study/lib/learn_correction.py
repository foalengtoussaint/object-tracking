"""Learned correction model using the tracking<->OMC link.

Idea (user): we have 669 reps with BOTH the video KF track and the mocap truth, so
instead of a hand-built shape prior we LEARN the tracker's systematic error as a
function of runtime features, and subtract it:

    corrected = KF_output + g(features)            g learned, target = mocap - KF

Diagnostic showed the residual is learnable (R^2~0.22 for magnitude; #cams r=-0.58,
displacement r=+0.60, occluded 33mm vs 4.5mm). Features are runtime-only (no mocap):
#cameras, displacement-from-rest, speed, occluded flag, phase, frames-since-consensus.

To generalise across participants/calibrations the residual is learned in a REP-LOCAL
frame: axis 1 = reach direction (rest->peak), axes 2,3 = orthogonal complement. So g
predicts (radial, lateral, vertical) error, not world XYZ.

Leave-one-PARTICIPANT-out, no leakage: g for participant P is trained on all OTHER
participants. Scored with the same sync+Kabsch GT metric on matched reps.

    python learn_correction.py
Run from repo root.
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "drink_study"))
import qtm_align as Q
import segment_cup_only as S
import tune_interp as T
from kf_consensus import kf_rts_on_consensus

from _paths import CACHE


def _rep_frame(base, rest):
    """Rep-local orthonormal basis: e1 = rest->peak reach dir, e2,e3 = complement."""
    disp = np.linalg.norm(base - rest, axis=1)
    pk = base[int(np.argmax(disp))] - rest
    e1 = pk / (np.linalg.norm(pk) + 1e-9)
    a = np.array([0, 0, 1.0]) if abs(e1[2]) < 0.9 else np.array([1.0, 0, 0])
    e2 = a - (a @ e1) * e1; e2 /= np.linalg.norm(e2) + 1e-9
    e3 = np.cross(e1, e2)
    return np.stack([e1, e2, e3])           # (3,3) rows = basis vectors


def build_rep(rep):
    """Return per-frame (features, residual_local, video_frame_idx, basis, rest) for
    one rep, or None. residual_local = (mocap - KF) projected into the rep frame."""
    L = T._load_rep(rep)
    if L is None:
        return None
    cons, mr, ph, ncams = L
    base = kf_rts_on_consensus(cons)
    sc = T._score(base, mr)
    if sc is None:
        return None
    # recover per-frame aligned mocap to form the residual VECTOR (not just |.|)
    vr = Q._resample(base, Q.VIDEO_FPS)
    lag, corr = Q._xcorr_lag(Q._speed(vr, Q.COMMON_HZ), Q._speed(mr, Q.COMMON_HZ))
    if corr < Q.MIN_SYNC_CORR:
        return None
    if lag >= 0:
        v = vr[lag:]; mo = mr[:len(v)]; mfi = np.arange(len(v))
    else:
        mo = mr[-lag:]; v = vr[:len(mo)]; mfi = np.arange(-lag, -lag + len(mo))
    Ln = min(len(v), len(mo)); v, mo, mfi = v[:Ln], mo[:Ln], mfi[:Ln]
    ok = ~(np.isnan(v).any(1) | np.isnan(mo).any(1))
    v, mo, mfi = v[ok], mo[ok], mfi[ok]
    if len(v) < 10:
        return None
    R, t, _ = Q.kabsch(mo, v, robust=True)
    mo_aligned = mo @ R.T + t                       # mocap in the KF's frame
    resid_world = mo_aligned - v                    # (N,3) mocap - KF
    rest = np.median(base[:30], 0)
    basis = _rep_frame(base, rest)

    # --- 3D kinematics in the REP-LOCAL FRAME (not scalars; direction kept) ---
    pos_l = (base - rest) @ basis.T                 # (T,3) position vs rest, rep frame
    vel_l = np.vstack([np.zeros(3), np.diff(pos_l, axis=0)]) * Q.COMMON_HZ   # 3D velocity
    acc_l = np.vstack([np.zeros(3), np.diff(vel_l, axis=0)]) * Q.COMMON_HZ   # 3D accel (how speed changed)
    valid = np.isfinite(cons).all(1)
    fsc = np.zeros(len(base)); c = 0
    for i in range(len(base)):
        c = 0 if (i < len(valid) and valid[i]) else c + 1
        fsc[i] = c
    nc = np.asarray(ncams[:len(base)], float)
    T_ = len(base)

    def at(arr, i):                                 # safe indexing with edge clamp
        return arr[min(max(i, 0), T_ - 1)]

    # centered window: a few frames BACK and a few frames AFTER (offline / RTS is acausal)
    WIN = [-5, -2, 0, 2, 5]
    vfi = np.clip(mfi, 0, T_ - 1)
    rows = []
    for fi in vfi:
        feat = []
        for dk in WIN:                              # 3D pos + 3D vel at each lag/lead
            feat += list(at(pos_l, fi + dk)); feat += list(at(vel_l, fi + dk))
        feat += list(acc_l[min(max(fi, 0), T_ - 1)])       # 3D acceleration at t
        # occlusion context across the window
        feat += [at(nc, fi - 5), at(nc, fi), at(nc, fi + 5),
                 at(fsc, fi), float(at(nc, fi) < 2)]
        rows.append(feat)
    feats = np.asarray(rows, float)
    resid_local = resid_world @ basis.T             # target: residual in rep frame
    return dict(feats=feats, resid_local=resid_local, vfi=vfi, basis=basis,
                base=base, mr=mr, ph=ph, video=rep["video"],
                pid=rep["video"].split("_")[0])


def main():
    print("building per-rep features + residuals (ETA ~4 min)...", flush=True)
    reps = []
    for rep in T._reps():
        r = build_rep(rep)
        if r is not None:
            reps.append(r)
    print(f"{len(reps)} reps usable", flush=True)
    pids = sorted({r["pid"] for r in reps})

    from sklearn.ensemble import HistGradientBoostingRegressor
    base_de, corr_de, changed = {}, {}, []
    for held in pids:
        train = [r for r in reps if r["pid"] != held]
        test = [r for r in reps if r["pid"] == held]
        Xtr = np.vstack([r["feats"] for r in train])
        Ytr = np.vstack([r["resid_local"] for r in train])
        models = [HistGradientBoostingRegressor(max_iter=150, max_depth=6).fit(Xtr, Ytr[:, k])
                  for k in range(3)]
        for r in test:
            pred_local = np.column_stack([m.predict(r["feats"]) for m in models])
            pred_world = pred_local @ r["basis"]        # back to world
            corrected = r["base"].copy()
            corrected[r["vfi"]] += pred_world           # KF + g(features)
            # score both, drinking-phase median error, matched
            for trk, store in [(r["base"], base_de), (corrected, corr_de)]:
                sc = T._score(trk, r["mr"])
                if sc is None:
                    continue
                rms, err, mfi = sc; mfi = np.clip(mfi, 0, len(r["ph"]) - 1)
                dm = r["ph"][mfi] == S.P_DRINK
                if dm.any():
                    store[r["video"]] = float(np.median(err[dm]))
            if r["video"] in base_de and r["video"] in corr_de:
                changed.append(r["video"])
        print(f"  held {held}: {len(test)} reps", flush=True)

    common = [k for k in changed if k in base_de and k in corr_de]
    b = np.array([base_de[k] for k in common]); c = np.array([corr_de[k] for k in common])
    print(f"\n=== LEARNED CORRECTION (LOPO), {len(common)} reps ===")
    print(f"  baseline KF drinking err : {np.median(b):.1f}mm")
    print(f"  KF + learned correction  : {np.median(c):.1f}mm")
    print(f"  delta: {np.median(c)-np.median(b):+.1f}mm  ({(np.median(c)/np.median(b)-1)*100:+.0f}%)")
    print(f"  improved {(c<b).sum()}, worsened {(c>b).sum()}, mean change {np.mean(c-b):+.1f}mm")
    json.dump({"n": len(common), "base": float(np.median(b)), "corrected": float(np.median(c)),
               "improved": int((c < b).sum()), "worsened": int((c > b).sum())},
              open(CACHE / "learn_correction.json", "w"), indent=2)
    print("\nwrote cache/learn_correction.json")


if __name__ == "__main__":
    main()
