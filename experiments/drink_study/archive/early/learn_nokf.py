"""Predict cup position WITHOUT the KF in the features, then KF-smooth after (user idea).

The KF-based correction/position models scored high R^2 by ECHOING the KF position
that was in their inputs -> useless (can't beat the KF). Here the model sees only
RAW signals (no kf/rts), so a good prediction means it genuinely learned position
from the camera evidence, not copied the filter. Pipeline:

    raw signals -> learned position model -> KF/RTS smoothing

Distinguishing raw input over the bare consensus: WHICH cameras agree (10-dim binary
'kept'), reprojection tightness (median_px), #cams -> this is the signal that
separates good consensus from confident-WRONG consensus (e.g. cam_4-alone = wrist),
which the KF features never had.

Features per frame (rep-local where 3D), centered window [-5,-2,0,2,5]:
  consensus position (3D, NaN->rest), consensus velocity (3D),
  per-camera kept flags (10), median_px, #cams, frames-since-consensus, occluded.
Target: true cup position (rep-local), LOPO. Output then KF/RTS-smoothed and scored.

    python learn_nokf.py
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
from learn_correction import _rep_frame

CACHE = ROOT / "experiments" / "drink_study" / "cache"
TRACKDIR = CACHE / "track3d_clean3d_refill"
CAMS = [f"cam_{i}" for i in range(1, 11)]
WIN = [-5, -2, 0, 2, 5]


def build_rep_nokf(rep):
    vpath = TRACKDIR / (rep["video"] + ".json")
    if not vpath.exists():
        return None
    tr = Q.load_trial(rep["c3d"])
    if not tr.gt_quality()["ok"]:
        return None
    d = json.loads(vpath.read_text())["frames"]
    cons = np.array([f["consensus"] if f.get("consensus") else [np.nan] * 3 for f in d], float)
    kept = np.array([[1.0 if c in set(f.get("kept", [])) else 0.0 for c in CAMS] for f in d])
    mpx = np.array([f.get("median_px") if f.get("median_px") is not None else 30.0 for f in d], float)
    ncams = kept.sum(1)
    if np.isfinite(cons).all(1).sum() < 20:
        return None
    # align mocap to a KF track (only to get the GT target + sync); KF NOT a feature
    base = kf_rts_on_consensus(cons)
    mr = Q._resample(tr.centroid(), tr.rate)
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
    mo_aligned = mo @ R.T + t                    # true cup in the KF's world frame

    rest = np.median(base[:30], 0)
    basis = _rep_frame(base, rest)
    # consensus in rep-local frame, NaN filled to rest (0 displacement) for features
    cons_filled = cons.copy()
    bad = ~np.isfinite(cons).all(1)
    cons_filled[bad] = rest
    cpos_l = (cons_filled - rest) @ basis.T
    cvel_l = np.vstack([np.zeros(3), np.diff(cpos_l, axis=0)]) * Q.COMMON_HZ
    valid = np.isfinite(cons).all(1)
    fsc = np.zeros(len(d)); c = 0
    for i in range(len(d)):
        c = 0 if valid[i] else c + 1
        fsc[i] = c
    Tn = len(d)

    def at(a, i): return a[min(max(i, 0), Tn - 1)]
    vfi = np.clip(mfi, 0, Tn - 1)
    rows = []
    for fi in vfi:
        ft = []
        for dk in WIN:
            ft += list(at(cpos_l, fi + dk)); ft += list(at(cvel_l, fi + dk))
        ft += list(at(kept, fi))                       # 10 camera-identity flags
        ft += [at(mpx, fi), at(ncams, fi), at(fsc, fi), float(at(ncams, fi) < 2)]
        rows.append(ft)
    feats = np.asarray(rows, float)
    true_local = (mo_aligned - rest) @ basis.T          # target: true cup, rep-local
    return dict(feats=feats, true_local=true_local, vfi=vfi, basis=basis, rest=rest,
                base=base, mr=mr, cons=cons, ncams=ncams,
                video=rep["video"], pid=rep["video"].split("_")[0])


def main():
    print("building no-KF features (ETA ~5 min)...", flush=True)
    reps = []
    for rep in T._reps():
        r = build_rep_nokf(rep)
        if r is not None:
            # phase from the mocap (GT) for scoring the drinking phase
            tr = Q.load_trial(rep["c3d"]); mr = Q._resample(tr.centroid(), tr.rate)
            r["ph"] = S.segment_cup_only(mr, fps=Q.COMMON_HZ)["phase"]
            reps.append(r)
    print(f"{len(reps)} reps, {reps[0]['feats'].shape[1]} features/frame", flush=True)
    pids = sorted({r["pid"] for r in reps})

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import r2_score
    b_de, n_de, common = {}, {}, []
    tr_r2, te_r2 = [], []
    for held in pids:
        trn = [r for r in reps if r["pid"] != held]; te = [r for r in reps if r["pid"] == held]
        Xtr = np.vstack([r["feats"] for r in trn]); Ytr = np.vstack([r["true_local"] for r in trn])
        Xte = np.vstack([r["feats"] for r in te]); Yte = np.vstack([r["true_local"] for r in te])
        M = [HistGradientBoostingRegressor(max_iter=150, max_depth=6).fit(Xtr, Ytr[:, k])
             for k in range(3)]
        tr_r2.append(np.mean([r2_score(Ytr[:, k], M[k].predict(Xtr)) for k in range(3)]))
        te_r2.append(np.mean([r2_score(Yte[:, k], M[k].predict(Xte)) for k in range(3)]))
        for r in te:
            pred_l = np.column_stack([m.predict(r["feats"]) for m in M])
            pred_w = pred_l @ r["basis"] + r["rest"]            # predicted positions (world)
            # KF/RTS-SMOOTH the model output (treat predictions as the measurement)
            meas = np.full_like(r["base"], np.nan); meas[r["vfi"]] = pred_w
            smoothed = kf_rts_on_consensus(meas)
            for trk, store in [(r["base"], b_de), (smoothed, n_de)]:
                sc = T._score(trk, r["mr"])
                if sc is None:
                    continue
                rms, err, mfi = sc; mfi = np.clip(mfi, 0, len(r["ph"]) - 1)
                dm = r["ph"][mfi] == S.P_DRINK
                if dm.any():
                    store[r["video"]] = float(np.median(err[dm]))
            if r["video"] in b_de and r["video"] in n_de:
                common.append(r["video"])
        print(f"  held {held}: {len(te)} reps", flush=True)

    common = [k for k in common if k in b_de and k in n_de]
    B = np.array([b_de[k] for k in common]); N = np.array([n_de[k] for k in common])
    print(f"\n=== NO-KF model -> KF-smoothed (LOPO), {len(common)} reps ===")
    print(f"  TRAIN R^2 {np.mean(tr_r2):.3f}   TEST(LOPO) R^2 {np.mean(te_r2):.3f}")
    print(f"  baseline KF drinking err : {np.median(B):.1f}mm")
    print(f"  no-KF model + KF smooth  : {np.median(N):.1f}mm  ({(np.median(N)/np.median(B)-1)*100:+.0f}%)")
    print(f"  improved {(N<B).sum()}, worsened {(N>B).sum()}, mean {np.mean(N-B):+.1f}mm")
    json.dump({"n": len(common), "train_r2": float(np.mean(tr_r2)),
               "test_r2": float(np.mean(te_r2)), "base": float(np.median(B)),
               "nokf": float(np.median(N)), "improved": int((N < B).sum()),
               "worsened": int((N > B).sum())},
              open(CACHE / "learn_nokf.json", "w"), indent=2)
    print("wrote cache/learn_nokf.json")


if __name__ == "__main__":
    main()
