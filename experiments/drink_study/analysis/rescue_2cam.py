"""2-camera gap-rescue on top of the consensus-anchored KF.

The >=3-cam inlier gate drops frames where the cup is seen by only 2 cameras that
agree (or by 3+ that don't tightly co-triangulate). On the P23 151359 drink dwell,
those rejected frames split cleanly into two populations:
  - cam_1/cam_9 pairs: reproj 1-8px, 40-64mm from the RTS prediction  -> the real cup
  - cam_1/cam_3 pairs: reproj <1px but 290-340mm from prediction      -> coherent distractor
Pairwise reproj alone can't tell them apart (the distractor is also tight); the
distance-to-prediction is what separates them.

Two-pass, so rescues are gated against an INDEPENDENT smoothed estimate (not the
filter's own running state) -> keeps the consensus-anchored 'cannot diverge'.
  pass 1: KF+RTS on hard >=3-cam consensus only            (current pipeline)
  rescue: on each rejected frame, take the tightest 2-cam pair; accept if its
          reproj <= PAIR_PX and it lands within PRED_MM of the pass-1 RTS point
  pass 2: KF+RTS on {hard consensus (tight R) + rescues (inflated R)} via per-frame R

    python experiments/drink_study/rescue_2cam.py P23_P23_drinking_right_20240716_151359
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import argparse, itertools, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import kf_accuracy as ka
from kalman_3d import load_calibration, triangulate_dlt, project
from cache_track3d import load_dets, kept_and_px
from kf_consensus import Q, R_MM, FPS

# rescue gates
PAIR_PX = 15.0          # tightest 2-cam pair must reproject within this (px)
PRED_MM = 120.0         # ...and land within this of the pass-1 RTS prediction (mm)
R_RESCUE = (90.0) ** 2  # inflated 3D measurement noise for a rescued (soft) point


def best_pair(obs, calib):
    """Tightest 2-cam pair: (max_reproj_px, X_3d, (camA, camB)) or None."""
    best = None
    keys = list(obs)
    for a, b in itertools.combinations(keys, 2):
        X = triangulate_dlt([calib[a], calib[b]], [np.array(obs[a]), np.array(obs[b])])
        e = max(float(np.hypot(*(project(calib[a], X)[0] - np.array(obs[a])))),
                float(np.hypot(*(project(calib[b], X)[0] - np.array(obs[b])))))
        if best is None or e < best[0]:
            best = (e, X, (a, b))
    return best


def kf_rts_varR(meas, Rs, fps=FPS, q=Q):
    """Linear constant-velocity KF + RTS with a PER-FRAME measurement noise.
    meas: (T,3) NaN where no measurement; Rs: (T,) scalar position variance per frame."""
    T = len(meas); dt = 1.0 / fps
    F = np.eye(6); F[:3, 3:] = dt * np.eye(3)
    H = np.zeros((3, 6)); H[:, :3] = np.eye(3)
    Qm = np.zeros((6, 6))
    Qm[:3, :3] = q * dt ** 3 / 3 * np.eye(3); Qm[:3, 3:] = q * dt ** 2 / 2 * np.eye(3)
    Qm[3:, :3] = q * dt ** 2 / 2 * np.eye(3); Qm[3:, 3:] = q * dt * np.eye(3)
    valid = np.isfinite(meas).all(1); idx = np.flatnonzero(valid)
    if len(idx) < 2:
        return np.full((T, 3), np.nan), np.full((T, 3), np.nan)
    x = np.zeros(6); x[:3] = meas[idx[0]]
    P = np.diag([50, 50, 50, 500, 500, 500.0]) ** 2
    xs_p, Ps_p, xs_u, Ps_u = [], [], [], []
    for t in range(T):
        x = F @ x; P = F @ P @ F.T + Qm
        xs_p.append(x.copy()); Ps_p.append(P.copy())
        if valid[t]:
            z = meas[t]; y = z - H @ x
            S = H @ P @ H.T + Rs[t] * np.eye(3)
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y; P = (np.eye(6) - K @ H) @ P
        xs_u.append(x.copy()); Ps_u.append(P.copy())
    xs_s = [None] * T; xs_s[-1] = xs_u[-1]
    for t in range(T - 2, -1, -1):
        C = Ps_u[t] @ F.T @ np.linalg.inv(Ps_p[t + 1])
        xs_s[t] = xs_u[t] + C @ (xs_s[t + 1] - xs_p[t + 1])
    return (np.array([u[:3] for u in xs_u]), np.array([s[:3] for s in xs_s]))


def run(dets, cams, calib, n, r_rescue=R_RESCUE, verbose=False):
    """Returns dict with consensus, rts_old (hard only), rts_new (rescued), rescued frames."""
    # hard >=3-cam consensus per frame
    cons = []
    for fr in range(n):
        obs = {c: dets[c][fr] for c in cams if dets[c][fr] is not None}
        kept, _ = kept_and_px(obs, calib) if len(obs) >= 2 else ([], None)
        if len(kept) >= 3:
            X = triangulate_dlt([calib[c] for c in kept], [np.array(obs[c]) for c in kept])
            cons.append(X)
        else:
            cons.append(np.array([np.nan] * 3))
    cons = np.array(cons, float)

    # pass 1: hard consensus only (R = R_MM everywhere it exists)
    R1 = np.where(np.isfinite(cons).all(1), R_MM, np.inf)
    _, rts_old = kf_rts_varR(cons, R1)

    # rescue: rejected frames -> tightest 2-cam pair gated vs pass-1 prediction
    meas = cons.copy(); Rs = R1.copy(); rescued = []
    for fr in range(n):
        if np.isfinite(cons[fr]).all():
            continue                                   # already have hard consensus
        if not np.isfinite(rts_old[fr]).all():
            continue                                   # no prediction to gate against
        obs = {c: dets[c][fr] for c in cams if dets[c][fr] is not None}
        if len(obs) < 2:
            continue
        bp = best_pair(obs, calib)
        if bp is None:
            continue
        e, X, pair = bp
        d = float(np.linalg.norm(X - rts_old[fr]))
        if e <= PAIR_PX and d <= PRED_MM:
            meas[fr] = X; Rs[fr] = r_rescue
            rescued.append((fr, pair, round(e, 1), round(d)))

    # pass 2: hard consensus (tight R) + rescues (soft R)
    _, rts_new = kf_rts_varR(meas, Rs)
    return dict(cons=cons, rts_old=rts_old, rts_new=rts_new, rescued=rescued, n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trial")
    ap.add_argument("--full", action="store_true",
                    help="rescued 2-cam point counts AS MUCH as >=3-cam consensus "
                         "(R=R_MM); default is soft (inflated R_RESCUE)")
    args = ap.parse_args()
    p = args.trial.split("_")[0]; stem = args.trial[len(p) + 1:]
    calib = load_calibration(f"data/calib/{p}/calibration.toml", target_size=ka.RES)
    detf = Path(f"experiments/drink_study/cache/student_dets_clean3d_refill/"
                f"{p}_{stem}__clean3d_refill__c0.25.json")
    dets = load_dets(detf, calib)
    cams = sorted(dets, key=lambda k: int(k.split("_")[1]))
    n = min(len(v) for v in dets.values())
    r_rescue = R_MM if args.full else R_RESCUE
    print(f"  rescue weight: {'FULL (R=R_MM, same as >=3-cam)' if args.full else 'SOFT (inflated R)'}")
    r = run(dets, cams, calib, n, r_rescue=r_rescue, verbose=True)

    cov_old = int(np.isfinite(r["cons"]).all(1).sum())
    print(f"[{args.trial}] {n} frames")
    print(f"  hard >=3-cam consensus frames: {cov_old}  ({cov_old/n*100:.0f}%)")
    print(f"  2-cam rescues accepted:        {len(r['rescued'])}  "
          f"(coverage now {(cov_old+len(r['rescued']))/n*100:.0f}%)")
    print(f"  gates: pair reproj <= {PAIR_PX}px AND <= {PRED_MM}mm from RTS pred")
    if r["rescued"]:
        print("  rescued frames (fr, pair, reproj_px, dist_to_pred_mm):")
        for fr, pair, e, d in r["rescued"]:
            print(f"    f{fr:3d}  {pair}  reproj={e:5.1f}px  d={d:4d}mm")
    suffix = "_full" if args.full else ""
    out = Path(f"experiments/drink_study/cache/rescue2cam_{args.trial}{suffix}.json")
    out.write_text(json.dumps({
        "rescued": [list(x) for x in r["rescued"]],
        "rts_old": [list(map(float, x)) if np.isfinite(x).all() else None for x in r["rts_old"]],
        "rts_new": [list(map(float, x)) if np.isfinite(x).all() else None for x in r["rts_new"]],
    }))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
