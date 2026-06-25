"""KF *interpolation/smoothing* on the consensus, WITHOUT the divergent 2D filtering.

The current pipeline's KF updates from raw per-camera 2D detections with a
Mahalanobis gate -> once it's wrong it rejects every true detection and coasts
away (the P24/P10 divergences). The consensus (>=3-cam gated triangulation) is
computed separately and stays correct.

This keeps what the KF is good at (constant-velocity dynamics -> smooth gap
interpolation) but feeds it the CONSENSUS as a direct 3D measurement with no
gating, so it can't diverge -- every consensus frame pulls the state back.
Predict-only on frames with no consensus = the interpolation. Then RTS smooth.

Compares, per rep, on cached data (no GPU):
  RTS_2d   : current pipeline (consensus->2D-gated KF->RTS)
  LININT   : consensus + linear interpolation (anchored, no dynamics)
  KFi      : consensus -> linear-KF (no gate) -> RTS  (this approach)

    python experiments/drink_study/kf_consensus.py
"""
from __future__ import annotations
import json, glob
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import segment_cup_only as sc
from scipy.signal import find_peaks

TRACK = Path("experiments/drink_study/cache/track3d")
FPS = 60.0
Q = 200.0 ** 2          # process noise (mm/s^2)^2, same as KalmanFilter3D
R_MM = 30.0 ** 2        # consensus 3D position noise (DLT), no gating


def kf_rts_on_consensus(cons, fps=FPS, q=Q, r=R_MM, return_both=False):
    """Linear constant-velocity KF (3D pos measurement = consensus) + RTS smooth.
    cons: (T,3) with NaN where no consensus. Returns (T,3) smoothed positions, or
    (filtered, smoothed) if return_both (filtered = causal, smoothed = RTS)."""
    T = len(cons)
    dt = 1.0 / fps
    F = np.eye(6); F[:3, 3:] = dt * np.eye(3)
    H = np.zeros((3, 6)); H[:, :3] = np.eye(3)
    Qm = np.zeros((6, 6))
    Qm[:3, :3] = q * dt ** 3 / 3 * np.eye(3); Qm[:3, 3:] = q * dt ** 2 / 2 * np.eye(3)
    Qm[3:, :3] = q * dt ** 2 / 2 * np.eye(3); Qm[3:, 3:] = q * dt * np.eye(3)
    Rm = r * np.eye(3)
    valid = np.isfinite(cons).all(1)
    idx = np.flatnonzero(valid)
    if len(idx) < 2:
        return np.full((T, 3), np.nan)
    # init at first consensus
    x = np.zeros(6); x[:3] = cons[idx[0]]
    P = np.diag([50, 50, 50, 500, 500, 500.0]) ** 2
    xs_p, Ps_p, xs_u, Ps_u = [], [], [], []
    for t in range(T):
        # predict
        x = F @ x; P = F @ P @ F.T + Qm
        xs_p.append(x.copy()); Ps_p.append(P.copy())
        # update with consensus (no gate)
        if valid[t]:
            z = cons[t]; y = z - H @ x
            S = H @ P @ H.T + Rm
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y; P = (np.eye(6) - K @ H) @ P
        xs_u.append(x.copy()); Ps_u.append(P.copy())
    # RTS backward
    xs_s = [None] * T; xs_s[-1] = xs_u[-1]
    for t in range(T - 2, -1, -1):
        C = Ps_u[t] @ F.T @ np.linalg.inv(Ps_p[t + 1])
        xs_s[t] = xs_u[t] + C @ (xs_s[t + 1] - xs_p[t + 1])
    smoothed = np.array([s[:3] for s in xs_s])
    if return_both:
        return np.array([u[:3] for u in xs_u]), smoothed
    return smoothed


def lin_interp(track):
    T = len(track); out = track.copy()
    v = np.isfinite(track).all(1); idx = np.flatnonzero(v)
    if len(idx) < 2:
        return out
    for a in range(3):
        out[:, a] = np.interp(np.arange(T), idx, track[idx, a])
    return out


def metrics(xyz):
    r = sc.segment_cup_only(xyz)
    drink = len(r["drink_runs"]) > 0
    rest = np.median(xyz[:30], 0); disp = np.linalg.norm(xyz - rest, axis=1)
    peak = disp.max(); arc = np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()
    detour = arc / (2 * peak) if peak > 50 else np.nan
    ds = np.array([np.median(disp[max(0, i - 4):i + 5]) for i in range(len(disp))])
    rng = ds.max() - ds.min()
    npk = len(find_peaks(ds, prominence=0.05 * rng, distance=int(0.4 * FPS))[0]) if rng > 0 else 0
    C = xyz - xyz.mean(0); ext = (C @ np.linalg.svd(C, full_matrices=False)[2].T)
    lat = float((ext.max(0) - ext.min(0))[2])
    return dict(drink=bool(drink), detour=float(detour), npk=int(npk), lat=lat,
                clean=bool(npk == 1 and detour < 1.6 and lat < 120))


def main():
    files = sorted(f for f in glob.glob(str(TRACK / "P*__pscale_4.json")) if "_summary" not in f)
    rows = []
    for f in files:
        d = json.loads(Path(f).read_text()); fr = d["frames"]
        cons = np.array([x["consensus"] if x["consensus"] else [np.nan] * 3 for x in fr], float)
        rts = np.array([x["rts"] if x["rts"] else [np.nan] * 3 for x in fr], float)
        if np.isfinite(cons).all(1).sum() < 10:
            continue
        m = dict(name=Path(f).stem.replace("__pscale_4", ""),
                 rts=metrics(lin_interp(rts)),
                 lin=metrics(lin_interp(cons)),
                 kfi=metrics(kf_rts_on_consensus(cons)))
        rows.append(m)
    R = len(rows)
    for k, lbl in [("rts", "RTS_2d (current)"), ("lin", "consensus+linint"), ("kfi", "consensus->KF->RTS")]:
        print(f"{lbl:22s}  drink {sum(r[k]['drink'] for r in rows):3d}/{R}   "
              f"clean {sum(r[k]['clean'] for r in rows):3d}/{R}   "
              f"median detour {np.median([r[k]['detour'] for r in rows if np.isfinite(r[k]['detour'])]):.3f}")
    print()
    print("the 7 KF-divergence reps — clean? (RTS_2d -> consensus+linint -> KFi):")
    flagged = ["P24_P24_drinking_right_20240724_105712", "P24_P24_drinking_right_20240724_105744",
               "P10_P10_drinking_right_20240202_153316", "P10_P10_drinking_right_20240202_153332",
               "P10_P10_drinking_right_20240202_153048", "P10_P10_drinking_right_20240202_153157",
               "P10_P10_drinking_right_20240202_153348"]
    by = {r["name"]: r for r in rows}
    for nm in flagged:
        r = by.get(nm)
        if r:
            print(f"  {nm:42s} {r['rts']['clean']!s:5} -> {r['lin']['clean']!s:5} -> {r['kfi']['clean']!s:5}  "
                  f"(KFi detour {r['kfi']['detour']:.2f})")
    json.dump(rows, open("experiments/drink_study/cache/kf_consensus.json", "w"), default=float, indent=1)
    print("\nwrote cache/kf_consensus.json")


if __name__ == "__main__":
    main()
