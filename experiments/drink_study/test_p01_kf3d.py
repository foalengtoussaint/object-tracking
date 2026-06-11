"""Does the temporal 3D KF improve filtering over per-frame 3D gating, on the
cached P01 rep? No re-inference, no retraining — pure analysis of cached dets.

Compares, per frame, the estimated cup 3D position from:
  (A) per-frame inlier-gated triangulation (>=3 agreeing cams, else NO estimate)
  (B) KalmanFilter3D fed every camera's raw 2D detection in time order (its EKF
      gate rejects the glass/bracelet; predict() coasts through occlusion)

Reports: frame coverage (how many frames get a cup position), and whether each
method sits on the REAL cup (consensus of the good cams) vs the static glass.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from kalman_3d import load_calibration, triangulate_dlt, project, KalmanFilter3D

REP = "P01_drinking_right_20231220_141546"
FPS = 60.0
calib = load_calibration("data/calib/P01/calibration.toml", target_size=(1920, 1080))
d = json.load(open(f"experiments/drink_study/cache/P01_{REP}__teacher__c0.25.json"))
dets = {c: [tuple(x) if x else None for x in v] for c, v in d.items()}
cams = sorted(dets, key=lambda k: int(k.split("_")[1]))
n = min(len(v) for v in dets.values())


def gated_consensus(obs, thr=30.0, minc=3):
    """Per-frame inlier-gated 3D point; None if <minc agreeing cams."""
    cur = dict(obs)
    while len(cur) >= 2:
        X = triangulate_dlt([calib[c] for c in cur], [np.array(cur[c]) for c in cur])
        e = {c: float(np.hypot(*(project(calib[c], X)[0] - np.array(cur[c])))) for c in cur}
        w = max(e, key=e.get)
        if e[w] <= thr:
            break
        del cur[w]
    if len(cur) < minc:
        return None
    X = triangulate_dlt([calib[c] for c in cur], [np.array(cur[c]) for c in cur])
    return X


# Reference "true cup" per frame = gated consensus where available (clean signal)
ref = [gated_consensus({c: dets[c][fr] for c in cams if dets[c][fr] is not None})
       for fr in range(n)]

# (A) gate-only coverage
gate_cov = sum(1 for r in ref if r is not None)

# (B) 3D KF fed every camera's raw detection, in time order
kf = KalmanFilter3D()
kf_est = []
for fr in range(n):
    t = fr / FPS
    obs = {c: dets[c][fr] for c in cams if dets[c][fr] is not None}
    if not kf.initialized:
        # seed from a gated consensus so we don't init on the glass
        if ref[fr] is not None:
            kf.init(ref[fr], t)
        kf_est.append(kf.x[:3].copy() if kf.initialized else None)
        continue
    kf.predict(t)
    # feed each camera; EKF gate rejects outliers (glass/bracelet)
    for c in obs:
        kf.update(calib[c], np.array(obs[c]), t)
    kf_est.append(kf.x[:3].copy())

kf_cov = sum(1 for e in kf_est if e is not None)

print(f"frames: {n}")
print(f"(A) gate-only coverage : {gate_cov}/{n} ({100*gate_cov/n:.0f}%)")
print(f"(B) 3D-KF coverage     : {kf_cov}/{n} ({100*kf_cov/n:.0f}%)  <- fills occlusion gaps")
print(f"    frames KF covers that gating could NOT: {sum(1 for fr in range(n) if ref[fr] is None and kf_est[fr] is not None)}")

# Sanity: where gate AND KF both have an estimate, how far apart (mm)? (KF should track the real cup)
both = [(ref[fr], kf_est[fr]) for fr in range(n) if ref[fr] is not None and kf_est[fr] is not None]
if both:
    dist = np.array([np.linalg.norm(a - b) for a, b in both])
    print(f"    KF vs gated-consensus agreement (both present): median {np.median(dist):.0f}mm  p90 {np.percentile(dist,90):.0f}mm")

# Did the KF ever lock onto the glass? Project KF estimate into cam_10, compare to glass pixel (1672,745)
glass = np.array([1672, 745])
on_glass = 0; checked = 0
for fr in range(n):
    if kf_est[fr] is None:
        continue
    uv, infr = project(calib["cam_10"], kf_est[fr])
    if infr:
        checked += 1
        if np.hypot(*(uv - glass)) < 80:
            on_glass += 1
print(f"    KF estimate projects onto the cam_10 GLASS spot: {on_glass}/{checked} frames "
      f"({100*on_glass/checked:.0f}%)  <- should be ~0 if KF tracks the real cup")
