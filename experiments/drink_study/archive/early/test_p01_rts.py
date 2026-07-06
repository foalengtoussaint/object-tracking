"""RTS smoother (uses FUTURE detections) on the cached P01 rep — offline label
cleaning, so the whole sequence is available. Forward EKF (per-camera, gates the
glass/bracelet) stores predicted+updated x/P and F each frame; a backward
Rauch-Tung-Striebel pass corrects every frame using future frames. The constant-
velocity motion model is applied in both directions => one smooth trajectory,
gaps interpolated between the detections before AND after.

No re-inference, no retrain. Compares forward-only vs RTS-smoothed, projects into
cam_10, renders a video.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import cv2
from kalman_3d import load_calibration, triangulate_dlt, project, KalmanFilter3D

REP = "P01_drinking_right_20231220_141546"
FPS = 60.0
Q_ACCEL = 200.0  # mm/s^2, matches KalmanFilter3D default

calib = load_calibration("data/calib/P01/calibration.toml", target_size=(1920, 1080))
d = json.load(open(f"experiments/drink_study/cache/P01_{REP}__teacher__c0.25.json"))
dets = {c: [tuple(x) if x else None for x in v] for c, v in d.items()}
cams = sorted(dets, key=lambda k: int(k.split("_")[1]))
n = min(len(v) for v in dets.values())


def gated(obs, thr=30, minc=3):
    cur = dict(obs)
    while len(cur) >= 2:
        X = triangulate_dlt([calib[c] for c in cur], [np.array(cur[c]) for c in cur])
        e = {c: float(np.hypot(*(project(calib[c], X)[0] - np.array(cur[c])))) for c in cur}
        w = max(e, key=e.get)
        if e[w] <= thr:
            break
        del cur[w]
    return (triangulate_dlt([calib[c] for c in cur], [np.array(cur[c]) for c in cur])
            if len(cur) >= minc else None)


ref = [gated({c: dets[c][fr] for c in cams if dets[c][fr] is not None}) for fr in range(n)]

# ---- forward EKF, storing terms needed for RTS ----
kf = KalmanFilter3D(process_noise=Q_ACCEL)
xf, Pf = [None]*n, [None]*n        # filtered (posterior) state/cov
xp, Pp = [None]*n, [None]*n        # predicted (prior) state/cov
Fs = [None]*n                      # transition used at each step
nused = [0]*n
t_prev = None
for fr in range(n):
    t = fr / FPS
    obs = {c: dets[c][fr] for c in cams if dets[c][fr] is not None}
    if not kf.initialized:
        if ref[fr] is not None:
            kf.init(ref[fr], t)
            xf[fr], Pf[fr] = kf.x.copy(), kf.P.copy()
            xp[fr], Pp[fr] = kf.x.copy(), kf.P.copy()
            Fs[fr] = np.eye(6); t_prev = t
        continue
    dt = t - t_prev
    F = np.eye(6); F[:3, 3:] = dt * np.eye(3)
    kf.predict(t)
    xp[fr], Pp[fr], Fs[fr] = kf.x.copy(), kf.P.copy(), F
    u = 0
    for c in obs:
        acc, _ = kf.update(calib[c], np.array(obs[c]), t)
        u += int(acc)
    xf[fr], Pf[fr], nused[fr] = kf.x.copy(), kf.P.copy(), u
    t_prev = t

# ---- backward RTS pass ----
xs = [x.copy() if x is not None else None for x in xf]
Ps = [P.copy() if P is not None else None for P in Pf]
idx = [fr for fr in range(n) if xf[fr] is not None]
for k in range(len(idx) - 2, -1, -1):
    fr, frn = idx[k], idx[k + 1]
    F = Fs[frn]
    C = Pf[fr] @ F.T @ np.linalg.inv(Pp[frn])
    xs[fr] = xf[fr] + C @ (xs[frn] - xp[frn])
    Ps[fr] = Pf[fr] + C @ (Ps[frn] - Pp[frn]) @ C.T

# ---- compare + render ----
both = [(xf[fr][:3], xs[fr][:3]) for fr in range(n) if xf[fr] is not None]
dist = np.array([np.linalg.norm(a - b) for a, b in both])
print(f"frames tracked: {len(both)}/{n}")
print(f"RTS vs forward-only position shift: median {np.median(dist):.0f}mm  p90 {np.percentile(dist,90):.0f}mm  max {dist.max():.0f}mm")

# velocity smoothness: jerk (change in velocity) magnitude, forward vs smoothed
def jerk(states):
    v = [s[3:] for s in states if s is not None]
    j = [np.linalg.norm(v[i] - v[i-1]) for i in range(1, len(v))]
    return np.median(j), np.percentile(j, 90)
jf = jerk([xf[fr] for fr in range(n)])
js = jerk([xs[fr] for fr in range(n)])
print(f"velocity-change (smoothness)  forward: median {jf[0]:.0f} p90 {jf[1]:.0f}  |  RTS: median {js[0]:.0f} p90 {js[1]:.0f}  (lower=smoother)")

OUT = "experiments/drink_study/debug_cam10/p01_cam10_rts.mp4"
cap = cv2.VideoCapture(f"/home/imove/Documents/clips/P01/{REP}.10.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 30
W = int(cap.get(3)); H = int(cap.get(4))
vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
fr = 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    gd = dets["cam_10"][fr] if fr < len(dets["cam_10"]) else None
    if gd is not None:
        cv2.circle(f, (int(gd[0]), int(gd[1])), 11, (0, 0, 255), 2)
    if fr < n and xf[fr] is not None:
        uv, infr = project(calib["cam_10"], xf[fr][:3])
        if infr:
            cv2.drawMarker(f, (int(uv[0]), int(uv[1])), (0, 200, 255), cv2.MARKER_CROSS, 30, 3)  # forward = orange
    if fr < n and xs[fr] is not None:
        uv, infr = project(calib["cam_10"], xs[fr][:3])
        if infr:
            cv2.drawMarker(f, (int(uv[0]), int(uv[1])), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 34, 3)  # RTS = green
    cv2.putText(f, f"P01 cam_10 fr{fr}  GREEN X=RTS(future-aware)  ORANGE +=forward-only  o=teacher glass",
                (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    vw.write(f); fr += 1
cap.release(); vw.release()
print(f"wrote {OUT}")
