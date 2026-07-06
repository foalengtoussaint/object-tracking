"""Test whether 3D cross-camera inlier-gating rejects the teacher's cam_10
static-glass false positives in P01 training data (now that P01 is calibrated).

Runs the COCO teacher on one P01 drinking rep's 10 cameras, triangulates the cup
from the agreeing cameras (inlier-gated at thr px), and reports — per camera —
how many of its detections are 3D-inliers vs rejected. cam_10's static-glass
detections should be REJECTED (they sit at a 3D point the other 9 cams disagree
with), while real-cup detections are kept.
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from kalman_3d import load_calibration, triangulate_dlt, project
from agreement import detect_rep, RES
from pseudo_label import CUP_LIKE_CLASSES
from ultralytics import YOLO

REP = "P01_drinking_right_20231220_141546"
CLIPS = Path("/home/imove/Documents/clips/P01")
THR = 30.0
CACHE = Path("experiments/drink_study/cache")


def cached_detect(rep):
    """Run the teacher once, cache raw dets so re-runs are instant (no re-inference)."""
    cf = CACHE / f"P01_{REP}__teacher__c0.25.json"
    if cf.exists():
        print(f"using cached teacher dets: {cf}", flush=True)
        d = json.loads(cf.read_text())
        return {c: [tuple(x) if x else None for x in v] for c, v in d.items()}
    print("running teacher on all cameras (will cache) ...", flush=True)
    model = YOLO("data/pretrained/yolo26x-seg.pt")
    dets = detect_rep(model, rep, conf=0.25, classes=CUP_LIKE_CLASSES, verbose=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    cf.write_text(json.dumps(dets))
    print(f"cached -> {cf}", flush=True)
    return dets


def inliers(obs, thr=THR, mc=2):
    cur = dict(obs)
    for _ in range(8):
        if len(cur) < 2:
            return set(cur)
        X = triangulate_dlt([calib[c] for c in cur], [np.array(cur[c]) for c in cur])
        e = {c: float(np.hypot(*(project(calib[c], X)[0] - np.array(cur[c])))) for c in cur}
        bad = [c for c in cur if e[c] > thr]
        if not bad or len(cur) - 1 < mc:
            break
        del cur[max(bad, key=lambda c: e[c])]
    return set(cur)


calib = load_calibration("data/calib/P01/calibration.toml", target_size=RES)
rep = {f"cam_{c}": CLIPS / f"{REP}.{c}.mp4" for c in range(1, 11)
       if f"cam_{c}" in calib and (CLIPS / f"{REP}.{c}.mp4").exists()}
print(f"rep cams: {sorted(rep, key=lambda k: int(k.split('_')[1]))}", flush=True)

dets = cached_detect(rep)
cams = sorted(dets, key=lambda k: int(k.split("_")[1]))
n = min(len(v) for v in dets.values())

detected = {c: 0 for c in cams}
kept = {c: 0 for c in cams}
rej = {c: 0 for c in cams}
for fr in range(n):
    obs = {c: dets[c][fr] for c in cams if dets[c][fr] is not None}
    if len(obs) < 3:
        continue
    inl = inliers(obs)
    for c in obs:
        detected[c] += 1
        (kept if c in inl else rej)[c] += 1

# also: cam_10 detection location spread (static glass => std ~0)
c10 = [dets["cam_10"][fr] for fr in range(n) if dets["cam_10"][fr] is not None]
print(f"\nteacher P01 cam_10: {len(c10)} dets", flush=True)
if c10:
    xs = np.array([p[0] for p in c10]); ys = np.array([p[1] for p in c10])
    print(f"  center x med={np.median(xs):.0f}({np.median(xs)/1920:.2f}) std={xs.std():.0f} | "
          f"y med={np.median(ys):.0f}({np.median(ys)/1080:.2f}) std={ys.std():.0f}")

print(f"\n3D inlier-gating @ {THR}px (need >=3 cams/frame):")
print(f"{'cam':>7} {'detected':>9} {'kept':>6} {'REJECTED':>9} {'%rej':>6}")
for c in cams:
    d = detected[c]; pct = 100 * rej[c] / d if d else 0
    flag = " <== cam_10" if c == "cam_10" else ""
    print(f"{c:>7} {d:>9} {kept[c]:>6} {rej[c]:>9} {pct:>5.0f}%{flag}")
