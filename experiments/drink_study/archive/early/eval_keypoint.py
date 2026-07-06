"""Eval the keypoint (pose) student the SAME way as the seg students, but reading
the predicted KEYPOINT (cup centroid) instead of a box. Reports per-cam detection
rate + gated 3D precision (tri_rate / median_px) on a participant. Verbose."""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import cv2
import run
from _paths import CLIPS_ROOT
from kalman_3d import load_calibration, triangulate_dlt, project
from agreement import RES
from pipeline_lib import camera_id

CONF = 0.25; THR = 30.0; MINC = 3


def kp_point(r):
    """Top cup keypoint (cx,cy) from a pose result, or None."""
    if r.keypoints is None or len(r.keypoints) == 0:
        return None
    # pick the instance with highest box conf if available
    kxy = r.keypoints.xy  # (n,1,2)
    if kxy is None or kxy.shape[0] == 0:
        return None
    i = 0
    if r.boxes is not None and len(r.boxes):
        i = int(r.boxes.conf.argmax())
    p = kxy[i, 0].tolist()
    if p[0] == 0 and p[1] == 0:
        return None
    return (float(p[0]), float(p[1]))


def gated(obs, calib):
    cur = dict(obs)
    while len(cur) >= 2:
        X = triangulate_dlt([calib[c] for c in cur], [np.array(cur[c]) for c in cur])
        e = {c: float(np.hypot(*(project(calib[c], X)[0] - np.array(cur[c])))) for c in cur}
        w = max(e, key=e.get)
        if e[w] <= THR:
            break
        del cur[w]
    if len(cur) < MINC:
        return None, set(), None
    X = triangulate_dlt([calib[c] for c in cur], [np.array(cur[c]) for c in cur])
    px = float(np.median([np.hypot(*(project(calib[c], X)[0] - np.array(cur[c]))) for c in cur]))
    return X, set(cur), px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="experiments/drink_study/runs/pscale_1_keypoint/weights/best.pt")
    ap.add_argument("--participant", default="P23")
    args = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(args.weights)
    calib = load_calibration(f"data/calib/{args.participant}/calibration.toml", target_size=RES)
    stem = run.reps_of(args.participant, "right")[0]
    cams = [c for c in range(1, 11) if f"cam_{c}" in calib
            and (CLIPS_ROOT / args.participant / f"{stem}.{c}.mp4").exists()]
    print(f"eval keypoint student on {args.participant} {stem}, {len(cams)} cams", flush=True)

    dets = {f"cam_{c}": [] for c in cams}
    by = {c: [0, 0] for c in cams}
    for c in cams:
        cap = cv2.VideoCapture(str(CLIPS_ROOT / args.participant / f"{stem}.{c}.mp4"))
        seq = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            r = model(f, conf=CONF, verbose=False)[0]
            p = kp_point(r)
            seq.append(p); by[c][0] += (p is not None); by[c][1] += 1
        cap.release(); dets[f"cam_{c}"] = seq
        print(f"  cam_{c}: {by[c][0]}/{by[c][1]} kp", flush=True)

    det_rate = {f"cam{c}": by[c][0] / by[c][1] for c in cams}
    n = min(len(v) for v in dets.values())
    tri = 0; pxs = []
    for fr in range(n):
        obs = {ck: dets[ck][fr] for ck in dets if dets[ck][fr] is not None}
        if len(obs) < 2:
            continue
        X, kept, px = gated(obs, calib)
        if X is not None:
            tri += 1; pxs.append(px)
    cams_k = sorted(det_rate, key=lambda k: int(k.replace("cam", "")))
    print(f"\n=== KEYPOINT student on {args.participant} ===", flush=True)
    print(f"  det-rate mean = {np.mean([det_rate[c] for c in cams_k]):.3f}", flush=True)
    print(f"  cam10         = {det_rate.get('cam10', 0):.2f}", flush=True)
    print(f"  tri_rate      = {tri / n:.3f}", flush=True)
    print(f"  median_px     = {np.median(pxs):.2f}" if pxs else "  median_px = n/a", flush=True)
    print("KP_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
