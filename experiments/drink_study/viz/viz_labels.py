"""Audit the TRAINING labels at the cup-at-mouth phase: overlay the refill filter's
labels on the actual P01 frames, distinguishing real detections from synthesized
fills, so you can see if any label drifts onto the face/mouth instead of the cup.

Per camera, per frame (reconstructs exactly what run_clean3d_fill would write):
  - run gated >=3-cam consensus from the cached teacher centroids
  - GREEN box   = real detection kept (in consensus inlier set)
  - YELLOW box  = SYNTHESIZED fill (cam didn't detect OR was ejected) -> cup
                  reprojected from consensus, sized by 35mm sphere (the current rule)
  - RED label   = no consensus this frame (no label emitted)
Box size uses the same apparent-radius logic as the trainer.

    python experiments/drink_study/viz_labels.py
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import argparse, sys, glob, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cv2
import numpy as np
import run
from _paths import CLIPS_ROOT
from kalman_3d import load_calibration, triangulate_dlt, project
from agreement import RES
from kf_accuracy import CUP_R

CONF = 0.25; THR = 30.0; MINC = 3
TW, TH = 480, 270


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
        return None, set()
    return triangulate_dlt([calib[c] for c in cur], [np.array(cur[c]) for c in cur]), set(cur)


def main():
    ap = argparse.ArgumentParser()
    # cams chosen to include face-seeing views; cam_10 = distant glass cam
    ap.add_argument("--cams", type=int, nargs="+", default=[2, 4, 7, 9, 10])
    ap.add_argument("--out", default="experiments/drink_study/cache/labels_mouth_P01.mp4")
    ap.add_argument("--fps", type=float, default=20.0)
    args = ap.parse_args()
    calib = load_calibration("data/calib/P01/calibration.toml", target_size=RES)
    tf = sorted(glob.glob("experiments/drink_study/cache/P01_*teacher__c0.25.json"))[0]
    # cache file is "P01_<stem>__teacher__c0.25.json" where <stem> ITSELF starts with
    # "P01_" -> strip only the leading cache prefix + the teacher suffix.
    stem = Path(tf).stem[len("P01_"):].replace("__teacher__c0.25", "")
    d = json.loads(Path(tf).read_text())
    dets = {c: [tuple(x) if x else None for x in v] for c, v in d.items() if c in calib}
    cam_keys = sorted(dets, key=lambda k: int(k.split("_")[1]))
    n = min(len(v) for v in dets.values())

    cams = [c for c in args.cams if f"cam_{c}" in calib]
    caps = {c: cv2.VideoCapture(str(CLIPS_ROOT / "P01" / f"{stem}.{c}.mp4")) for c in cams}
    cols = min(5, len(cams)); rows = (len(cams) + cols - 1) // cols
    gw, gh = cols * TW, rows * TH + 40
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (gw, gh))
    print(f"rendering {n} frames, label audit cams {cams} -> {args.out}", flush=True)

    def app_r(cam, X):
        c0 = project(calib[cam], X)[0]
        Xo = X.copy(); Xo[0] += CUP_R
        return c0, max(float(np.hypot(*(project(calib[cam], Xo)[0] - c0))), 6.0)

    for fr in range(n):
        obs = {c: dets[c][fr] for c in cam_keys if dets[c][fr] is not None}
        X, kept = gated(obs, calib)
        canvas = np.zeros((gh, gw, 3), np.uint8)
        has = X is not None
        cv2.rectangle(canvas, (0, 0), (gw, 40), (0, 100, 0) if has else (0, 0, 150), -1)
        cv2.putText(canvas, ("LABELS (green=real, yellow=fill)  " if has else "no consensus -> no label  ")
                    + f"frame {fr}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        for i, c in enumerate(cams):
            ok, img = caps[c].read()
            if not ok:
                img = np.zeros((RES[1], RES[0], 3), np.uint8)
            ckey = f"cam_{c}"
            if has:
                c0, r = app_r(ckey, X)
                if ckey in kept:                          # real kept detection
                    z = dets[ckey][fr]
                    p = (int(z[0]), int(z[1]))
                    cv2.rectangle(img, (p[0]-int(r), p[1]-int(r)), (p[0]+int(r), p[1]+int(r)), (0, 255, 0), 5)
                    lab, col = "real", (0, 255, 0)
                else:                                     # synthesized fill (no det or ejected)
                    p = (int(c0[0]), int(c0[1]))
                    cv2.rectangle(img, (p[0]-int(r), p[1]-int(r)), (p[0]+int(r), p[1]+int(r)), (0, 255, 255), 5)
                    lab, col = ("fill (ejected)" if dets[ckey][fr] is not None else "fill"), (0, 255, 255)
            else:
                lab, col = "no label", (0, 0, 255)
            tile = cv2.resize(img, (TW, TH))
            cv2.putText(tile, f"cam_{c}: {lab}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            rr, cc = divmod(i, cols)
            canvas[40 + rr*TH:40+(rr+1)*TH, cc*TW:(cc+1)*TW] = tile
        vw.write(canvas)
        if (fr + 1) % 100 == 0:
            print(f"  {fr+1}/{n}", flush=True)
    vw.release()
    for c in caps.values():
        c.release()
    print(f"DONE wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
