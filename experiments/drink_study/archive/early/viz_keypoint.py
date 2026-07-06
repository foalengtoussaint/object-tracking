"""Render the keypoint (pose) student's predicted cup centroid per camera on a clip,
with the gated >=3-cam consensus, so we can EYEBALL whether the points land on the
cup or on the body (the overfitting question). Verbose per-cam progress."""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cv2, numpy as np
import run
from _paths import CLIPS_ROOT
from kalman_3d import load_calibration, triangulate_dlt, project
from agreement import RES

CONF = 0.25; THR = 30.0; MINC = 3; TW, TH = 480, 270
W = "experiments/drink_study/runs/pscale_1_keypoint/weights/best.pt"


def kp(r):
    if r.keypoints is None or r.keypoints.xy is None or r.keypoints.xy.shape[0] == 0:
        return None
    i = int(r.boxes.conf.argmax()) if (r.boxes is not None and len(r.boxes)) else 0
    p = r.keypoints.xy[i, 0].tolist()
    return None if (p[0] == 0 and p[1] == 0) else (float(p[0]), float(p[1]))


def gated(obs, calib):
    cur = dict(obs)
    while len(cur) >= 2:
        X = triangulate_dlt([calib[c] for c in cur], [np.array(cur[c]) for c in cur])
        e = {c: float(np.hypot(*(project(calib[c], X)[0] - np.array(cur[c])))) for c in cur}
        w = max(e, key=e.get)
        if e[w] <= THR:
            break
        del cur[w]
    return (set(cur), X) if len(cur) >= MINC else (set(), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", default="P23")
    ap.add_argument("--out", default="experiments/drink_study/cache/keypoint_P23.mp4")
    ap.add_argument("--fps", type=float, default=30.0)
    a = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(W)
    calib = load_calibration(f"data/calib/{a.participant}/calibration.toml", target_size=RES)
    stem = run.reps_of(a.participant, "right")[0]
    cams = [c for c in range(1, 11) if f"cam_{c}" in calib
            and (CLIPS_ROOT / a.participant / f"{stem}.{c}.mp4").exists()]
    caps = {c: cv2.VideoCapture(str(CLIPS_ROOT / a.participant / f"{stem}.{c}.mp4")) for c in cams}
    n = min(int(caps[c].get(cv2.CAP_PROP_FRAME_COUNT)) for c in cams)
    cols = min(5, len(cams)); rows = (len(cams) + cols - 1) // cols
    BAN = 46; gw, gh = cols * TW, rows * TH + BAN
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (gw, gh))
    print(f"rendering {n} frames x {len(cams)} cams (keypoint) -> {a.out}", flush=True)
    for fr in range(n):
        frames, pts = {}, {}
        for c in cams:
            ok, img = caps[c].read()
            img = img if ok else np.zeros((RES[1], RES[0], 3), np.uint8)
            p = kp(model(img, conf=CONF, verbose=False)[0])
            if p is not None:
                pts[f"cam_{c}"] = p
            frames[c] = (img, p)
        kept, X = gated(dict(pts), calib)
        canvas = np.zeros((gh, gw, 3), np.uint8)
        ok_con = X is not None
        cv2.rectangle(canvas, (0, 0), (gw, BAN), (0, 120, 0) if ok_con else (0, 0, 160), -1)
        cv2.putText(canvas, (f"CONSENSUS {len(kept)} cams" if ok_con else "NO CONSENSUS")
                    + f"  frame {fr}", (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        for i, c in enumerate(cams):
            img, p = frames[c]; ck = f"cam_{c}"
            if p is not None:
                col = (0, 255, 0) if ck in kept else (0, 165, 255)  # green inlier / orange ejected
                cv2.circle(img, (int(p[0]), int(p[1])), 14, col, -1)
                cv2.circle(img, (int(p[0]), int(p[1])), 14, (0, 0, 0), 2)
            tile = cv2.resize(img, (TW, TH))
            lab = "kp" if p is not None else "none"
            cv2.putText(tile, f"cam_{c}: {lab}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if p is not None else (0, 0, 255), 2)
            rr, cc = divmod(i, cols)
            canvas[BAN + rr*TH:BAN+(rr+1)*TH, cc*TW:(cc+1)*TW] = tile
        vw.write(canvas)
        if (fr + 1) % 100 == 0:
            print(f"  {fr+1}/{n}", flush=True)
    vw.release()
    for c in caps.values():
        c.release()
    print(f"DONE wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
