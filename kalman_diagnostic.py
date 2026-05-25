"""Re-run YOLO detection + Kalman filter on a clip and log per-frame state:
position prediction, position sigma (1-D), Mahalanobis to best candidate,
acceptance. Caches detection results to .candidates.pkl for fast re-runs.

Usage:
    python kalman_diagnostic.py clips/foo.mp4                    # full table
    python kalman_diagnostic.py clips/foo.mp4 --frames 280-345   # subset
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from pseudo_label import CUP_LIKE_CLASSES, KalmanFilter2D


def detection_pass(clip: Path, weights: str, conf: float) -> dict:
    cache = clip.with_suffix(".candidates.pkl")
    if cache.exists():
        print(f"using cached detections: {cache.name}")
        with open(cache, "rb") as f:
            return pickle.load(f)
    print(f"running {weights} on {clip.name}...")
    model = YOLO(weights)
    cands: dict[int, list] = {}
    n = 0
    for frame_idx, result in enumerate(model.predict(
            source=str(clip), classes=CUP_LIKE_CLASSES, conf=conf,
            stream=True, verbose=False)):
        n = frame_idx + 1
        if (result.boxes is None or len(result.boxes) == 0
                or result.masks is None):
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        cands[frame_idx] = [(centers[i], boxes[i], float(confs[i]))
                            for i in range(len(boxes))]
    out = {"candidates": cands, "n_frames": n}
    with open(cache, "wb") as f:
        pickle.dump(out, f)
    print(f"cached → {cache}")
    return out


def parse_frames(spec: str | None, n_frames: int) -> range:
    if not spec:
        return range(n_frames)
    a, _, b = spec.partition("-")
    return range(int(a), int(b) + 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", type=Path)
    ap.add_argument("--weights", default="yolo11x-seg.pt")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--gate", type=float, default=13.82)
    ap.add_argument("--max-miss", type=int, default=30)
    ap.add_argument("--frames", default=None,
                    help="Frame range to print, e.g. 280-345")
    args = ap.parse_args()

    det = detection_pass(args.clip, args.weights, args.conf)
    candidates = det["candidates"]
    n_total = det["n_frames"]

    # Run Kalman, recording state at each frame
    kf = KalmanFilter2D()
    initialized = False
    misses = 0
    rows = []
    for frame_idx in range(n_total):
        cands = candidates.get(frame_idx, [])
        n_cand = len(cands)

        if not initialized:
            if cands:
                i = int(np.argmax([c[2] for c in cands]))
                kf.init(cands[i][0])
                rows.append(dict(
                    frame=frame_idx, n_cand=n_cand, accepted=True,
                    pred_x=cands[i][0][0], pred_y=cands[i][0][1],
                    sigma_x=float(np.sqrt(kf.P[0, 0])),
                    sigma_y=float(np.sqrt(kf.P[1, 1])),
                    mahal=0.0, misses=0, status="INIT",
                ))
                initialized = True
                misses = 0
            else:
                rows.append(dict(
                    frame=frame_idx, n_cand=0, accepted=False,
                    pred_x=None, pred_y=None,
                    sigma_x=None, sigma_y=None, mahal=None,
                    misses=0, status="WAITING",
                ))
            continue

        kf.predict()
        pred_x, pred_y = kf.x[0], kf.x[1]
        sx = float(np.sqrt(kf.P[0, 0]))
        sy = float(np.sqrt(kf.P[1, 1]))

        if not cands:
            misses += 1
            rows.append(dict(
                frame=frame_idx, n_cand=0, accepted=False,
                pred_x=pred_x, pred_y=pred_y,
                sigma_x=sx, sigma_y=sy, mahal=None,
                misses=misses, status="MISS",
            ))
            continue

        ds = [kf.mahalanobis(c[0]) for c in cands]
        i = int(np.argmin(ds))
        d_best = ds[i]

        if d_best <= args.gate and misses <= args.max_miss:
            kf.update(cands[i][0])
            status = "ACCEPT"
            accepted = True
            misses = 0
        elif d_best <= args.gate and misses > args.max_miss:
            kf.init(cands[i][0])
            status = "REINIT"
            accepted = True
            misses = 0
        else:
            status = "REJECT"
            accepted = False
            misses += 1

        rows.append(dict(
            frame=frame_idx, n_cand=n_cand, accepted=accepted,
            pred_x=pred_x, pred_y=pred_y,
            sigma_x=sx, sigma_y=sy, mahal=d_best,
            misses=misses, status=status,
        ))

    # Print table for requested frames
    frange = parse_frames(args.frames, n_total)
    print(f"\n{'frame':>5}  {'#cd':>3}  {'pred_x':>7}  {'pred_y':>7}  "
          f"{'sig_x':>6}  {'sig_y':>6}  {'mahal':>6}  {'miss':>4}  status")
    print("-" * 78)
    for r in rows:
        if r["frame"] not in frange:
            continue
        px = f"{r['pred_x']:7.1f}" if r['pred_x'] is not None else "    -  "
        py = f"{r['pred_y']:7.1f}" if r['pred_y'] is not None else "    -  "
        sx = f"{r['sigma_x']:6.2f}" if r['sigma_x'] is not None else "    -"
        sy = f"{r['sigma_y']:6.2f}" if r['sigma_y'] is not None else "    -"
        mh = f"{r['mahal']:6.2f}" if r['mahal'] is not None else "    -"
        print(f"{r['frame']:5d}  {r['n_cand']:3d}  {px}  {py}  "
              f"{sx}  {sy}  {mh}  {r['misses']:4d}  {r['status']}")


if __name__ == "__main__":
    main()
