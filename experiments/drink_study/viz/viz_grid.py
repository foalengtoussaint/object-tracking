"""Multi-camera grid overlay video of a student's cup detections on a held-out clip.

Runs the model on each chosen camera of one held-out rep, draws each detection
(box + conf), tiles the cameras into a grid, writes an mp4. Defaults to the refill
(reject-then-fill) student on P06 cams 1/3/4/8/10 -- cam_10 included since that's
the camera the refill filter fixed.

    python experiments/drink_study/viz_grid.py
    python experiments/drink_study/viz_grid.py --weights <pt> --participant P06 --cams 1 3 4 8 10
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cv2
import numpy as np
import run
from _paths import CLIPS_ROOT

DEF_W = "experiments/drink_study/runs/pscale_1_clean3d_refill/weights/best_3df1.pt"
CONF = 0.25
TILE = (480, 270)            # per-cam tile size (w,h), 16:9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEF_W)
    ap.add_argument("--participant", default="P06")
    ap.add_argument("--cams", type=int, nargs="+", default=[1, 3, 4, 8, 10])
    ap.add_argument("--out", default="experiments/drink_study/cache/refill_grid.mp4")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    stem = run.reps_of(args.participant, "right")[0]
    caps = {}
    for c in args.cams:
        f = CLIPS_ROOT / args.participant / f"{stem}.{c}.mp4"
        if f.exists():
            caps[c] = cv2.VideoCapture(str(f))
    cams = list(caps)
    n = min(int(caps[c].get(cv2.CAP_PROP_FRAME_COUNT)) for c in cams)

    # grid layout: up to 3 cols
    cols = min(3, len(cams)); rows = (len(cams) + cols - 1) // cols
    tw, th = TILE
    gw, gh = cols * tw, rows * th
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (gw, gh))
    print(f"rendering {n} frames, {len(cams)} cams -> {args.out}", flush=True)

    for fr in range(n):
        canvas = np.zeros((gh, gw, 3), np.uint8)
        for i, c in enumerate(cams):
            ok, img = caps[c].read()
            if not ok:
                continue
            r = model(img, conf=CONF, verbose=False)[0]
            nd = 0
            if r.boxes is not None and len(r.boxes):
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    cf = float(b.conf[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
                    cv2.putText(img, f"{cf:.2f}", (x1, max(0, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    nd += 1
            tile = cv2.resize(img, (tw, th))
            label = f"cam_{c}" + ("  DET" if nd else "")
            col = (0, 255, 0) if nd else (0, 0, 255)
            cv2.putText(tile, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            rr, cc = divmod(i, cols)
            canvas[rr * th:(rr + 1) * th, cc * tw:(cc + 1) * tw] = tile
        cv2.putText(canvas, f"frame {fr}/{n}", (8, gh - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        vw.write(canvas)
        if (fr + 1) % 100 == 0:
            print(f"  {fr+1}/{n}", flush=True)
    vw.release()
    for c in caps.values():
        c.release()
    print(f"DONE wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
