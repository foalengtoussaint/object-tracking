"""cam_10 side-by-side: reject (0.51) vs refill (0.74) student detections on one
held-out clip. Left = reject-only, right = reject-then-fill. Shows the cam_10
recovery directly: refill should box the real cup where reject misses / mis-fires.

    python experiments/drink_study/viz_sidebyside.py
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

REJECT = "experiments/drink_study/runs/pscale_1_clean3d/weights/best.pt"
REFILL = "experiments/drink_study/runs/pscale_1_clean3d_refill/weights/best_3df1.pt"
CONF = 0.25
TW, TH = 640, 360


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", default="P06")
    ap.add_argument("--cam", type=int, default=10)
    ap.add_argument("--out", default="experiments/drink_study/cache/cam10_reject_vs_refill.mp4")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()
    from ultralytics import YOLO
    mr, mf = YOLO(REJECT), YOLO(REFILL)
    stem = run.reps_of(args.participant, "right")[0]
    clip = CLIPS_ROOT / args.participant / f"{stem}.{args.cam}.mp4"
    cap = cv2.VideoCapture(str(clip))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (2 * TW, TH))
    print(f"rendering {n} frames cam_{args.cam} -> {args.out}", flush=True)

    def draw(img, model, title, color):
        r = model(img, conf=CONF, verbose=False)[0]
        nd = 0
        if r.boxes is not None and len(r.boxes):
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
                cv2.putText(img, f"{float(b.conf[0]):.2f}", (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                nd += 1
        t = cv2.resize(img, (TW, TH))
        cv2.putText(t, f"{title}{'  DET' if nd else '  (none)'}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color if nd else (0, 0, 255), 2)
        return t

    for fr in range(n):
        ok, img = cap.read()
        if not ok:
            break
        left = draw(img.copy(), mr, "REJECT (cam10=0.51)", (0, 200, 255))
        right = draw(img.copy(), mf, "REFILL (cam10=0.74)", (0, 255, 0))
        vw.write(np.hstack([left, right]))
        if (fr + 1) % 100 == 0:
            print(f"  {fr+1}/{n}", flush=True)
    vw.release(); cap.release()
    print(f"DONE wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
