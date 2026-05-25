"""Fine-tune a YOLO-seg model on the single-class cup dataset built by
pseudo_label.py.

Usage:
    python finetune.py                          # yolo11n-seg, 50 epochs
    python finetune.py --weights yolo11s-seg.pt --epochs 80
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("dataset/data.yaml"))
    ap.add_argument("--weights", default="yolo26n-seg.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="my_cup")
    args = ap.parse_args()

    if not args.data.exists():
        raise SystemExit(
            f"data yaml not found: {args.data}\n"
            f"run record_clips.py + pseudo_label.py first."
        )

    model = YOLO(args.weights)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
    )
    best = Path("runs/segment") / args.name / "weights" / "best.pt"
    print(f"\ndone. best weights: {best}")
    print(f"run live:  python live_cup_detect.py --weights {best}")


if __name__ == "__main__":
    main()
