"""Fine-tune a YOLO-seg student on a single-class dataset built by
pseudo_label.py.

Usage:
    python finetune.py                                            # defaults: yolo26n, 50 epochs, gen1
    python finetune.py --data data/datasets/gen2/data.yaml --name cup_gen2 --epochs 10
    python finetune.py --weights data/pretrained/yolo26s-seg.pt --batch 8

Outputs land in data/runs/segment/<name>/. Next step:
    python evaluate.py --weights data/runs/segment/<name>/weights/best.pt --clips data/clips/test
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/datasets/gen1/data.yaml"))
    ap.add_argument("--weights", default="data/pretrained/yolo26n-seg.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="my_cup")
    ap.add_argument("--project", default="data/runs/segment",
                    help="Where ultralytics writes <name>/weights/best.pt etc.")
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
        project=args.project,
        name=args.name,
    )
    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\ndone. best weights: {best}")
    print(f"evaluate:  python evaluate.py --weights {best} --clips data/clips/test")


if __name__ == "__main__":
    main()
