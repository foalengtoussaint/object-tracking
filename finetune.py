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


def train_student(data: Path, weights: str = "data/pretrained/yolo26n-seg.pt",
                  epochs: int = 50, imgsz: int = 640, batch: int = 16,
                  name: str = "my_cup",
                  project: str = "data/runs/segment") -> Path:
    """Fine-tune a YOLO-seg student and return the path to best.pt.

    Pass an absolute `project` to avoid ultralytics nesting the run under its
    own runs_dir. Reused by finetune's CLI and pipeline.py's loss_plateau_train.
    """
    data = Path(data)
    if not data.exists():
        raise SystemExit(
            f"data yaml not found: {data}\n"
            f"run record_clips.py + pseudo_label.py first."
        )
    model = YOLO(weights)
    model.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
    )
    return Path(project) / name / "weights" / "best.pt"


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

    best = train_student(args.data, args.weights, args.epochs, args.imgsz,
                         args.batch, args.name, args.project)
    print(f"\ndone. best weights: {best}")
    print(f"evaluate:  python evaluate.py --weights {best} --clips data/clips/test")


if __name__ == "__main__":
    main()
