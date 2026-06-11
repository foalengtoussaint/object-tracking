"""Use a teacher YOLO-seg model to auto-label a directory of MP4 clips,
filter out bad/duplicate predictions via a single-object Kalman filter,
and write a YOLO single-class segmentation dataset.

Output layout:
    <out>/images/train/<clip>_f<idx>.jpg
    <out>/labels/train/<clip>_f<idx>.txt   ("0 x1 y1 x2 y2 ...")
    <out>/data.yaml

Only frames where the Kalman filter accepted a detection get written —
no-detection and gate-rejected frames are dropped entirely.

Examples
--------
    # COCO teacher labeling a fresh cup recording
    python pseudo_label.py data/clips/train --out data/datasets/gen1 \\
        --weights data/pretrained/yolo26x-seg.pt

    # Single-class fine-tuned teacher (e.g. self-distillation)
    python pseudo_label.py data/clips/train --out data/datasets/gen2 \\
        --weights data/runs/segment/cup_5cam_demo/weights/best.pt \\
        --conf 0.25 --classes 0
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

from kalman import GATE, MAX_MISS, filter_detections


# Default class filter: COCO bottle, wine glass, cup, bowl, vase.
CUP_LIKE_CLASSES = [39, 40, 41, 45, 75]


def label_clip(model: YOLO, clip: Path, conf: float, classes: list[int] | None,
               gate: float, max_miss: int, use_kf: bool = True,
               ) -> tuple[dict[int, "numpy.ndarray"], dict[str, int]]:
    """Run teacher inference + KF filter on one clip.

    Returns:
        accepted  {frame_idx: polygon}  KF-accepted detections, one per frame
        stats     totals for logging

    use_kf=False is the ablation baseline: keep each frame's top-confidence
    detection (no spatial/temporal gating) instead of running the Kalman filter.
    """
    # Pass 1: collect every candidate detection per frame, with its polygon.
    per_frame: list[list[dict]] = []
    polys_per_frame: list[list] = []
    for result in model.predict(source=str(clip), classes=classes, conf=conf,
                                stream=True, verbose=False):
        if (result.boxes is None or len(result.boxes) == 0
                or result.masks is None):
            per_frame.append([])
            polys_per_frame.append([])
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        polys = list(result.masks.xyn)
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        per_frame.append([
            {"cx": float(c[0]), "cy": float(c[1]), "conf": float(cf), "_idx": i}
            for i, (c, cf) in enumerate(zip(centers, confs))
        ])
        polys_per_frame.append(polys)

    accepted: dict[int, "numpy.ndarray"] = {}
    if use_kf:
        # Pass 2: single-object Kalman filter decides which (if any) det per frame.
        filtered = filter_detections(per_frame, gate=gate, max_miss=max_miss)
        for idx, ff in enumerate(filtered):
            if ff.accepted is not None:
                accepted[idx] = polys_per_frame[idx][ff.accepted["_idx"]]
        stats = {
            "total_frames": len(filtered),
            "frames_with_detections": sum(1 for p in per_frame if p),
            "accepted": len(accepted),
            "rejected": sum(1 for ff in filtered if ff.role == "rejected"),
            "predicted": sum(1 for ff in filtered if ff.role == "predicted"),
        }
    else:
        # Ablation: keep the top-confidence detection per frame, no gating.
        for idx, dets in enumerate(per_frame):
            if dets:
                top = max(dets, key=lambda d: d["conf"])
                accepted[idx] = polys_per_frame[idx][top["_idx"]]
        stats = {
            "total_frames": len(per_frame),
            "frames_with_detections": sum(1 for p in per_frame if p),
            "accepted": len(accepted),
            "rejected": 0,
            "predicted": 0,
        }
    return accepted, stats


def resolve_classes(spec: str) -> list[int] | None:
    """Map a --classes string ('all' | 'cup-like' | '0,1,2') to a class list."""
    if spec == "all":
        return None
    if spec == "cup-like":
        return CUP_LIKE_CLASSES
    return [int(x) for x in spec.split(",")]


def build_dataset(clips_dir: Path, out: Path, weights: str, conf: float,
                  classes: list[int] | None, gate: float = GATE,
                  max_miss: int = MAX_MISS, class_name: str = "object",
                  model: "YOLO | None" = None, use_kf: bool = True) -> dict:
    """Pseudo-label every clip in `clips_dir` and write a YOLO-seg dataset.

    Returns aggregate stats: total labeled images and a per-clip breakdown
    (clip stem -> labeled-frame count). Pass a preloaded `model` to skip
    reloading weights. Reused by both pseudo_label's CLI and pipeline.py.
    """
    img_dir = out / "images" / "train"
    lbl_dir = out / "labels" / "train"
    for d in (img_dir, lbl_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    clips = sorted(clips_dir.glob("*.mp4"))
    if not clips:
        raise SystemExit(f"no .mp4 files in {clips_dir}")

    if model is None:
        print(f"loading {weights}...")
        model = YOLO(weights)

    total = 0
    per_clip: dict[str, int] = {}
    for clip in clips:
        print(f"\nclip: {clip.name}")
        accepted, stats = label_clip(
            model, clip, conf=conf, classes=classes,
            gate=gate, max_miss=max_miss, use_kf=use_kf)
        print(f"  total frames:           {stats['total_frames']}")
        print(f"  frames w/ detections:   {stats['frames_with_detections']}")
        print(f"  accepted after Kalman:  {stats['accepted']}")
        print(f"  rejected as outliers:   {stats['rejected']}")

        cap = cv2.VideoCapture(str(clip))
        frame_idx = 0
        kept = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in accepted:
                stem = f"{clip.stem}_f{frame_idx:05d}"
                cv2.imwrite(str(img_dir / f"{stem}.jpg"), frame)
                poly = accepted[frame_idx]
                coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
                (lbl_dir / f"{stem}.txt").write_text(f"0 {coords}\n")
                kept += 1
            frame_idx += 1
        cap.release()
        total += kept
        per_clip[clip.stem] = kept
        print(f"  wrote {kept} labeled frames")

    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/train",
        "nc": 1,
        "names": [class_name],
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    print(f"\ntotal labeled images: {total}")
    print(f"dataset: {out}/data.yaml")
    return {"total": total, "per_clip": per_clip}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips_dir", type=Path, help="Directory of .mp4 clips")
    ap.add_argument("--out", type=Path, default=Path("data/datasets/gen1"))
    ap.add_argument("--weights", default="data/pretrained/yolo26x-seg.pt")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--gate", type=float, default=GATE,
                    help="Mahalanobis gate (default %(default)s ≈ 99.9% chi-sq, 2 DoF)")
    ap.add_argument("--max-miss", type=int, default=MAX_MISS,
                    help="Reinit Kalman after this many consecutive missed frames")
    ap.add_argument("--classes", default="cup-like",
                    help="'cup-like' (default COCO subset), 'all' (no filter), "
                         "or comma-separated class ints (e.g. '0' for a "
                         "single-class fine-tuned model)")
    ap.add_argument("--name", default="my_cup", help="single-class name in data.yaml")
    ap.add_argument("--no-kf", action="store_true",
                    help="ablation: keep top-conf detection per frame, skip the Kalman filter")
    args = ap.parse_args()

    classes = resolve_classes(args.classes)
    build_dataset(args.clips_dir, args.out, args.weights, args.conf, classes,
                  gate=args.gate, max_miss=args.max_miss, class_name=args.name,
                  use_kf=not args.no_kf)
    print(f"next:    python finetune.py --data {args.out}/data.yaml --name <run_name>")


if __name__ == "__main__":
    main()
