"""Evaluate any YOLO-seg model on any directory of clips and print
precision/recall numbers using the single-object Kalman filter as the
"is this detection on the tracked object?" judge.

Two-step internally:
    1. Cache per-frame detections to JSON (skipped if already cached).
    2. Run KF over the cache and aggregate counts into PR proxies.

The caching step is the slow one. Re-running with the same --weights/--clips/
--conf reads from cache and finishes in <1 second.

Metric definitions (also documented in experiments/2026-05-26_cup_5cam_demo.md):
    recall   = frames with ≥1 detection / total frames
               (justified by "tracked object visible in every frame")
    P_loose  = (total_dets - phantoms - teleports) / total_dets
               (NMS duplicates count as correct — still on the object)
    P_strict = (total_dets - phantoms - teleports - duplicates) / total_dets

Examples
--------
    # Held-out test set for the gen2 model
    python evaluate.py --weights data/runs/segment/cup_5cam_demo_gen2/weights/best.pt \\
        --clips data/clips/test

    # Same model on its own training set (overfit check)
    python evaluate.py --weights data/runs/segment/cup_5cam_demo_gen2/weights/best.pt \\
        --clips data/clips/train

    # COCO teacher with a class filter
    python evaluate.py --weights data/pretrained/yolo26x-seg.pt --clips data/clips/test \\
        --classes 39,40,41,45,75 --conf 0.25
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from kalman import filter_detections


def cache_detections(model: YOLO, clip_dir: Path, out_dir: Path,
                     conf: float, classes: list[int] | None) -> None:
    """Run model on every cam_*.mp4 in clip_dir, write per-frame dets to JSON.

    Skips any clip whose JSON already exists; safe to re-run.
    """
    clips = sorted(clip_dir.glob("cam_*.mp4"))
    if not clips:
        raise SystemExit(f"no cam_*.mp4 files in {clip_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for clip in clips:
        cam = clip.stem.rsplit("_", 2)[0]
        out_path = out_dir / f"{cam}.json"
        if out_path.exists():
            print(f"  {cam}: cached")
            continue
        cap = cv2.VideoCapture(str(clip))
        frames, idx = [], 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            res = model(frame, classes=classes, conf=conf, verbose=False)[0]
            dets = []
            if res.boxes is not None and len(res.boxes) > 0:
                bxs = res.boxes.xyxy.cpu().numpy()
                cfs = res.boxes.conf.cpu().numpy()
                for b, c in zip(bxs, cfs):
                    cx, cy = float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2)
                    dets.append({
                        "cx": cx, "cy": cy, "conf": float(c),
                        "bbox": [float(b[0]), float(b[1]),
                                 float(b[2]), float(b[3])],
                    })
            frames.append({"frame": idx, "dets": dets})
            idx += 1
        cap.release()
        out_path.write_text(json.dumps(frames))
        print(f"  {cam}: {idx} frames")


def compute_metrics(cache_dir: Path) -> dict:
    """Aggregate KF-filter counts across all per-cam JSON files in cache_dir."""
    cams = sorted(p.stem for p in cache_dir.glob("*.json"))
    if not cams:
        raise SystemExit(f"no cached *.json files in {cache_dir}")

    agg = dict(F=0, det_F=0, total=0, dup=0, phantom=0, tele=0,
               accepted=0, predicted=0, rejected=0)

    for cam in cams:
        data = json.loads((cache_dir / f"{cam}.json").read_text())
        per_frame = [d["dets"] for d in data]
        filtered = filter_detections(per_frame)

        agg["F"] += len(per_frame)
        agg["det_F"] += sum(1 for p in per_frame if p)
        agg["total"] += sum(len(p) for p in per_frame)
        for ff in filtered:
            agg["dup"] += ff.dup
            agg["phantom"] += ff.phantom
            agg["tele"] += ff.tele
            if ff.role == "accepted":
                agg["accepted"] += 1
            elif ff.role == "predicted":
                agg["predicted"] += 1
            elif ff.role == "rejected":
                agg["rejected"] += 1

    real_err = agg["phantom"] + agg["tele"]
    recall = agg["det_F"] / agg["F"] if agg["F"] else 0.0
    p_loose = (agg["total"] - real_err) / agg["total"] if agg["total"] else 0.0
    p_strict = (agg["total"] - real_err - agg["dup"]) / agg["total"] if agg["total"] else 0.0
    f1_loose = (2 * p_loose * recall / (p_loose + recall)
                if (p_loose + recall) > 0 else 0.0)
    f1_strict = (2 * p_strict * recall / (p_strict + recall)
                 if (p_strict + recall) > 0 else 0.0)

    return dict(**agg, recall=recall, p_loose=p_loose, p_strict=p_strict,
                f1_loose=f1_loose, f1_strict=f1_strict)


def print_table(name: str, m: dict) -> None:
    header = (f"{'config':<32} {'frames':>7} {'det_F':>6} {'tot_det':>8} "
              f"{'dup':>5} {'phan':>5} {'tele':>5} "
              f"{'recall':>7} {'P_loose':>8} {'F1_loose':>9} "
              f"{'P_strict':>9} {'F1_strict':>10}")
    row = (f"{name:<32} {m['F']:>7} {m['det_F']:>6} {m['total']:>8} "
           f"{m['dup']:>5} {m['phantom']:>5} {m['tele']:>5} "
           f"{m['recall']:>7.3f} {m['p_loose']:>8.3f} {m['f1_loose']:>9.3f} "
           f"{m['p_strict']:>9.3f} {m['f1_strict']:>10.3f}")
    print(header)
    print(row)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True,
                    help="Path to a YOLO-seg .pt file")
    ap.add_argument("--clips", type=Path, required=True,
                    help="Directory of cam_*.mp4 clips to evaluate on")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="Detection confidence threshold (default 0.25)")
    ap.add_argument("--classes", default=None,
                    help="Optional class filter: 'all' or comma-separated "
                         "ints (e.g. '39,40,41,45,75'). Default: all classes.")
    ap.add_argument("--cache", type=Path, default=Path("data/detections_cache"),
                    help="Root dir for cached per-frame JSON")
    ap.add_argument("--name", default=None,
                    help="Cache subdir name. Default: <weights stem>_<conf>")
    args = ap.parse_args()

    classes = None
    if args.classes and args.classes != "all":
        classes = [int(x) for x in args.classes.split(",")]

    name = args.name or f"{Path(args.weights).stem}_{args.conf}"
    cache_dir = args.cache / args.clips.name / name

    print(f"weights:  {args.weights}")
    print(f"clips:    {args.clips}")
    print(f"conf:     {args.conf}")
    print(f"classes:  {classes if classes else 'all'}")
    print(f"cache:    {cache_dir}\n")

    print("caching detections...")
    model = YOLO(args.weights)
    cache_detections(model, args.clips, cache_dir, args.conf, classes)

    print("\ncomputing metrics...")
    m = compute_metrics(cache_dir)
    print()
    print_table(name, m)


if __name__ == "__main__":
    main()
