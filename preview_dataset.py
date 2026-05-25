"""Build a preview MP4 of every labeled frame in a YOLO single-class dataset,
with the polygon mask overlaid in green. Lets you visually confirm what the
auto-labeler kept.

Usage:
    python preview_dataset.py                       # dataset/ -> dataset_preview.mp4
    python preview_dataset.py --data dataset --out preview.mp4 --fps 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("dataset"))
    ap.add_argument("--out", type=Path, default=Path("dataset_preview.mp4"))
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--tail", type=int, default=40,
                    help="Number of past centroids to draw as trajectory tail")
    args = ap.parse_args()

    img_dir = args.data / "images" / "train"
    lbl_dir = args.data / "labels" / "train"
    imgs = sorted(img_dir.glob("*.jpg"))
    if not imgs:
        raise SystemExit(f"no images in {img_dir}")

    first = cv2.imread(str(imgs[0]))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(args.out),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (w, h))
    tail: list[tuple[int, int]] = []
    for i, img_path in enumerate(imgs):
        frame = cv2.imread(str(img_path))
        lbl = lbl_dir / f"{img_path.stem}.txt"
        if lbl.exists():
            overlay = frame.copy()
            polys = []
            for line in lbl.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                coords = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(-1, 2)
                poly = (coords * np.array([w, h])).astype(np.int32)
                polys.append(poly)
                cv2.fillPoly(overlay, [poly], (0, 255, 0))
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
            for poly in polys:
                cv2.polylines(frame, [poly], True, (0, 255, 0), 2)
            if polys:
                cx = int(np.mean(polys[0][:, 0]))
                cy = int(np.mean(polys[0][:, 1]))
                tail.append((cx, cy))
                tail = tail[-args.tail:]
        # draw trajectory tail (oldest faintest)
        for j in range(1, len(tail)):
            alpha = j / len(tail)            # 0..1
            color = (0, int(165 * alpha), int(255 * alpha))  # red→yellow tail
            cv2.line(frame, tail[j - 1], tail[j], color, 2)
        if tail:
            cv2.circle(frame, tail[-1], 5, (0, 255, 255), -1)
        cv2.putText(frame, f"{i+1}/{len(imgs)} {img_path.name}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        writer.write(frame)

    writer.release()
    print(f"wrote {len(imgs)} frames @ {args.fps} fps to {args.out}")
    print(f"play with:  xdg-open {args.out}")


if __name__ == "__main__":
    main()
