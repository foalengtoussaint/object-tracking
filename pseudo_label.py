"""Auto-label MP4 clips with a big YOLO-seg, then run a single-object Kalman
filter on the detection centroids so only the cup that moves smoothly through
the scene is kept — at most one detection per frame, false positives gated out
by Mahalanobis distance from the predicted position.

Outputs a YOLO single-class seg dataset:
    <out>/images/train/<clip>_f<idx>.jpg
    <out>/labels/train/<clip>_f<idx>.txt   ("0 x1 y1 x2 y2 ...")
    <out>/data.yaml
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

# bottle, wine glass, cup, bowl, vase
CUP_LIKE_CLASSES = [39, 40, 41, 45, 75]


class KalmanFilter2D:
    """Constant-velocity Kalman filter for a 2D centroid (state = x, y, vx, vy)."""

    def __init__(self, process_noise: float = 4.0, meas_noise: float = 8.0):
        dt = 1.0
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        q = process_noise ** 2
        self.Q = q * np.array([
            [dt**4 / 4, 0, dt**3 / 2, 0],
            [0, dt**4 / 4, 0, dt**3 / 2],
            [dt**3 / 2, 0, dt**2, 0],
            [0, dt**3 / 2, 0, dt**2],
        ])
        self.R = (meas_noise ** 2) * np.eye(2)
        self.x = None
        self.P = None

    def init(self, z):
        self.x = np.array([z[0], z[1], 0.0, 0.0])
        self.P = np.diag([10.0, 10.0, 100.0, 100.0])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def innovation_stats(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return y, S

    def mahalanobis(self, z) -> float:
        y, S = self.innovation_stats(z)
        return float(y @ np.linalg.inv(S) @ y)

    def update(self, z):
        y, S = self.innovation_stats(z)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P


def label_clip(model: YOLO, clip: Path, conf: float, gate: float,
               max_miss: int) -> tuple[dict[int, np.ndarray], dict[str, int]]:
    """Run detection + single-object Kalman filter. Returns
    {frame_idx: polygon} of accepted detections."""
    # Pass 1: collect all candidates per frame
    candidates: dict[int, list[tuple[np.ndarray, np.ndarray, float]]] = {}
    for frame_idx, result in enumerate(model.predict(
            source=str(clip), classes=CUP_LIKE_CLASSES, conf=conf,
            stream=True, verbose=False)):
        if (result.boxes is None or len(result.boxes) == 0
                or result.masks is None):
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        polys = result.masks.xyn  # list of (N,2) normalized
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        candidates[frame_idx] = list(zip(centers, polys, confs))
    n_total_frames = frame_idx + 1
    n_cand_frames = len(candidates)

    # Pass 2: run Kalman filter forward across all frames
    kf = KalmanFilter2D()
    accepted: dict[int, np.ndarray] = {}
    misses = 0
    initialized = False
    n_rejected = 0
    n_dropped_when_uninit = 0

    for frame_idx in range(n_total_frames):
        cands = candidates.get(frame_idx, [])
        if not initialized:
            # Initialize from the first frame that has any detection
            if cands:
                # Pick highest-confidence candidate as anchor
                centers = [c[0] for c in cands]
                confs = [c[2] for c in cands]
                i = int(np.argmax(confs))
                kf.init(centers[i])
                accepted[frame_idx] = cands[i][1]
                initialized = True
                misses = 0
                n_dropped_when_uninit += len(cands) - 1
            continue

        kf.predict()
        if not cands:
            misses += 1
            continue

        # Gate every candidate by Mahalanobis distance
        scored = [(kf.mahalanobis(c[0]), c) for c in cands]
        scored.sort(key=lambda s: s[0])
        best_d, best_c = scored[0]
        if best_d <= gate and misses <= max_miss:
            kf.update(best_c[0])
            accepted[frame_idx] = best_c[1]
            misses = 0
            n_rejected += len(cands) - 1
        elif best_d <= gate and misses > max_miss:
            # Long occlusion — re-initialize from best candidate
            kf.init(best_c[0])
            accepted[frame_idx] = best_c[1]
            misses = 0
            n_rejected += len(cands) - 1
        else:
            # No candidate within gate → reject all, keep predicting
            misses += 1
            n_rejected += len(cands)

    stats = {
        "total_frames": n_total_frames,
        "frames_with_detections": n_cand_frames,
        "accepted_frames": len(accepted),
        "rejected_after_init": n_rejected,
        "dropped_before_init": n_dropped_when_uninit,
    }
    return accepted, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips_dir", type=Path, help="Directory of .mp4 clips")
    ap.add_argument("--out", type=Path, default=Path("dataset"))
    ap.add_argument("--weights", default="yolo26x-seg.pt")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--gate", type=float, default=13.82,
                    help="Mahalanobis gate (chi-sq 2-DoF: 5.99=95%, 9.21=99%, 13.82=99.9%)")
    ap.add_argument("--max-miss", type=int, default=30,
                    help="Reinit Kalman after this many consecutive missed frames")
    args = ap.parse_args()

    img_dir = args.out / "images" / "train"
    lbl_dir = args.out / "labels" / "train"
    for d in (img_dir, lbl_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    clips = sorted(args.clips_dir.glob("*.mp4"))
    if not clips:
        raise SystemExit(f"no .mp4 files in {args.clips_dir}")

    print(f"loading {args.weights}...")
    model = YOLO(args.weights)

    total = 0
    for clip in clips:
        print(f"\nclip: {clip.name}")
        accepted, stats = label_clip(
            model, clip, conf=args.conf, gate=args.gate, max_miss=args.max_miss)
        print(f"  total frames:           {stats['total_frames']}")
        print(f"  frames w/ detections:   {stats['frames_with_detections']}")
        print(f"  accepted after Kalman:  {stats['accepted_frames']}")
        print(f"  rejected as outliers:   {stats['rejected_after_init']}")

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
        print(f"  wrote {kept} labeled frames")

    data_yaml = {
        "path": str(args.out.resolve()),
        "train": "images/train",
        "val": "images/train",
        "nc": 1,
        "names": ["my_cup"],
    }
    (args.out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    print(f"\ntotal labeled images: {total}")
    print(f"dataset: {args.out}/data.yaml")


if __name__ == "__main__":
    main()
