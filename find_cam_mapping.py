"""Brute-force the stream->calibration mapping by trying all N! permutations.

For each permutation of calibration sections assigned to ZMQ stream names,
triangulate the YOLO cup centroid and carve on a coarse grid. The correct
mapping is the one with the most voxels kept across all views.

Works with any number of cameras (2, 3, 4, …).
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import cv2
import numpy as np
import toml
from ultralytics import YOLO

from calibration import load_calibration
from cam_streams import load_streams
from visual_hull import (CALIB_TOML, COCO_CUP_CLASS, IMAGE_SIZE,
                         mask_centroid, triangulate_centroid, yolo_cup_mask)

# Coarse grid: 300mm cube at 5mm voxels = 216k voxels (fast)
SIDE_MM = 300.0
VOXEL_MM = 5.0


def _calib_sections(toml_path: str) -> tuple[str, ...]:
    """Return all camera section names from a calibration TOML."""
    data = toml.load(str(toml_path))
    return tuple(k for k in data if k != "metadata")


def carve_coarse(masks, cams, center, min_views: int):
    half = SIDE_MM / 2
    g = np.arange(-half, half, VOXEL_MM)
    gx, gy, gz = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack([gx.ravel() + center[0], gy.ravel() + center[1], gz.ravel() + center[2]], axis=1)
    votes = np.zeros(len(pts), dtype=np.int32)
    per_cam_hits = {}
    for name, cam in cams.items():
        h, w = masks[name].shape
        pcam = pts @ cam.R.T + cam.t
        z = pcam[:, 2]
        ok_z = z > 1.0
        uvw = pcam @ cam.K.T
        with np.errstate(invalid="ignore", divide="ignore"):
            u = uvw[:, 0] / z
            v = uvw[:, 1] / z
        in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        ok = ok_z & in_bounds
        idx = np.where(ok)[0]
        u_i = u[idx].astype(np.int32)
        v_i = v[idx].astype(np.int32)
        inside = masks[name][v_i, u_i] > 0
        votes[idx[inside]] += 1
        per_cam_hits[name] = int(inside.sum())
    kept = int((votes >= min_views).sum())
    return kept, per_cam_hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("triplet_dir", type=Path)
    ap.add_argument("--calib", default=CALIB_TOML)
    ap.add_argument("--config", default=None,
                    help="Path to cam_config_server.yaml (auto-detected if omitted)")
    args = ap.parse_args()

    streams = list(load_streams(args.config).keys())
    calib_sections = _calib_sections(args.calib)
    n = len(streams)
    print(f"streams ({n}): {streams}")
    print(f"calib sections ({len(calib_sections)}): {calib_sections}")
    if n != len(calib_sections):
        raise SystemExit(
            f"stream count ({n}) != calibration section count ({len(calib_sections)})"
        )
    min_views = n  # require all cameras to agree

    print("loading frames + running YOLO once per stream...")
    model = YOLO("yolo11n-seg.pt")
    frames, masks = {}, {}
    for s in streams:
        matches = sorted(args.triplet_dir.glob(f"*{s}.png"))
        if not matches:
            raise SystemExit(f"no *{s}.png in {args.triplet_dir}")
        f = cv2.imread(str(matches[-1]))
        m = yolo_cup_mask(model, f)
        if m is None:
            raise SystemExit(f"no cup detected in {s}")
        frames[s], masks[s] = f, m
        print(f"  {s}: {matches[-1].name}  mask={int((m>0).sum())}px")

    print()
    results = []
    for perm in itertools.permutations(calib_sections):
        mapping = dict(zip(streams, perm))
        cams = load_calibration(args.calib, mapping, target_image_size=IMAGE_SIZE)
        masks_u = {}
        for s, cam in cams.items():
            masks_u[s] = masks[s]
            cam.dist = np.zeros(5)
        centroids = {n: mask_centroid(m) for n, m in masks_u.items()}
        try:
            center = triangulate_centroid(centroids, cams)
        except Exception as e:
            print(f"  {mapping} -> triangulation failed: {e}")
            continue
        kept, per_cam = carve_coarse(masks_u, cams, center, min_views)
        results.append((kept, mapping, center, per_cam))
        print(f"  {mapping}  kept={kept:>6}  center={center.round(0)}  hits={per_cam}")

    results.sort(reverse=True, key=lambda r: r[0])
    best = results[0]
    print()
    print(f"BEST mapping (kept={best[0]} voxels):")
    for k, v in best[1].items():
        print(f"  {k!r}: {v!r},")
    print(f"\nPass this as --cam-mapping to visual_hull.py, or set CAM_MAPPING in the file.")


if __name__ == "__main__":
    main()
