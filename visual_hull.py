"""Carve a 3D cup model from N calibrated views using YOLOv11-seg silhouettes.

Inputs:  one PNG per camera (matching the names in CAM_MAPPING) + TOML calib.
Outputs: <out>/cup_points.ply  (occupied voxel centers as a point cloud)
         <out>/cup_mesh.ply    (Poisson surface reconstruction)
         <out>/masks/<cam>.png (debug overlays of YOLO masks)

All world coords are in millimeters (matches iMOVE calibration).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from ultralytics import YOLO

from calibration import Camera, load_calibration


# ---- tuning constants (millimeters) ----------------------------------------
CALIB_TOML = ("/home/imove/Documents/iMOVE/DATA/sub-test/20260428-152404/trials/"
              "calibration_20260428_152404_calibration.filtered.1-2-3.toml")
CAM_MAPPING = {"cam_left": "cam_0", "cam_center": "cam_1", "cam_right": "cam_2"}
IMAGE_SIZE = (1280, 720)

# Working volume: cube of side VOLUME_SIDE_MM, centered on the triangulated
# cup centroid (auto-detected from YOLO masks). The Charuco world origin is
# usually not where the cup sits, so we don't anchor on (0,0,0).
VOLUME_SIDE_MM = 400.0
VOXEL_SIZE_MM = 2.0

YOLO_WEIGHTS = "yolo11n-seg.pt"
COCO_CUP_CLASS = 41
# ----------------------------------------------------------------------------


def yolo_cup_mask(model: YOLO, frame: np.ndarray) -> np.ndarray | None:
    """Returns a binary uint8 mask (H,W) at frame resolution, or None."""
    res = model(frame, classes=[COCO_CUP_CLASS], verbose=False)[0]
    if res.masks is None or len(res.masks.data) == 0:
        return None
    # pick highest-confidence detection
    conf = res.boxes.conf.cpu().numpy()
    best = int(np.argmax(conf))
    m = res.masks.data[best].cpu().numpy()
    h, w = frame.shape[:2]
    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    return (m > 0.5).astype(np.uint8) * 255


def project_world_points(pts_w: np.ndarray, cam: Camera) -> np.ndarray:
    """pts_w: (N,3) in mm. Returns (N,2) pixel coords (no distortion: frames are
    already undistorted before being passed in)."""
    pts_cam = pts_w @ cam.R.T + cam.t        # (N,3)
    z = pts_cam[:, 2]
    uvw = pts_cam @ cam.K.T
    uv = uvw[:, :2] / z[:, None]
    return uv, z


def mask_centroid(mask: np.ndarray) -> np.ndarray:
    """Returns (u, v) centroid of a binary mask, or None if mask is empty."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.mean(), ys.mean()])


def triangulate_centroid(centroids: dict[str, np.ndarray],
                         cams: dict[str, Camera]) -> np.ndarray:
    """DLT triangulation across N>=2 views. Returns (3,) world point in mm."""
    rows = []
    for name, uv in centroids.items():
        P = cams[name].P
        u, v = uv
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    A = np.stack(rows, axis=0)
    _, _, vt = np.linalg.svd(A)
    X = vt[-1]
    return (X[:3] / X[3])


def carve(masks: dict[str, np.ndarray], cams: dict[str, Camera],
          center: np.ndarray, min_views: int) -> np.ndarray:
    """Returns (M,3) world-space voxel centers (mm) that survived carving."""
    half = VOLUME_SIDE_MM / 2.0
    vmin = center - half
    vmax = center + half
    xs = np.arange(vmin[0], vmax[0], VOXEL_SIZE_MM)
    ys = np.arange(vmin[1], vmax[1], VOXEL_SIZE_MM)
    zs = np.arange(vmin[2], vmax[2], VOXEL_SIZE_MM)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    print(f"  voxel grid: {len(xs)}x{len(ys)}x{len(zs)} = {len(pts):,} voxels")

    votes = np.zeros(len(pts), dtype=np.int32)
    for name, cam in cams.items():
        mask = masks[name]
        h, w = mask.shape
        uv, z = project_world_points(pts, cam)
        u = np.round(uv[:, 0]).astype(np.int32)
        v = np.round(uv[:, 1]).astype(np.int32)
        in_front = z > 1.0
        in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        ok = in_front & in_bounds
        idx = np.where(ok)[0]
        inside = mask[v[idx], u[idx]] > 0
        votes[idx[inside]] += 1
        print(f"  {name}: {inside.sum():,} voxel hits")

    keep = votes >= min_views
    print(f"  kept {keep.sum():,} voxels (>= {min_views} views)")
    return pts[keep]


def _parse_cam_mapping(s: str) -> dict[str, str]:
    """Parse 'stream1:section1,stream2:section2,...' into a dict."""
    result = {}
    for pair in s.split(","):
        k, _, v = pair.partition(":")
        if not k or not v:
            raise argparse.ArgumentTypeError(
                f"invalid --cam-mapping entry {pair!r}, expected stream:section"
            )
        result[k.strip()] = v.strip()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("triplet_dir", type=Path,
                    help="Directory with one PNG per camera named *<stream_name>.png")
    ap.add_argument("--out", type=Path, default=Path("cup_model"))
    ap.add_argument("--calib", default=CALIB_TOML)
    ap.add_argument("--cam-mapping", default=None, type=_parse_cam_mapping,
                    help="stream:section pairs, e.g. cam_left:cam_0,cam_center:cam_1 "
                         "(default: uses CAM_MAPPING constant in this file)")
    ap.add_argument("--min-views", type=int, default=None,
                    help="Voxels kept if seen by at least this many cameras "
                         "(default: all cameras — strict mode)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "masks").mkdir(exist_ok=True)

    cam_mapping = args.cam_mapping if args.cam_mapping else CAM_MAPPING
    cams = load_calibration(args.calib, cam_mapping, target_image_size=IMAGE_SIZE)
    min_views = args.min_views if args.min_views is not None else len(cams)
    print(f"cameras: {list(cams.keys())}  min_views={min_views}/{len(cams)}")

    # Load one PNG per stream (most-recent file matching the name)
    frames = {}
    for name in cam_mapping:
        matches = sorted(args.triplet_dir.glob(f"*{name}.png"))
        if not matches:
            raise SystemExit(f"no file matching *{name}.png in {args.triplet_dir}")
        frames[name] = cv2.imread(str(matches[-1]))
        print(f"loaded {matches[-1].name}  ({frames[name].shape[1]}x{frames[name].shape[0]})")

    # Undistort and rebuild K-only intrinsics for the rectified frames
    for name, cam in cams.items():
        und = cv2.undistort(frames[name], cam.K, cam.dist)
        frames[name] = und
        cam.dist = np.zeros(5)

    print("running YOLO...")
    model = YOLO(YOLO_WEIGHTS)
    masks: dict[str, np.ndarray] = {}
    for name, frame in frames.items():
        m = yolo_cup_mask(model, frame)
        if m is None:
            raise SystemExit(f"YOLO found no cup in {name} — check framing/lighting")
        masks[name] = m
        overlay = frame.copy()
        overlay[m > 0] = (0.5 * overlay[m > 0] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        cv2.imwrite(str(args.out / "masks" / f"{name}.png"), overlay)
        print(f"  {name}: mask area = {int((m > 0).sum())} px")

    centroids = {n: mask_centroid(m) for n, m in masks.items()}
    centroids = {n: c for n, c in centroids.items() if c is not None}
    if len(centroids) < 2:
        raise SystemExit("need >=2 valid mask centroids to triangulate scene center")
    center = triangulate_centroid(centroids, cams)
    print(f"triangulated cup centroid: {center.round(1)} mm  "
          f"(carving {VOLUME_SIDE_MM:.0f}mm cube around it)")

    print("carving...")
    pts = carve(masks, cams, center, min_views)
    if len(pts) == 0:
        raise SystemExit(
            "0 voxels survived. Likely CAM_MAPPING is wrong (cam_left/center/right "
            "swapped). Check masks/ overlays to confirm which physical view is which.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=VOXEL_SIZE_MM * 4, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(20)
    o3d.io.write_point_cloud(str(args.out / "cup_points.ply"), pcd)
    print(f"  wrote {args.out/'cup_points.ply'}")

    try:
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8)
        o3d.io.write_triangle_mesh(str(args.out / "cup_mesh.ply"), mesh)
        print(f"  wrote {args.out/'cup_mesh.ply'}  ({len(mesh.triangles)} tris)")
    except Exception as e:
        print(f"  Poisson meshing failed ({e}); point cloud only")


if __name__ == "__main__":
    main()
