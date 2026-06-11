"""Overlay predicted cup pose as mesh wireframe on each frame, write mp4.

Validation tool: lets you visually confirm the predicted 6D pose places
the mesh correctly on the 2D image. No rerun, no GPU compositing — just
cv2.projectPoints + cv2.polylines on the mesh edges.

Mesh is mm; megapose pose is meters → we scale vertices to meters once.

Usage:
    conda activate object_tracking
    PYTHONUNBUFFERED=1 python pose_6d/overlay_megapose_mp4.py \\
        --clip data/clips/test/cam_1_20260526_163138.mp4 \\
        --cam cam_1 --every 10 --max-frames 200 \\
        --out data/recordings/cup_overlay.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import pyrender
import toml
import trimesh
from scipy.spatial.transform import Rotation, Slerp
from ultralytics import YOLO

_PROJ = Path("/home/imove/Documents/object_tracking")
DEFAULT_WEIGHTS = str(_PROJ / "data/runs/segment/cup_5cam_demo_gen2/weights/best.pt")
DEFAULT_CAL = str(_PROJ / "data/calibration_local.toml")
MESH_PATH = str(_PROJ / "pose_6d/meshes/cup.obj")
EXAMPLE_DIR = _PROJ / "pose_6d/megapose6d/local_data/examples/my-cup"
MEGAPOSE_CWD = _PROJ / "pose_6d/megapose6d"
LABEL = "my-cup"


def load_cam_K(cal_path: str, cam_name: str, target_size: tuple[int, int]) -> np.ndarray:
    data = toml.load(cal_path)
    for k, b in data.items():
        if not k.startswith("cam_") or not isinstance(b, dict):
            continue
        name = b.get("name", "")
        normalized = name if name.startswith("cam_") else (
            f"cam_{name[3:]}" if name.startswith("cam") else f"cam_{name}"
        )
        if normalized == cam_name:
            K = np.array(b["matrix"], float)
            cal_w, cal_h = b["size"]
            K[0, :] *= target_size[0] / cal_w
            K[1, :] *= target_size[1] / cal_h
            return K
    raise SystemExit(f"cam {cam_name} not in {cal_path}")


def run_megapose_subprocess(rgb_bgr: np.ndarray, K: np.ndarray, bbox: list[float],
                            H: int, W: int, timeout: int = 30) -> dict | None:
    (EXAMPLE_DIR / "inputs").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(EXAMPLE_DIR / "image_rgb.png"), rgb_bgr)
    json.dump({"K": K.tolist(), "resolution": [H, W]},
              open(EXAMPLE_DIR / "camera_data.json", "w"))
    json.dump([{"label": LABEL, "bbox_modal": [int(round(v)) for v in bbox]}],
              open(EXAMPLE_DIR / "inputs" / "object_data.json", "w"))
    out_json = EXAMPLE_DIR / "outputs" / "object_data.json"
    out_json.unlink(missing_ok=True)

    cmd = ["conda", "run", "-n", "megapose", "--no-capture-output",
           "python", "-m", "megapose.scripts.run_inference_on_example",
           LABEL, "--run-inference"]
    clean_env = {k: v for k, v in os.environ.items()
                 if not k.startswith("CONDA_") and k not in {
                     "PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH",
                     "VIRTUAL_ENV", "_CE_M", "_CE_CONDA"}}
    clean_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    clean_env["CUDA_VISIBLE_DEVICES"] = "0"
    clean_env["EGL_VISIBLE_DEVICES"] = "0"
    t0 = time.time()
    try:
        res = subprocess.run(cmd, cwd=str(MEGAPOSE_CWD), env=clean_env,
                             capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  megapose timeout after {timeout}s", flush=True)
        return None
    dt = time.time() - t0
    if res.returncode != 0:
        tail = res.stderr.strip().splitlines()[-3:]
        print(f"  megapose failed in {dt:.1f}s: {tail}", flush=True)
        return None
    if not out_json.exists():
        return None
    pred = json.load(open(out_json))[0]
    pred["_dt_s"] = dt
    return pred


def quat_xyzw_to_R(q: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(q).as_matrix()


def bbox_to_t(bbox: list[float], K: np.ndarray, height_m: float) -> np.ndarray:
    """Back-project bbox to cup-centroid 3D point in camera frame.

    Assumes cup is upright and its physical height is `height_m`. Uses
    bbox height in pixels + camera fy for depth via similar triangles,
    then unprojects bbox center.
    """
    x1, y1, x2, y2 = bbox
    h_pix = max(y2 - y1, 1.0)
    Z = K[1, 1] * height_m / h_pix
    cx_pix = 0.5 * (x1 + x2)
    cy_pix = 0.5 * (y1 + y2)
    X = (cx_pix - K[0, 2]) * Z / K[0, 0]
    Y = (cy_pix - K[1, 2]) * Z / K[1, 1]
    return np.array([X, Y, Z])


# OpenCV camera frame (x right, y down, z fwd) → OpenGL camera frame (x right, y up, z back)
CV2GL = np.diag([1.0, -1.0, -1.0, 1.0])


def build_pyrender_scene(mesh: trimesh.Trimesh, K: np.ndarray, W: int, H: int):
    """Return (scene, mesh_node, renderer). Mesh assumed already in meters."""
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.4, 0.4, 0.4])
    rmesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    mesh_node = scene.add(rmesh, pose=np.eye(4))
    cam = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1],
                                    cx=K[0, 2], cy=K[1, 2],
                                    znear=0.01, zfar=10.0)
    scene.add(cam, pose=CV2GL)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=CV2GL)
    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    return scene, mesh_node, renderer


def overlay_textured(img: np.ndarray, scene, mesh_node, renderer,
                     R: np.ndarray, t: np.ndarray, alpha: float = 0.85) -> None:
    """Render mesh at pose (R,t) and alpha-blend over img (BGR, in place)."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    scene.set_pose(mesh_node, pose=T)
    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    mask = depth > 0
    if not mask.any():
        return
    rgb_bgr = cv2.cvtColor(color[..., :3], cv2.COLOR_RGB2BGR)
    img[mask] = (alpha * rgb_bgr[mask] + (1.0 - alpha) * img[mask]).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--cam", required=True)
    ap.add_argument("--cal", default=DEFAULT_CAL)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--mesh", default=MESH_PATH)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--mp-timeout", type=int, default=30)
    ap.add_argument("--no-interp", action="store_true",
                    help="only draw mesh on detection frames (else slerp)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.clip)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.clip}")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("clip empty")
    H, W = frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print(f"clip: {n_total} frames @ {fps:.1f}fps  {W}x{H}", flush=True)

    K = load_cam_K(args.cal, args.cam, (W, H))

    print("loading YOLO on CPU...", flush=True)
    yolo = YOLO(args.weights)
    yolo.to("cpu")

    print("loading mesh...", flush=True)
    mesh = trimesh.load(args.mesh)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    mesh.apply_scale(0.001)  # mm → m, keeps texture/material
    cup_height_m = float(mesh.bounds[1, 2] - mesh.bounds[0, 2])  # Z-up after rotation
    print(f"mesh: {len(mesh.vertices)} verts (scaled to meters), "
          f"height={cup_height_m*1000:.1f}mm", flush=True)

    print("building pyrender scene...", flush=True)
    scene, mesh_node, renderer = build_pyrender_scene(mesh, K, W, H)

    n_max = min(args.max_frames or n_total, n_total)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W, H))
    if not writer.isOpened():
        raise SystemExit(f"cannot open writer for {args.out}")

    # pass 1: run megapose on every Nth frame, cache poses
    # if no detection at frame K, search frames K±1..K±5 for one
    print("pass 1: detection + megapose...", flush=True)
    poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}  # frame -> (R, t)
    bboxes: dict[int, list[float]] = {}
    t_p1 = time.time()
    used_frames: set[int] = set()
    search_deltas = [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5]
    targets = list(range(0, n_max + 1, args.every))
    for target in targets:
        det_frame: int | None = None
        det_bbox: list[float] | None = None
        det_img: np.ndarray | None = None
        for d in search_deltas:
            f_try = target + d
            if f_try < 0 or f_try >= n_total or f_try in used_frames:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_try)
            ok, f = cap.read()
            if not ok:
                continue
            res = yolo.predict(f, classes=[0], conf=args.conf,
                               verbose=False, device="cpu")[0]
            if res.boxes is None or len(res.boxes) == 0:
                continue
            confs = res.boxes.conf.cpu().numpy()
            det_bbox = res.boxes.xyxy.cpu().numpy()[int(np.argmax(confs))].tolist()
            det_frame = f_try
            det_img = f
            break
        if det_frame is None:
            print(f"  [{target}] no detection in ±5", flush=True)
            continue
        used_frames.add(det_frame)
        bboxes[det_frame] = det_bbox
        tag = f"{det_frame}" + (f" (target {target})" if det_frame != target else "")
        print(f"  [{tag}] megapose...", flush=True)
        pred = run_megapose_subprocess(det_img, K, det_bbox, H, W,
                                        timeout=args.mp_timeout)
        if pred is None:
            continue
        q = np.asarray(pred["TWO"][0], float)
        t = np.asarray(pred["TWO"][1], float)
        poses[det_frame] = (quat_xyzw_to_R(q), t)
        print(f"  [{det_frame}] OK ({pred['_dt_s']:.1f}s)  "
              f"t=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})", flush=True)

    print(f"pass 1 done in {time.time()-t_p1:.1f}s, "
          f"{len(poses)}/{len(range(0, n_max, args.every))} poses",
          flush=True)
    if not poses:
        raise SystemExit("no poses estimated; aborting")

    # precompute keyframe offsets between megapose t and bbox-derived t
    # so the trajectory matches megapose exactly at keys
    pose_keys = sorted(poses.keys())
    key_offsets = {k: poses[k][1] - bbox_to_t(bboxes[k], K, cup_height_m)
                   for k in pose_keys}
    n_end = min(n_max + 1, n_total)

    # pass 2a: walk every frame, run YOLO, store any new bboxes
    print("pass 2a: per-frame YOLO bbox detection...", flush=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for frame_idx in range(n_end):
        ok, f = cap.read()
        if not ok:
            break
        if frame_idx in bboxes:
            continue
        if not (pose_keys and pose_keys[0] <= frame_idx <= pose_keys[-1]):
            continue
        res = yolo.predict(f, classes=[0], conf=args.conf,
                           verbose=False, device="cpu")[0]
        if res.boxes is not None and len(res.boxes) > 0:
            confs = res.boxes.conf.cpu().numpy()
            bboxes[frame_idx] = res.boxes.xyxy.cpu().numpy()[
                int(np.argmax(confs))].tolist()
    n_yolo_hits = sum(1 for fi in range(n_end) if fi in bboxes)

    # pass 2b: fill bbox holes by lerping between nearest YOLO detections
    bbox_frames = sorted(bboxes.keys())
    interp_bboxes: dict[int, list[float]] = dict(bboxes)
    n_bbox_lerp = 0
    if len(bbox_frames) >= 2:
        for frame_idx in range(bbox_frames[0], bbox_frames[-1] + 1):
            if frame_idx in interp_bboxes:
                continue
            j = next(i for i, k in enumerate(bbox_frames) if k > frame_idx)
            k0, k1 = bbox_frames[j-1], bbox_frames[j]
            b0 = np.asarray(bboxes[k0], float)
            b1 = np.asarray(bboxes[k1], float)
            a = (frame_idx - k0) / (k1 - k0)
            interp_bboxes[frame_idx] = ((1.0 - a) * b0 + a * b1).tolist()
            n_bbox_lerp += 1
    print(f"  YOLO hits: {n_yolo_hits}, bbox-lerp fills: {n_bbox_lerp}",
          flush=True)

    # pass 2c: render & write
    print("pass 2c: rendering overlay video...", flush=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    n_hybrid = 0
    n_lerp = 0
    for frame_idx in range(n_end):
        ok, f = cap.read()
        if not ok:
            break
        bbox_now = interp_bboxes.get(frame_idx)

        pose = None
        if frame_idx in poses:
            pose = poses[frame_idx]
        elif not args.no_interp and pose_keys and pose_keys[0] <= frame_idx <= pose_keys[-1]:
            j = next(i for i, k in enumerate(pose_keys) if k > frame_idx)
            k0, k1 = pose_keys[j-1], pose_keys[j]
            R0, _ = poses[k0]
            R1, _ = poses[k1]
            a = (frame_idx - k0) / (k1 - k0)
            slerp = Slerp([k0, k1], Rotation.from_matrix([R0, R1]))
            R = slerp([frame_idx]).as_matrix()[0]
            if bbox_now is not None:
                off = (1.0 - a) * key_offsets[k0] + a * key_offsets[k1]
                t = bbox_to_t(bbox_now, K, cup_height_m) + off
                n_hybrid += 1
            else:
                t = (1.0 - a) * poses[k0][1] + a * poses[k1][1]
                n_lerp += 1
            pose = (R, t)

        if pose is not None:
            overlay_textured(f, scene, mesh_node, renderer, pose[0], pose[1])
        if bbox_now is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox_now]
            if frame_idx in poses:
                color = (0, 255, 255)  # bright yellow: megapose key
            elif frame_idx in bboxes:
                color = (200, 200, 0)  # dim yellow: per-frame YOLO hit
            else:
                color = (180, 180, 180)  # gray: bbox-lerped
            cv2.rectangle(f, (x1, y1), (x2, y2), color, 2)

        cv2.putText(f, f"{frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(f)
    print(f"  hybrid (bbox+offset) frames: {n_hybrid}, "
          f"pure-lerp fallback: {n_lerp}", flush=True)

    writer.release()
    cap.release()
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
