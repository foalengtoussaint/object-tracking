"""Offline MegaPose runner — subprocess-per-frame architecture.

Why subprocess: MegaPose's panda3d_batch_renderer deadlocks when reused
across multiple inferences in a single process (torch.multiprocessing
pipe issue). The standalone `run_inference_on_example.py` works because
it's one-shot. We call it as a subprocess per detected frame.

Runs in the `object_tracking` env (YOLO + rerun lives there). The
subprocess invokes `conda run -n megapose ...` for each MegaPose call.

Slow: ~5-10s per frame because the megapose model loads each time. For
50 frames with --every 10 = 5 megapose calls = ~50s total.

Usage:
    conda activate object_tracking
    PYTHONUNBUFFERED=1 python pose_6d/offline_megapose.py \\
        --clip data/clips/test/cam_1_20260526_163138.mp4 \\
        --cam cam_1 \\
        --every 10 \\
        --max-frames 50 \\
        --out data/recordings/cup_megapose.rrd
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import rerun as rr
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


def log_mesh_static(mesh: trimesh.Trimesh, entity: str) -> None:
    verts = np.asarray(mesh.vertices, np.float32)
    faces = np.asarray(mesh.faces, np.uint32)
    rr.log(entity, rr.Mesh3D(vertex_positions=verts, triangle_indices=faces,
                              albedo_factor=[120, 200, 120, 200]), static=True)


def run_megapose_subprocess(rgb_bgr: np.ndarray, K: np.ndarray, bbox: list[float],
                            H: int, W: int, timeout: int = 120) -> dict | None:
    """Write frame+bbox+K to example dir, run megapose, return pose JSON."""
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
    import os as _os
    clean_env = {k: v for k, v in _os.environ.items()
                 if not k.startswith("CONDA_") and k not in {
                     "PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH",
                     "VIRTUAL_ENV", "_CE_M", "_CE_CONDA"}}
    clean_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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
        print(f"  megapose ran but no output file", flush=True)
        return None
    pred = json.load(open(out_json))[0]
    pred["_dt_s"] = dt
    return pred


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
    ap.add_argument("--mp-timeout", type=int, default=120)
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

    print("loading YOLO on CPU (frees GPU for megapose)...", flush=True)
    yolo = YOLO(args.weights)
    yolo.to("cpu")

    print("loading mesh for rerun viz...", flush=True)
    rer_mesh = trimesh.load(args.mesh)
    if isinstance(rer_mesh, trimesh.Scene):
        rer_mesh = trimesh.util.concatenate(list(rer_mesh.geometry.values()))

    rr.init("megapose_offline", recording_id=uuid4())
    rr.save(args.out)
    rr.log("world", rr.ViewCoordinates.RDF, static=True)
    rr.log(f"world/{args.cam}", rr.Transform3D(translation=[0,0,0]), static=True)
    rr.log(f"world/{args.cam}",
           rr.Pinhole(image_from_camera=K, resolution=[W, H],
                      camera_xyz=rr.ViewCoordinates.RDF), static=True)
    log_mesh_static(rer_mesh, "world/cup_6d")
    print(f"recording to {args.out}", flush=True)

    n_inferred = 0
    n_detected = 0
    frame_idx = 0
    t_start = time.time()
    poses: list[tuple[int, np.ndarray, np.ndarray]] = []  # (frame, t_m, q_xyzw)
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and frame_idx >= args.max_frames):
            break

        t_sec = frame_idx / fps
        rr.set_time("time", duration=t_sec)
        rr.set_time("frame", sequence=frame_idx)

        ok_jpg, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        rr.log(f"world/{args.cam}/image",
               rr.EncodedImage(contents=buf.tobytes(), media_type="image/jpeg"))

        if frame_idx % args.every == 0:
            res = yolo.predict(frame, classes=[0], conf=args.conf, verbose=False, device="cpu")[0]
            boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0,4))
            if len(boxes):
                n_detected += 1
                confs = res.boxes.conf.cpu().numpy()
                bbox = boxes[int(np.argmax(confs))].tolist()
                rr.log(f"world/{args.cam}/image/bbox",
                       rr.Boxes2D(array=[bbox], array_format=rr.Box2DFormat.XYXY,
                                  colors=[[255,255,0]]))
                print(f"  [{frame_idx}/{n_total}] running megapose subprocess...", flush=True)
                pred = run_megapose_subprocess(frame, K, bbox, H, W,
                                                timeout=args.mp_timeout)
                if pred is not None:
                    n_inferred += 1
                    q_xyzw = np.asarray(pred["TWO"][0], float)
                    t_m = np.asarray(pred["TWO"][1], float)
                    poses.append((frame_idx, t_m, q_xyzw))
                    if len(poses) == 1:
                        rr.log("world/cup_6d",
                               rr.Transform3D(translation=t_m.tolist(),
                                              rotation=rr.Quaternion(xyzw=q_xyzw.tolist())))
                    else:
                        f0, t0, q0 = poses[-2]
                        f1, t1, q1 = poses[-1]
                        slerp = Slerp([f0, f1], Rotation.from_quat([q0, q1]))
                        for fi in range(f0 + 1, f1 + 1):
                            a = (fi - f0) / (f1 - f0)
                            ti = (1.0 - a) * t0 + a * t1
                            qi = slerp([fi]).as_quat()[0]
                            rr.set_time("frame", sequence=fi)
                            rr.set_time("time", duration=fi / fps)
                            rr.log("world/cup_6d",
                                   rr.Transform3D(translation=ti.tolist(),
                                                  rotation=rr.Quaternion(xyzw=qi.tolist())))
                        rr.set_time("frame", sequence=frame_idx)
                        rr.set_time("time", duration=frame_idx / fps)
                    print(f"  [{frame_idx}] OK ({pred['_dt_s']:.1f}s)  "
                          f"t={[round(v,3) for v in t_m]}", flush=True)

        frame_idx += 1

    cap.release()
    print(f"\ndone: {frame_idx} frames, {n_detected} detections, "
          f"{n_inferred} pose estimates, "
          f"total {time.time()-t_start:.1f}s", flush=True)
    print(f"rrd: {args.out}", flush=True)


if __name__ == "__main__":
    main()
