"""Rerun 3D viewer for the 5-cam live tracker.

Subscribes to the same ZMQ streams `live_track.py` uses, places each cam
in 3D via its calibration, and logs frames + (optionally) live YOLO+KF
results to a local Rerun web viewer. Independent of the DataJoint
`imove_object_tracking` schema — files-only, no DB writes.

Entities logged
---------------
    world/cam_<i>                Transform3D + Pinhole          (static)
    world/cam_<i>/image          JPEG-encoded frame             (per tick)
    world/cam_<i>/image/cup_2d   YOLO mask centroid             (per tick)
    world/cup/centroid           3D KF state (mm)               (per tick)
    world/cup/trail              accumulated 3D trajectory      (per tick)

Run
---
    python viz_rerun.py --check           # just log calibration, no ZMQ
    python viz_rerun.py                   # raw streams only, no detection
    python viz_rerun.py --detect          # run YOLO + 3D KF, log everything
    python viz_rerun.py --connect URL     # attach to an existing viewer

Open `http://127.0.0.1:9090/` to view (or whatever the script prints).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import rerun as rr
import zmq

from kalman_3d import (CamCalib, KalmanFilter3D, load_calibration, project,
                       triangulate_dlt)
from live_track import best_detection, decode, open_sockets
from pseudo_label import CUP_LIKE_CLASSES
from recording.cam_streams import load_streams

DEFAULT_CAL = "data/calibration.toml"
DEFAULT_WEIGHTS = "data/runs/segment/cup_5cam_demo_gen2/weights/best.pt"
MM_PER_M = 1000.0
TRAIL_MAX = 600  # keep the last N KF positions in the trail line
VIZ_MAX_WIDTH = 640       # downscale frames + intrinsics to this width before logging
VIZ_JPEG_QUALITY = 80
VIZ_MAX_FPS_PER_CAM = 15  # throttle per-cam frame logging


def cam_pose_world(cam: CamCalib) -> tuple[np.ndarray, np.ndarray]:
    """Invert world->camera extrinsics to get camera pose in world frame.

    The TOML stores (R, t) such that X_cam = R @ X_world + t. Rerun's
    Transform3D wants the camera's pose IN world coords: R_cw = R^T,
    t_cw = -R^T @ t. Returns (R_cw 3x3, t_cw 3-vec in METERS).
    """
    R_cw = cam.R.T
    t_cw = -R_cw @ cam.t / MM_PER_M
    return R_cw, t_cw


def rot_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> (x, y, z, w) quaternion. Stable branch pick."""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s
    return np.array([qx, qy, qz, qw], dtype=np.float64)


def weiszfeld_geometric_median(points: np.ndarray, max_iter: int = 100,
                                eps: float = 1e-2) -> np.ndarray:
    """Geometric median of N 3D points via Weiszfeld iteration.

    Robust to outliers (vs. plain mean): one wildly off point pulls the mean
    arbitrarily but barely moves the median. Mirror of iMOVE's
    `weiszfeld_geometric_median` from MultiCameraTracking/analysis/camera.py.
    """
    pts = np.asarray(points, dtype=np.float64)
    median = np.nanmean(pts, axis=0)
    for _ in range(max_iter):
        d = np.linalg.norm(pts - median, axis=1)
        w = 1.0 / (d + 1e-9)
        new_median = np.nansum(pts * w[:, None], axis=0) / np.nansum(w)
        if np.linalg.norm(new_median - median) < eps:
            return new_median
        median = new_median
    return median


def robust_triangulate(cam_list: list[CamCalib],
                        pts: list[np.ndarray]
                        ) -> tuple[np.ndarray, float, np.ndarray]:
    """Pairwise-DLT + geometric-median triangulation (Roy et al. 2022).

    Returns (X_world, max_reproj_resid_px, per_cam_weights).

    Each pair of cams is DLT-triangulated independently → N×(N-1)/2 candidate
    3D points. The geometric median of those candidates is the robust 3D
    estimate — an outlier cam (false detection) drags only the pairs it's
    in, which become outliers and the median ignores them. With only 2 cams,
    falls back to plain DLT.
    """
    n = len(cam_list)
    if n < 2:
        raise ValueError("need >=2 cams")
    if n == 2:
        X = triangulate_dlt(cam_list, pts)
        return X, _max_reproj_px(cam_list, pts, X), np.ones(2)

    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            try:
                Xij = triangulate_dlt([cam_list[i], cam_list[j]],
                                       [pts[i], pts[j]])
                candidates.append(Xij)
            except Exception:
                continue
    if not candidates:
        return triangulate_dlt(cam_list, pts), 1e6, np.ones(n)
    cand_arr = np.asarray(candidates)
    X = weiszfeld_geometric_median(cand_arr)
    # Per-cam weight: how well its pair-triangulations agree with the median.
    # iMOVE uses exp(-err²/σ²) with σ=150 mm; same here.
    sigma_mm = 150.0
    pair_dist = np.linalg.norm(cand_arr - X, axis=1)
    weights = np.zeros(n)
    cnt = np.zeros(n)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            w = float(np.exp(-(pair_dist[k] ** 2) / (sigma_mm ** 2)))
            weights[i] += w; weights[j] += w
            cnt[i] += 1; cnt[j] += 1
            k += 1
    weights = weights / np.maximum(cnt, 1)
    return X, _max_reproj_px(cam_list, pts, X), weights


def _max_reproj_px(cam_list: list[CamCalib], pts: list[np.ndarray],
                    X: np.ndarray) -> float:
    resids = []
    for c, p in zip(cam_list, pts):
        uv, ok = project(c, X)
        if ok:
            resids.append(float(np.linalg.norm(uv - p)))
    return max(resids) if resids else 1e6


def viz_scale_for(cam: CamCalib) -> float:
    """Image-side downscale factor used everywhere we log this cam."""
    return min(1.0, VIZ_MAX_WIDTH / cam.size[0])


def log_static_cameras(cams: dict[str, CamCalib]) -> None:
    """One-time: place each cam in 3D + register its pinhole intrinsics.

    Pinhole resolution + intrinsics are logged at the downscaled viz size
    so the (downscaled) image we log per-frame and any 2D overlays (which
    we also scale) all share the same coordinate system.
    """
    rr.log("world", rr.ViewCoordinates.RDF, static=True)
    for name, cam in cams.items():
        base = f"world/{name}"
        R_cw, t_cw = cam_pose_world(cam)
        rr.log(
            base,
            rr.Transform3D(translation=t_cw,
                           rotation=rr.Quaternion(xyzw=rot_to_quat_xyzw(R_cw))),
            static=True,
        )
        s = viz_scale_for(cam)
        W = int(round(cam.size[0] * s))
        H = int(round(cam.size[1] * s))
        rr.log(
            f"{base}/image",
            rr.Pinhole(
                resolution=[W, H],
                focal_length=[float(cam.K[0, 0]) * s, float(cam.K[1, 1]) * s],
                principal_point=[float(cam.K[0, 2]) * s, float(cam.K[1, 2]) * s],
                camera_xyz=rr.ViewCoordinates.RDF,
            ),
            static=True,
        )


def log_frame_downscaled(name: str, frame_bgr: np.ndarray, scale: float) -> None:
    """Downscale by `scale` and JPEG-re-encode before logging."""
    if scale < 1.0:
        h, w = frame_bgr.shape[:2]
        frame_bgr = cv2.resize(frame_bgr,
                                (int(round(w * scale)), int(round(h * scale))),
                                interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame_bgr,
                            [cv2.IMWRITE_JPEG_QUALITY, VIZ_JPEG_QUALITY])
    if not ok:
        return
    rr.log(f"world/{name}/image",
           rr.EncodedImage(contents=buf.tobytes(), media_type="image/jpeg"))


def log_kf_state(kf: KalmanFilter3D, trail: list[np.ndarray]) -> None:
    if not kf.initialized:
        return
    X_m = kf.x[:3] / MM_PER_M
    rr.log("world/cup/centroid",
           rr.Points3D(X_m.reshape(1, 3), colors=[[255, 80, 80, 255]], radii=[0.02]))
    trail.append(X_m.copy())
    if len(trail) > TRAIL_MAX:
        del trail[: len(trail) - TRAIL_MAX]
    if len(trail) >= 2:
        rr.log("world/cup/trail",
               rr.LineStrips3D([np.asarray(trail)], colors=[[255, 80, 80, 180]], radii=[0.003]))


def log_cup_2d(cam_name: str, cx: float, cy: float, accepted: bool,
               scale: float) -> None:
    color = [0, 220, 0, 255] if accepted else [200, 60, 60, 255]
    rr.log(f"world/{cam_name}/image/cup_2d",
           rr.Points2D([[cx * scale, cy * scale]], colors=[color], radii=[6.0]))


def run_check(cams: dict[str, CamCalib], hold_s: float = 3600.0) -> None:
    """No ZMQ, no YOLO — just log the static cameras + world reference frame
    so you can visually confirm the calibration in Rerun (frustums pointing
    inward, table at origin).
    """
    log_static_cameras(cams)
    # Origin reference: 100mm axes + grid of 1mm dots at world origin.
    axis_len = 0.1  # 100 mm
    rr.log("world/origin/axes",
           rr.Arrows3D(
               vectors=[[axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]],
               origins=[[0, 0, 0]] * 3,
               colors=[[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]],
           ),
           static=True)
    rr.log("world/origin/marker",
           rr.Points3D([[0, 0, 0]], colors=[[255, 255, 255, 255]], radii=[0.01]),
           static=True)
    print(f"viz_rerun --check: logged {len(cams)} cams + origin reference.")
    print(f"viz_rerun --check: open the viewer and confirm frustums look right.")
    print(f"viz_rerun --check: ctrl-c to exit (auto-exit in {hold_s:.0f}s).")
    try:
        time.sleep(hold_s)
    except KeyboardInterrupt:
        print("\nviz_rerun: bye")


def run_stream_only(cams: dict[str, CamCalib], socks: dict[str, zmq.Socket]) -> None:
    """No detection — just push frames + static cam frustums.

    Useful as the first sanity check: open the viewer, see the 5 frustums
    line up around the table, scrub the timeline.
    """
    print("viz_rerun: stream-only mode. ctrl-c to stop.")
    poller = zmq.Poller()
    for s in socks.values():
        poller.register(s, zmq.POLLIN)
    log_static_cameras(cams)
    scales = {n: viz_scale_for(c) for n, c in cams.items()}
    min_dt = 1.0 / VIZ_MAX_FPS_PER_CAM
    last_log_t = {n: 0.0 for n in socks}
    frame_idx = 0
    t0 = time.time()
    try:
        while True:
            events = dict(poller.poll(timeout=10))
            got_any = False
            now = time.time()
            for name, s in socks.items():
                if s not in events:
                    continue
                raw = s.recv(flags=zmq.NOBLOCK)
                if now - last_log_t[name] < min_dt:
                    continue  # throttle
                frame = decode(raw)
                if frame is None:
                    continue
                rr.set_time("frame_idx", sequence=frame_idx)
                rr.set_time("time", duration=now - t0)
                log_frame_downscaled(name, frame, scales.get(name, 1.0))
                last_log_t[name] = now
                got_any = True
            if got_any:
                frame_idx += 1
    except KeyboardInterrupt:
        print("\nviz_rerun: bye")


def run_detect(cams: dict[str, CamCalib], socks: dict[str, zmq.Socket],
               weights: Path, conf: float, classes,
               meas_noise_mm: float = 30.0,
               resid_to_mm: float = 2.0,
               reinit_after_s: float = 1.5) -> None:
    """Full pipeline: YOLO + 3D KF on each cam, log everything to Rerun.

    Each iteration: collect all cams' mask centroids, triangulate once to a
    single 3D point, do ONE `kf.update_3d` with measurement noise driven by
    the triangulation residual (cams agree → low noise → high gain; cams
    disagree → high noise → low gain). This avoids the asymmetry of the
    per-cam sequential 2D EKF, where 5 cams agreeing on a translation
    produced a small response but 5 cams disagreeing during rotation
    produced wild jumps.
    """
    from ultralytics import YOLO
    print(f"viz_rerun: loading {weights} ...")
    model = YOLO(str(weights))
    print("viz_rerun: tracking. ctrl-c to stop.")

    poller = zmq.Poller()
    for s in socks.values():
        poller.register(s, zmq.POLLIN)
    log_static_cameras(cams)
    scales = {n: viz_scale_for(c) for n, c in cams.items()}
    min_dt = 1.0 / VIZ_MAX_FPS_PER_CAM
    last_log_t = {n: 0.0 for n in socks}

    kf = KalmanFilter3D()
    last_accepted_t = 0.0
    trail: list[np.ndarray] = []
    frame_idx = 0
    t0 = time.time()

    try:
        while True:
            events = dict(poller.poll(timeout=10))
            now = time.time()
            rr.set_time("frame_idx", sequence=frame_idx)
            rr.set_time("time", duration=now - t0)

            if kf.initialized and (now - last_accepted_t) > reinit_after_s:
                print(f"reinit: {now - last_accepted_t:.1f}s without an accept "
                      "— dropping KF state")
                kf = KalmanFilter3D()
                trail.clear()

            dets_this_iter: dict[str, tuple[float, float]] = {}
            anything = False
            for name, s in socks.items():
                if s not in events:
                    continue
                raw = s.recv(flags=zmq.NOBLOCK)
                anything = True
                frame = decode(raw)
                if frame is None:
                    continue
                cam = cams.get(name)
                scale = scales.get(name, 1.0)
                if now - last_log_t[name] >= min_dt:
                    log_frame_downscaled(name, frame, scale)
                    last_log_t[name] = now
                if cam is None:
                    continue
                det = best_detection(model, frame, conf=conf, classes=classes)
                if det is None:
                    continue
                _cx, _cy, bbox, _poly, _area = det
                # Bbox center rather than mask polygon centroid: more stable
                # when a hand occludes part of the cup (mask centroid shifts
                # toward the visible pixels; bbox bounds stay anchored to the
                # cup's overall extent in the image).
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                dets_this_iter[name] = (cx, cy)

            # Need ≥2 cams agreeing on a 2D centroid to make a 3D measurement.
            if len(dets_this_iter) >= 2:
                cam_names = list(dets_this_iter)
                cam_list = [cams[n] for n in cam_names]
                pts = [np.array(dets_this_iter[n]) for n in cam_names]
                X_meas, max_resid_px, weights = robust_triangulate(
                    cam_list, pts)
                sigma_mm = meas_noise_mm + resid_to_mm * max_resid_px
                R3 = (sigma_mm ** 2) * np.eye(3)

                if not kf.initialized:
                    kf.init(X_meas, t=now)
                    last_accepted_t = now
                    print(f"init: KF seeded at {X_meas.round(1)} mm from "
                          f"{cam_names} (resid {max_resid_px:.1f}px, "
                          f"weights {weights.round(2).tolist()})")
                    accepted = True
                else:
                    accepted, _d2 = kf.update_3d(
                        X_meas, t=now, R=R3, gate=float("inf"))
                    if accepted:
                        last_accepted_t = now
                # Per-cam dot color = green if that cam contributed (high
                # weight); dim red if outlier (low weight).
                for k, n in enumerate(cam_names):
                    cx, cy = dets_this_iter[n]
                    is_inlier = weights[k] > 0.5
                    log_cup_2d(n, cx, cy, accepted=is_inlier and accepted,
                               scale=scales.get(n, 1.0))
            else:
                # Only 0 or 1 cam — no 3D update. Still draw single-cam dots.
                for n, (cx, cy) in dets_this_iter.items():
                    log_cup_2d(n, cx, cy, accepted=False,
                               scale=scales.get(n, 1.0))

            if anything:
                if kf.initialized:
                    kf.predict(now)
                    log_kf_state(kf, trail)
                frame_idx += 1
    except KeyboardInterrupt:
        print("\nviz_rerun: bye")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cal", default=DEFAULT_CAL,
                    help="aniposelib calibration TOML")
    ap.add_argument("--host", default="127.0.0.1",
                    help="ZMQ host serving cam frames")
    ap.add_argument("--check", action="store_true",
                    help="No ZMQ/YOLO — log static cameras + origin only")
    ap.add_argument("--detect", action="store_true",
                    help="Run YOLO + 3D KF (default: stream frames only)")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS,
                    help="YOLO-seg .pt (only used with --detect)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--meas-noise-mm", type=float, default=30.0,
                    help="3D measurement noise floor (mm). KF uses "
                         "sigma = meas_noise_mm + resid_to_mm * residual_px.")
    ap.add_argument("--resid-to-mm", type=float, default=2.0,
                    help="Pixel-residual-to-3D-noise multiplier (mm/px). "
                         "Bigger = more dampening when cams disagree.")
    ap.add_argument("--classes", default="cup-like",
                    help="'cup-like' | 'all' | comma-separated ints")
    ap.add_argument("--connect", default=None,
                    help="Connect to an existing Rerun viewer at this gRPC URL "
                         "(e.g. rerun+http://127.0.0.1:9876/proxy). Default: "
                         "start a new web viewer on ports 9876/9090.")
    ap.add_argument("--web-port", type=int, default=9090)
    ap.add_argument("--grpc-port", type=int, default=9876)
    args = ap.parse_args()

    rr.init("object_tracking", recording_id=uuid4())
    if args.connect:
        rr.connect_grpc(args.connect)
        print(f"viz_rerun: connected to existing viewer at {args.connect}")
    else:
        import urllib.parse
        server_uri = rr.serve_grpc(grpc_port=args.grpc_port)
        rr.serve_web_viewer(web_port=args.web_port, open_browser=False,
                            connect_to=server_uri)
        encoded = urllib.parse.quote(
            f"rerun+http://127.0.0.1:{args.grpc_port}/proxy", safe="")
        print(f"viz_rerun: open http://127.0.0.1:{args.web_port}/?url={encoded}")

    if args.classes == "all":
        classes = None
    elif args.classes == "cup-like":
        classes = CUP_LIKE_CLASSES
    else:
        classes = [int(x) for x in args.classes.split(",")]

    cams = load_calibration(args.cal)
    print(f"calibrated: {list(cams)}")

    if args.check:
        run_check(cams)
        return

    streams = load_streams()
    print(f"streams: {list(streams)}")
    ctx, socks = open_sockets(streams, args.host)
    try:
        if args.detect:
            run_detect(cams, socks, Path(args.weights), args.conf, classes,
                       meas_noise_mm=args.meas_noise_mm,
                       resid_to_mm=args.resid_to_mm)
        else:
            run_stream_only(cams, socks)
    finally:
        for s in socks.values():
            s.close()
        ctx.term()


if __name__ == "__main__":
    main()
