"""Live 5-camera object tracking with a shared 3D Kalman filter.

Subscribes to all ZMQ streams from `recording/cam_server.py`, runs a YOLO-seg
teacher per cam, fuses the per-cam centroids into a 6D world-frame state
(via `kalman_3d.KalmanFilter3D`), and renders a 5-tile mosaic:

- green box/mask on each cam that accepted a fresh detection this frame
- yellow box (translated last-accepted mask) when the cam missed but the
  shared 3D state still says the object is in view

Two modes:
    python live_track.py --check                # one-shot cal sanity check
    python live_track.py                        # full live tracking + viz

`--check` uses an ArUco board on the table: detects markers in every cam,
triangulates each marker corner from the cams that see it, then projects
back. It prints per-cam mean/max pixel reprojection error and overlays
green (detected) vs red (re-projected) corners. Calibration is good when
the residuals are <~5 px and the red dots sit on top of the green ones in
every cam.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import zmq

from kalman_3d import (CamCalib, KalmanFilter3D, load_calibration, project,
                       triangulate_dlt)
from pseudo_label import CUP_LIKE_CLASSES
from recording.cam_streams import load_streams

DEFAULT_CAL = "data/calibration.toml"
DEFAULT_WEIGHTS = "data/runs/segment/cup_5cam_demo_gen2/weights/best.pt"
MOSAIC_TILE = (640, 360)   # (W, H) per tile in the mosaic
MOSAIC_COLS = 3


def predicted_region(kf: KalmanFilter3D, cam: CamCalib,
                     last_bbox: np.ndarray | None,
                     margin: float = 1.6,
                     fallback_half: float = 80.0,
                     ) -> tuple[float, float, float, float] | None:
    """KF-predicted accept region in `cam`'s image plane.

    Returns (pred_cx, pred_cy, half_w, half_h). Centered on the projected
    KF state, sized from this cam's last accepted bbox + margin. No
    shape model — just translates the most recent observation to where
    the 3D centroid now projects.
    """
    if not kf.initialized:
        return None
    uv, ok = kf.project_into(cam)
    if not ok:
        return None
    pred_cx, pred_cy = float(uv[0]), float(uv[1])
    if last_bbox is None:
        return pred_cx, pred_cy, fallback_half, fallback_half
    half_w = (last_bbox[2] - last_bbox[0]) * margin / 2.0
    half_h = (last_bbox[3] - last_bbox[1]) * margin / 2.0
    return pred_cx, pred_cy, half_w, half_h


def _polygon_centroid_and_area(poly_pix: np.ndarray
                                ) -> tuple[float, float, float]:
    """Centroid (cx, cy) and area (px²) of a 2D polygon by the shoelace
    formula. `poly_pix` is Nx2 in pixel coords. Falls back to mean if the
    polygon is degenerate (area <= 0).
    """
    x = poly_pix[:, 0]
    y = poly_pix[:, 1]
    # Shoelace: signed area = ½ Σ (x_i · y_{i+1} − x_{i+1} · y_i)
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area2 = float(np.sum(cross))
    area_px = abs(area2) * 0.5
    if area_px < 1.0:
        return float(x.mean()), float(y.mean()), 0.0
    cx = float(np.sum((x + np.roll(x, -1)) * cross)) / (3.0 * area2)
    cy = float(np.sum((y + np.roll(y, -1)) * cross)) / (3.0 * area2)
    return cx, cy, area_px


def best_detection(model, frame, conf: float, classes):
    """Run model on one frame, return the highest-confidence detection as
    (cx, cy, bbox_xyxy, polygon_xyn, mask_area_px), or None.

    cx, cy come from the polygon centroid when a mask is available (more
    accurate than the bbox center), else from the bbox center. mask_area_px
    is the shoelace area of the polygon in pixel², or 0 if no mask.
    """
    res = model(frame, classes=classes, conf=conf, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    confs = res.boxes.conf.cpu().numpy()
    i = int(np.argmax(confs))
    b = res.boxes.xyxy[i].cpu().numpy()
    poly = None
    mask_area_px = 0.0
    cx = float((b[0] + b[2]) / 2)
    cy = float((b[1] + b[3]) / 2)
    if res.masks is not None:
        poly = res.masks.xyn[i]    # normalized 0..1 polygon
        h, w = frame.shape[:2]
        poly_pix = poly * np.array([w, h], dtype=np.float32)
        cx, cy, mask_area_px = _polygon_centroid_and_area(poly_pix)
    return cx, cy, b, poly, mask_area_px


def draw_overlay(tile: np.ndarray, det, predicted, in_front_proj):
    """Draw boxes / projected predictions onto a tile (in-place).

    det        — (cx, cy, bbox, poly, mask_area_px) for an accepted detection, or None
    predicted  — (pred_uv, last_bbox_at_capture, last_poly) when the cam
                 missed but the 3D state projects in-frame, else None
    in_front_proj — (uv, in_front) of the 3D state into this cam (for a
                 small reference marker)
    """
    h, w = tile.shape[:2]
    if det is not None:
        cx, cy, b, poly = det[0], det[1], det[2], det[3]
        x0, y0, x1, y1 = (int(round(v)) for v in b)
        cv2.rectangle(tile, (x0, y0), (x1, y1), (0, 220, 0), 2)
        if poly is not None and len(poly):
            pts = (poly * [w, h]).astype(np.int32)
            cv2.polylines(tile, [pts], True, (0, 220, 0), 1)
        cv2.circle(tile, (int(cx), int(cy)), 4, (0, 220, 0), -1)
    elif predicted is not None:
        pred_uv, last_bbox, last_poly = predicted
        cx, cy = pred_uv
        if last_bbox is not None:
            bw = (last_bbox[2] - last_bbox[0]) / 2
            bh = (last_bbox[3] - last_bbox[1]) / 2
            cv2.rectangle(tile,
                          (int(cx - bw), int(cy - bh)),
                          (int(cx + bw), int(cy + bh)),
                          (0, 220, 220), 2)
        if last_poly is not None and len(last_poly):
            pts = (last_poly * [w, h]).astype(np.float32)
            # translate polygon so its centroid lands on (cx, cy)
            centroid = pts.mean(axis=0)
            pts = (pts + np.array([cx, cy]) - centroid).astype(np.int32)
            cv2.polylines(tile, [pts], True, (0, 220, 220), 1)
        cv2.drawMarker(tile, (int(cx), int(cy)), (0, 220, 220),
                       cv2.MARKER_CROSS, 14, 2)
    elif in_front_proj is not None:
        uv, ok = in_front_proj
        if ok:
            cv2.drawMarker(tile, (int(uv[0]), int(uv[1])), (255, 100, 100),
                           cv2.MARKER_TILTED_CROSS, 10, 1)


def build_mosaic(tiles: dict[str, np.ndarray], cam_order: list[str],
                 size=MOSAIC_TILE, cols=MOSAIC_COLS) -> np.ndarray:
    tw, th = size
    n = len(cam_order)
    rows = (n + cols - 1) // cols
    out = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    for i, name in enumerate(cam_order):
        r, c = i // cols, i % cols
        f = tiles.get(name)
        if f is None:
            cv2.putText(out, f"{name}: no frame", (c * tw + 20, r * th + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1)
            continue
        small = cv2.resize(f, (tw, th))
        out[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = small
        cv2.putText(out, name, (c * tw + 8, r * th + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def open_sockets(streams: dict[str, int], host: str
                 ) -> tuple[zmq.Context, dict[str, zmq.Socket]]:
    ctx = zmq.Context.instance()
    socks: dict[str, zmq.Socket] = {}
    for name, port in streams.items():
        s = ctx.socket(zmq.SUB)
        s.setsockopt(zmq.CONFLATE, 1)         # always drop to latest frame
        s.setsockopt_string(zmq.SUBSCRIBE, "")
        s.connect(f"tcp://{host}:{port}")
        socks[name] = s
    return ctx, socks


def decode(raw: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)


ARUCO_DICTS = [
    ("DICT_4X4_250", cv2.aruco.DICT_4X4_250),    # iMOVE Charuco default
    ("DICT_5X5_250", cv2.aruco.DICT_5X5_250),
    ("DICT_6X6_250", cv2.aruco.DICT_6X6_250),
    ("DICT_4X4_50",  cv2.aruco.DICT_4X4_50),
    ("DICT_5X5_50",  cv2.aruco.DICT_5X5_50),
]


def detect_aruco_any(frame: np.ndarray
                     ) -> tuple[str, dict[int, np.ndarray]]:
    """Try each common dictionary; return (dict_name, {id: 4x2 corners}) for
    whichever gives detections. Empty dict if none match.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for name, dict_id in ARUCO_DICTS:
        ad = cv2.aruco.getPredefinedDictionary(dict_id)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(ad, params)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None and len(ids):
            return name, {int(i): c.reshape(4, 2) for i, c in zip(ids.flatten(), corners)}
    return "", {}


def run_check(cams: dict[str, CamCalib], socks: dict[str, zmq.Socket],
              timeout_s: float = 5.0) -> None:
    """Detect ArUco markers in every cam, triangulate each corner from cams
    that see it in common, project back. Print per-cam reprojection error and
    show an overlay (green=detected, red=re-projected).
    """
    print(f"check: collecting one frame per cam (up to {timeout_s}s)...")
    poller = zmq.Poller()
    for s in socks.values():
        poller.register(s, zmq.POLLIN)

    frames: dict[str, np.ndarray] = {}
    t_start = time.time()
    while time.time() - t_start < timeout_s and len(frames) < len(socks):
        events = dict(poller.poll(timeout=200))
        for name, s in socks.items():
            if s in events and name not in frames:
                f = decode(s.recv(flags=zmq.NOBLOCK))
                if f is not None:
                    frames[name] = f
    print(f"check: got frames from {sorted(frames)}")

    # Detect markers per cam — pick the dictionary by majority vote
    detections: dict[str, dict[int, np.ndarray]] = {}
    dict_used: dict[str, str] = {}
    for name, f in frames.items():
        d_name, d = detect_aruco_any(f)
        detections[name] = d
        dict_used[name] = d_name
        print(f"  {name}: dict={d_name or 'NONE'}  markers={sorted(d.keys())}")

    # marker_id -> [(cam_name, corner_idx, (u, v)), ...] for triangulation
    by_id: dict[int, dict[str, np.ndarray]] = {}
    for cam_name, dets in detections.items():
        if cam_name not in cams:
            continue
        for mid, corners in dets.items():
            by_id.setdefault(mid, {})[cam_name] = corners  # 4x2

    shared = [mid for mid, seen in by_id.items() if len(seen) >= 2]
    if not shared:
        print("check: FAIL — no marker is visible in >=2 calibrated cams.")
        return

    print(f"check: triangulating {len(shared)} shared marker(s): {shared}")
    per_cam_errs: dict[str, list[float]] = {n: [] for n in frames}
    triangulated: list[tuple[int, int, np.ndarray]] = []  # (mid, corner_idx, X)

    for mid in shared:
        seen = by_id[mid]
        cam_keys = list(seen.keys())
        cam_calibs = [cams[n] for n in cam_keys]
        for ci in range(4):
            pts = [seen[n][ci] for n in cam_keys]
            X = triangulate_dlt(cam_calibs, pts)
            triangulated.append((mid, ci, X))
            # Reproject into every detecting cam (incl. those not used)
            for n, corners in seen.items():
                uv, ok = project(cams[n], X)
                if ok:
                    err = float(np.linalg.norm(uv - corners[ci]))
                    per_cam_errs[n].append(err)

    print()
    print(f"{'cam':<8} {'n_pts':>6} {'mean':>8} {'median':>8} {'max':>8}")
    for n in sorted(per_cam_errs):
        e = np.array(per_cam_errs[n])
        if not len(e):
            continue
        print(f"{n:<8} {len(e):>6} {e.mean():>7.2f}px {np.median(e):>7.2f}px {e.max():>7.2f}px")

    # Render mosaic: detected corners green, reprojected red
    tiles = {}
    for name, f in frames.items():
        tile = f.copy()
        for mid, corners in detections.get(name, {}).items():
            for c in corners:
                cv2.circle(tile, (int(c[0]), int(c[1])), 4, (0, 220, 0), -1)
        if name in cams:
            for mid, ci, X in triangulated:
                uv, ok = project(cams[name], X)
                if ok:
                    cv2.drawMarker(tile, (int(uv[0]), int(uv[1])),
                                   (0, 0, 255), cv2.MARKER_CROSS, 12, 1)
        tiles[name] = tile

    mosaic = build_mosaic(tiles, sorted(socks.keys()))
    cv2.putText(mosaic, "GREEN=detected corner  RED=reprojected from 3D",
                (10, mosaic.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("live_track --check (ArUco)", mosaic)
    print("check: press any key in the window to exit")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_live(model, cams: dict[str, CamCalib], socks: dict[str, zmq.Socket],
             conf: float, classes, show_every_ms: int = 33,
             reinit_after_s: float = 1.5,
             record_path: Path | None = None,
             duration_s: float | None = None) -> None:
    """Run the live 3D tracker.

    reinit_after_s — if no cam has accepted a detection (or seeded the KF)
    for this many seconds, the KF state is considered stale and dropped, so
    the next 2 fresh detections re-triangulate from scratch. Prevents the
    classic constant-velocity-drift trap where the cup leaves all cams,
    state coasts to nonsense, then every reappearing detection fails the
    Mahalanobis gate forever.
    """
    print("live: tracking — press q in the window to quit")
    poller = zmq.Poller()
    for s in socks.values():
        poller.register(s, zmq.POLLIN)

    # Optional mosaic recorder: open lazily on the first rendered frame so
    # we know the mosaic resolution. imshow stays active throughout so the
    # user can still see the feed. We write at a fixed nominal RECORD_FPS
    # and pad each newly-rendered frame to fill its wall-clock time slot
    # — keeps playback in real-time even when the render loop runs slower
    # than 30 Hz.
    RECORD_FPS = 30
    recorder = None
    record_start_t: float | None = None
    frames_written = 0
    if record_path is not None:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"recording: will write {record_path} ({RECORD_FPS} fps)"
              + (f" for {duration_s:.1f}s" if duration_s else ""))

    kf = KalmanFilter3D()
    cam_names = sorted(socks.keys())
    last_frame: dict[str, np.ndarray] = {}
    last_bbox: dict[str, np.ndarray] = {}     # per-cam, last accepted xyxy
    last_poly: dict[str, np.ndarray] = {}     # per-cam, last accepted polygon

    # Init buffer: collect first detections from any 2+ cams to triangulate
    init_dets: dict[str, tuple] = {}    # name -> (cx, cy, bbox)
    last_accepted_t = 0.0      # wall-clock time of the most recent accept
    t_render_last = 0.0

    # Rolling 1-second per-cam accept / reject counters for the overlay.
    accept_cnt: dict[str, int] = {n: 0 for n in cam_names}
    reject_cnt: dict[str, int] = {n: 0 for n in cam_names}
    last_d2: dict[str, float] = {n: 0.0 for n in cam_names}
    t_counter_reset = 0.0


    while True:
        events = dict(poller.poll(timeout=10))
        now = time.time()
        per_frame_overlay = {}     # name -> ('det'|'predicted'|'proj', payload)

        # Drop the KF if it has been coasting too long with no accepts. The
        # next 2 detections from any cams will then re-triangulate fresh
        # and seed a new state, regardless of where the stale state was.
        if kf.initialized and (now - last_accepted_t) > reinit_after_s:
            print(f"reinit: KF coasted {now - last_accepted_t:.1f}s with no "
                  f"accepts — dropping state and re-seeding from next dets")
            kf = KalmanFilter3D()
            init_dets = {}

        for name, s in socks.items():
            if s not in events:
                continue
            frame = decode(s.recv(flags=zmq.NOBLOCK))
            if frame is None:
                continue
            last_frame[name] = frame
            if name not in cams:
                continue
            cam = cams[name]
            det = best_detection(model, frame, conf=conf, classes=classes)
            if det is None:
                continue
            cx, cy, bbox, poly, mask_area_px = det
            bbox_w = float(bbox[2] - bbox[0])

            if not kf.initialized:
                init_dets[name] = (cx, cy, bbox)
                if len(init_dets) >= 2:
                    pick = list(init_dets.keys())
                    pts = [np.array(init_dets[n][:2]) for n in pick]
                    X0 = triangulate_dlt([cams[n] for n in pick], pts)
                    kf.init(X0, t=now)
                    last_accepted_t = now
                    init_dets = {}
                    print(f"init: KF seeded at {X0.round(1)} mm from {pick}")
                per_frame_overlay[name] = ("det", det)
                last_bbox[name] = bbox
                last_poly[name] = poly
                continue

            # Geometric gate: is the YOLO mask centroid inside a region
            # sized from this cam's last accepted bbox (translated to the
            # KF-projected position, + 1.6× margin)?
            region = predicted_region(kf, cam, last_bbox.get(name),
                                       margin=1.6)
            in_region = (region is None
                          or (abs(cx - region[0]) <= region[2]
                              and abs(cy - region[1]) <= region[3]))

            if not in_region:
                accepted, d2 = False, 0.0
            else:
                # 2D EKF update on the mask centroid. The Jacobian only
                # observes image-plane components, so single-cam updates
                # constrain 2 of 3 dof and don't fake depth. Depth
                # observability comes from multiple cams' perpendicular
                # constraints combining across the same frames.
                accepted, d2 = kf.update(
                    cam, np.array([cx, cy]), t=now, gate=float("inf"))

            last_d2[name] = d2
            if accepted:
                accept_cnt[name] += 1
                last_accepted_t = now
                last_bbox[name] = bbox
                last_poly[name] = poly
                per_frame_overlay[name] = ("det", det)
            else:
                reject_cnt[name] += 1
                # Keep the rejected detection so we can render where YOLO
                # actually fired — lets us tell "KF prediction wrong" apart
                # from "YOLO false positive".
                per_frame_overlay[name] = ("rejected", (cx, cy, bbox, d2))

        # Consensus reinit: if >=2 cams rejected this cycle and their
        # detections triangulate to a consistent 3D point, the cameras
        # collectively disagree with the KF — trust the cameras, drop the
        # KF, and re-seed at the consensus. Fast recovery when the cup
        # moves under occlusion and reappears somewhere else.
        if kf.initialized:
            rejected = {n: payload for n, (tag, payload)
                         in per_frame_overlay.items()
                         if tag == "rejected" and n in cams}
            if len(rejected) >= 2:
                cam_keys = list(rejected.keys())[:3]
                pts = [np.array([rejected[n][0], rejected[n][1]])
                       for n in cam_keys]
                try:
                    X_cons = triangulate_dlt([cams[n] for n in cam_keys], pts)
                    residuals = []
                    for n, pt in zip(cam_keys, pts):
                        uv, ok = project(cams[n], X_cons)
                        if ok:
                            residuals.append(float(np.linalg.norm(uv - pt)))
                    consensus_px = max(residuals) if residuals else 1e9
                except Exception:
                    consensus_px = 1e9
                    X_cons = None
                if X_cons is not None and consensus_px < 30.0:
                    print(f"consensus reinit: {len(cam_keys)} cams agree on "
                          f"{X_cons.round(1)} mm (max resid {consensus_px:.1f}px);"
                          f" KF was at {kf.x[:3].round(1)} mm")
                    kf = KalmanFilter3D()
                    kf.init(X_cons, t=now)
                    last_accepted_t = now
                    init_dets = {}
                    for n in cam_keys:
                        cxn, cyn, bn, _ = rejected[n]
                        per_frame_overlay[n] = ("det",
                                                 (cxn, cyn, bn, None, 0.0))
                        last_bbox[n] = bn
                        accept_cnt[n] += 1
                        reject_cnt[n] = max(0, reject_cnt[n] - 1)

        # Throttle rendering to ~30 Hz
        if (now - t_render_last) * 1000 < show_every_ms:
            continue
        t_render_last = now
        if kf.initialized:
            kf.predict(now)

        # Reset accept/reject counters once per second
        if now - t_counter_reset >= 1.0:
            accept_cnt = {n: 0 for n in cam_names}
            reject_cnt = {n: 0 for n in cam_names}
            t_counter_reset = now

        tiles = {}
        for name in cam_names:
            f = last_frame.get(name)
            if f is None:
                continue
            tile = f.copy()
            h, w = tile.shape[:2]
            tag, payload = per_frame_overlay.get(name, (None, None))
            det_overlay = payload if tag == "det" else None
            rejected = payload if tag == "rejected" else None
            pred_overlay = None
            proj_overlay = None
            if det_overlay is None and kf.initialized and name in cams:
                uv, ok = kf.project_into(cams[name])
                if ok and 0 <= uv[0] < w and 0 <= uv[1] < h:
                    # Predicted bbox = this cam's last accepted bbox,
                    # translated to where the 3D centroid now projects.
                    # No shape model — just "the cup was that big here,
                    # now it's over there".
                    pred_overlay = (uv, last_bbox.get(name),
                                     last_poly.get(name))
                elif ok:
                    proj_overlay = (uv, ok)
            draw_overlay(tile, det_overlay, pred_overlay, proj_overlay)
            # Cyan rectangle = the KF's predicted accept region. Any YOLO
            # detection whose centroid lands outside this gets the red REJ
            # box treatment.
            region = (predicted_region(kf, cams[name], last_bbox.get(name),
                                        margin=1.6)
                      if name in cams else None)
            if region is not None:
                gcx, gcy, ghw, ghh = region
                gx0, gy0 = int(gcx - ghw), int(gcy - ghh)
                gx1, gy1 = int(gcx + ghw), int(gcy + ghh)
                cv2.rectangle(tile, (gx0, gy0), (gx1, gy1), (220, 220, 0), 1)
            if rejected is not None:
                rcx, rcy, rbox, rd2 = rejected
                x0, y0, x1, y1 = (int(round(v)) for v in rbox)
                cv2.rectangle(tile, (x0, y0), (x1, y1), (40, 40, 200), 2)
                cv2.putText(tile, "REJ outside region", (x0, max(15, y0 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 200), 2)
            # Per-cam accept/reject counter + most recent Mahalanobis d²
            a, r = accept_cnt[name], reject_cnt[name]
            color = (0, 220, 0) if a >= r else (0, 80, 220)
            txt = f"+{a} -{r}  d2={last_d2[name]:5.1f}"
            cv2.putText(tile, txt, (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            tiles[name] = tile

        mosaic = build_mosaic(tiles, cam_names)
        if kf.initialized:
            X = kf.x[:3]
            cv2.putText(mosaic,
                        f"3D pos (mm): {X[0]:+7.1f} {X[1]:+7.1f} {X[2]:+7.1f}",
                        (10, mosaic.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("live_track", mosaic)
        if recorder is None and record_path is not None:
            h, w = mosaic.shape[:2]
            recorder = cv2.VideoWriter(
                str(record_path), cv2.VideoWriter_fourcc(*"mp4v"),
                float(RECORD_FPS), (w, h))
            record_start_t = now
        if recorder is not None:
            # Pad-write: bring frames_written up to the number that would
            # have been written at RECORD_FPS in the elapsed wall-clock
            # time. Duplicates the current mosaic when render is slow.
            target = int((now - record_start_t) * RECORD_FPS) + 1
            while frames_written < target:
                recorder.write(mosaic)
                frames_written += 1
            if duration_s is not None and (now - record_start_t) >= duration_s:
                print(f"recording: done — wrote {record_path} "
                      f"({frames_written} frames)")
                break
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break
    if recorder is not None:
        recorder.release()
        print(f"recording: closed {record_path}")
    cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS,
                    help="YOLO-seg .pt (default: %(default)s)")
    ap.add_argument("--cal", default=DEFAULT_CAL,
                    help="aniposelib TOML calibration (default: iMOVE 5-cam rig)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--classes", default="cup-like",
                    help="'cup-like' | 'all' | comma-separated ints")
    ap.add_argument("--record", type=Path, default=None,
                    help="If set, write the rendered mosaic to this mp4 path")
    ap.add_argument("--duration", type=float, default=None,
                    help="Stop after this many seconds when --record is set "
                         "(default: until you press q)")
    ap.add_argument("--check", action="store_true",
                    help="One-shot cal sanity check: triangulate + project back")
    args = ap.parse_args()

    if args.classes == "all":
        classes = None
    elif args.classes == "cup-like":
        classes = CUP_LIKE_CLASSES
    else:
        classes = [int(x) for x in args.classes.split(",")]

    streams = load_streams()
    cams = load_calibration(args.cal)
    print(f"streams: {list(streams)}    calibrated: {list(cams)}")
    missing = set(streams) - set(cams)
    if missing:
        print(f"warning: no calibration for {missing} — those cams won't fuse")

    ctx, socks = open_sockets(streams, args.host)
    try:
        if args.check:
            # ArUco-only check — doesn't need YOLO weights
            run_check(cams, socks)
        else:
            from ultralytics import YOLO
            print(f"loading {args.weights}...")
            model = YOLO(args.weights)
            run_live(model, cams, socks, args.conf, classes,
                     record_path=args.record, duration_s=args.duration)
    finally:
        for s in socks.values():
            s.close()
        ctx.term()


if __name__ == "__main__":
    main()
