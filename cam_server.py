"""Minimal ZMQ camera server using OpenCV.

Reads cam_config_server.yaml, opens each ZMQ-enabled camera with OpenCV,
and publishes JPEG frames on the configured ports — same wire protocol as
the full teleimager server, but requires only opencv-python and pyzmq.

Usage:
    python cam_server.py                         # uses teleimager/cam_config_server.yaml
    python cam_server.py --config path/to.yaml
    # Ctrl-C to stop
"""
from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import zmq
import yaml

_DEFAULT_CONFIG = Path(__file__).parent / "teleimager" / "cam_config_server.yaml"


def _load_config(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _publish_camera(name: str, video_id: int, port: int,
                    img_shape: tuple[int, int], fps: int,
                    stop: threading.Event, ctx: zmq.Context) -> None:
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 2)
    sock.bind(f"tcp://*:{port}")

    cap = cv2.VideoCapture(video_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_shape[1])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_shape[0])
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        print(f"[{name}] ERROR: could not open /dev/video{video_id}", flush=True)
        return

    print(f"[{name}] /dev/video{video_id} -> port {port} @ {fps} fps", flush=True)
    interval = 1.0 / fps
    next_t = time.monotonic()

    while not stop.is_set():
        ret, frame = cap.read()
        if not ret:
            print(f"[{name}] read failed, retrying...", flush=True)
            time.sleep(0.1)
            continue

        if frame.shape[1] != img_shape[1] or frame.shape[0] != img_shape[0]:
            frame = cv2.resize(frame, (img_shape[1], img_shape[0]))

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            sock.send(buf.tobytes(), flags=zmq.NOBLOCK)

        next_t += interval
        sleep = next_t - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)
        else:
            next_t = time.monotonic()

    cap.release()
    sock.close()
    print(f"[{name}] stopped", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="Path to cam_config_server.yaml (auto-detected if omitted)")
    args = ap.parse_args()

    cfg_path = Path(args.config) if args.config else _DEFAULT_CONFIG
    if not cfg_path.exists():
        sys.exit(f"config not found: {cfg_path}")

    config = _load_config(cfg_path)
    ctx = zmq.Context()
    stop = threading.Event()
    threads = []

    for name, cam_cfg in config.items():
        if not isinstance(cam_cfg, dict) or not cam_cfg.get("enable_zmq"):
            continue
        video_id = cam_cfg.get("video_id")
        port = cam_cfg.get("zmq_port")
        img_shape = cam_cfg.get("image_shape", [480, 640])   # [H, W]
        fps = cam_cfg.get("fps", 30)
        if video_id is None or port is None:
            print(f"[{name}] skipping — missing video_id or zmq_port")
            continue
        t = threading.Thread(
            target=_publish_camera,
            args=(name, int(video_id), int(port),
                  tuple(img_shape), int(fps), stop, ctx),
            daemon=True,
        )
        t.start()
        threads.append(t)

    if not threads:
        sys.exit("no ZMQ-enabled cameras found in config")

    def _shutdown(sig, frame):
        print("\nstopping...", flush=True)
        stop.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"camera server running ({len(threads)} stream(s)) — Ctrl-C to stop")
    stop.wait()
    for t in threads:
        t.join(timeout=2.0)
    ctx.term()


if __name__ == "__main__":
    main()
