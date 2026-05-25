"""Capture a synchronized JPEG frame from each of the 3 ZMQ camera streams.

Drains stale frames first, then waits until a fresh frame has arrived from
every camera within a short window. The most-recent frame per camera is saved.

Usage:
    python capture_triplet.py [output_dir] [--yolo]
    # SPACE = capture, y = toggle YOLO overlay, q = quit
    # --yolo starts with the overlay already enabled
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import zmq

from cam_streams import load_streams

COCO_CUP_CLASS = 41
YOLO_WEIGHTS = "yolo11n-seg.pt"

SYNC_WINDOW_S = 0.10   # all three frames must arrive within this window


def _drain(socket: zmq.Socket) -> None:
    while True:
        try:
            socket.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            return


def grab_triplet(sockets: dict[str, zmq.Socket], timeout_s: float = 2.0
                 ) -> dict[str, np.ndarray]:
    for s in sockets.values():
        _drain(s)

    deadline = time.monotonic() + timeout_s
    arrival: dict[str, float] = {}
    latest: dict[str, bytes] = {}

    while time.monotonic() < deadline:
        for name, s in sockets.items():
            try:
                raw = s.recv(flags=zmq.NOBLOCK)
                latest[name] = raw
                arrival[name] = time.monotonic()
            except zmq.Again:
                pass

        if len(arrival) == len(sockets):
            spread = max(arrival.values()) - min(arrival.values())
            if spread <= SYNC_WINDOW_S:
                break
        time.sleep(0.001)
    else:
        missing = set(sockets) - set(arrival)
        raise RuntimeError(f"timeout waiting for cameras: missing={missing}")

    frames = {}
    for name, raw in latest.items():
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"{name}: JPEG decode failed")
        frames[name] = frame
    return frames


def overlay_cup(frame: np.ndarray, model) -> np.ndarray:
    """Run YOLO on `frame` and return a copy with cup mask+bbox+conf drawn."""
    res = model(frame, classes=[COCO_CUP_CLASS], verbose=False)[0]
    out = frame.copy()
    if res.boxes is None or len(res.boxes) == 0:
        cv2.putText(out, "no cup", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
        return out
    confs = res.boxes.conf.cpu().numpy()
    xyxy = res.boxes.xyxy.cpu().numpy()
    if res.masks is not None:
        h, w = frame.shape[:2]
        for i, m in enumerate(res.masks.data.cpu().numpy()):
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            sel = m > 0.5
            out[sel] = (0.5 * out[sel] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    for (x1, y1, x2, y2), c in zip(xyxy.astype(int), confs):
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"cup {c:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?", default="captures", type=Path)
    ap.add_argument("--yolo", action="store_true",
                    help="Start with the live YOLO overlay enabled")
    ap.add_argument("--host", default="127.0.0.1",
                    help="IP of the camera server (default: 127.0.0.1)")
    ap.add_argument("--config", default=None,
                    help="Path to cam_config_server.yaml (auto-detected if omitted)")
    args = ap.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    yolo_on = args.yolo
    model = None

    streams = load_streams(args.config)
    print(f"Connecting to {len(streams)} stream(s) on {args.host}: "
          + ", ".join(f"{n}:{p}" for n, p in streams.items()))

    ctx = zmq.Context()
    sockets = {}
    for name, port in streams.items():
        s = ctx.socket(zmq.SUB)
        s.setsockopt(zmq.CONFLATE, 1)   # must come before connect
        s.setsockopt_string(zmq.SUBSCRIBE, "")
        s.connect(f"tcp://{args.host}:{port}")
        sockets[name] = s

    print("Click a cam window, then:  SPACE/c = capture,  y = toggle YOLO,  q = quit")
    saved = 0
    while True:
        try:
            frames = grab_triplet(sockets, timeout_s=2.0)
        except RuntimeError as e:
            print(f"[warn] {e}")
            if cv2.waitKey(50) & 0xFF == ord("q"):
                break
            continue

        for name, f in frames.items():
            display = overlay_cup(f, model) if (yolo_on and model is not None) else f
            cv2.imshow(name, display)

        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            print(f"  [key={key}]")
        if key in (ord("q"), 27):
            break
        if key == ord("y"):
            yolo_on = not yolo_on
            if yolo_on and model is None:
                print("  loading YOLO...")
                from ultralytics import YOLO
                model = YOLO(YOLO_WEIGHTS)
                print("  YOLO loaded")
            print(f"  YOLO overlay: {'ON' if yolo_on else 'OFF'}")
        if key in (ord(" "), ord("c"), 13, 10):
            stamp = time.strftime("%Y%m%d_%H%M%S")
            for name, f in frames.items():
                path = out_dir / f"{stamp}_{name}.png"
                cv2.imwrite(str(path), f)   # clean frame, no overlay
                print(f"  saved {path}")
            saved += 1

    cv2.destroyAllWindows()
    print(f"done. {saved} triplet(s) saved to {out_dir}/")


if __name__ == "__main__":
    main()
