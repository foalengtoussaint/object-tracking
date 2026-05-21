"""Record one MP4 per ZMQ camera stream for Charuco calibration.

Subscribes to all three streams, decodes JPEGs, writes each to its own MP4 at
30 fps. Move the Charuco board through every part of the workspace and through
every viewing angle each camera can see; aim for ~30 s of footage with the
board fully visible in at least one cam at all times.

Output (default out_dir = calib_recordings/<timestamp>/):
    cam-1.mp4   <- cam_left
    cam-2.mp4   <- cam_center
    cam-3.mp4   <- cam_right

The cam-N naming matches what aniposelib / iMOVE expect.

Usage:
    python record_calibration.py [out_dir]   # SPACE = start/stop, q = quit
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import zmq

PORTS = {
    "cam_left": (55555, "cam-1"),
    "cam_center": (55556, "cam-2"),
    "cam_right": (55557, "cam-3"),
}
FPS = 30
IMAGE_SIZE = (1280, 720)  # (W, H)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?", default=None, type=Path)
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = Path("calib_recordings") / time.strftime("%Y%m%d_%H%M%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output: {args.out_dir}")

    ctx = zmq.Context()
    sockets, writers, paths = {}, {}, {}
    for stream, (port, name) in PORTS.items():
        s = ctx.socket(zmq.SUB)
        s.setsockopt(zmq.RCVHWM, 60)   # 2s of buffer per cam
        s.setsockopt_string(zmq.SUBSCRIBE, "")
        s.connect(f"tcp://127.0.0.1:{port}")
        sockets[stream] = s
        paths[stream] = args.out_dir / f"{name}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    recording = False
    counts = {s: 0 for s in PORTS}
    t_start = None

    print("SPACE = start/stop recording, q = quit  (focus a preview window first)")
    while True:
        for stream, sock in sockets.items():
            try:
                raw = sock.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                continue
            frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if frame.shape[1] != IMAGE_SIZE[0] or frame.shape[0] != IMAGE_SIZE[1]:
                frame = cv2.resize(frame, IMAGE_SIZE)

            if recording:
                writers[stream].write(frame)
                counts[stream] += 1

            disp = frame.copy()
            tag = f"REC {counts[stream]:5d}" if recording else "idle"
            color = (0, 0, 255) if recording else (200, 200, 200)
            cv2.putText(disp, f"{stream}  {tag}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow(stream, disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            if not recording:
                for stream, path in paths.items():
                    writers[stream] = cv2.VideoWriter(
                        str(path), fourcc, FPS, IMAGE_SIZE)
                    if not writers[stream].isOpened():
                        raise SystemExit(f"failed to open writer {path}")
                t_start = time.monotonic()
                recording = True
                print(f"[REC] start")
            else:
                recording = False
                dur = time.monotonic() - t_start
                for stream, w in writers.items():
                    w.release()
                writers.clear()
                print(f"[REC] stop  {dur:.1f}s  frames per cam:")
                for s, c in counts.items():
                    print(f"   {s}: {c}  -> {paths[s]}")

    for w in writers.values():
        w.release()
    cv2.destroyAllWindows()
    print(f"done. videos in {args.out_dir}")


if __name__ == "__main__":
    main()
