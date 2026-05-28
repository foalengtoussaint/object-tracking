"""Record an MP4 clip from one ZMQ camera stream.

Usage:
    python recording/record_clips.py                                # 60s of the first stream
    python recording/record_clips.py --stream cam_1 --duration 30
    python recording/record_clips.py --out data/clips/test --duration 30
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import zmq

from cam_streams import load_streams


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default=None,
                    help="Stream name (default: first in YAML)")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--out", type=Path, default=Path("data/clips/train"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--config", default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    streams = load_streams(args.config)
    stream_name = args.stream or next(iter(streams))
    if stream_name not in streams:
        raise SystemExit(f"unknown stream {stream_name!r}, options: {list(streams)}")
    port = streams[stream_name]

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{stream_name}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{args.host}:{port}")
    sock.setsockopt_string(zmq.SUBSCRIBE, "")

    print(f"recording {stream_name} from tcp://{args.host}:{port} for {args.duration}s")
    print("waiting for first frame...")
    raw = sock.recv()
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = frame.shape[:2]

    writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (w, h))
    writer.write(frame)
    n = 1
    t0 = time.time()
    while time.time() - t0 < args.duration:
        try:
            raw = sock.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            time.sleep(0.005)
            continue
        f = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if f is not None:
            writer.write(f)
            n += 1

    writer.release()
    sock.close()
    ctx.term()
    print(f"wrote {n} frames to {out_path}")


if __name__ == "__main__":
    main()
