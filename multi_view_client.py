"""Live preview of all ZMQ camera streams from the teleimager server.

Usage:
    python multi_view_client.py                         # server on localhost
    python multi_view_client.py --host 192.168.1.10     # remote server
    python multi_view_client.py --config path/to/cam_config_server.yaml
    # Press q in any window to quit.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
import zmq

from cam_streams import load_streams


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1",
                    help="IP of the camera server (default: 127.0.0.1)")
    ap.add_argument("--config", default=None,
                    help="Path to cam_config_server.yaml (auto-detected if omitted)")
    args = ap.parse_args()

    streams = load_streams(args.config)
    print(f"Connecting to {len(streams)} stream(s) on {args.host}: "
          + ", ".join(f"{n}:{p}" for n, p in streams.items()))

    context = zmq.Context()
    sockets = {}
    for name, port in streams.items():
        s = context.socket(zmq.SUB)
        s.connect(f"tcp://{args.host}:{port}")
        s.setsockopt_string(zmq.SUBSCRIBE, "")
        sockets[name] = s

    print("Connected. Press 'q' in any window to exit.")
    while True:
        for name, socket in sockets.items():
            try:
                raw = socket.recv(flags=zmq.NOBLOCK)
                frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    cv2.imshow(name, frame)
            except zmq.Again:
                pass

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
