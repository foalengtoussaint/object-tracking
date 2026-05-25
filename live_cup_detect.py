"""Live YOLO cup detection with per-stream Kalman tracker.

For every ZMQ stream in cam_config_server.yaml, runs YOLO-seg and feeds the
top detection into a constant-velocity Kalman filter. When the detector
misses a frame (or returns an outlier), the filter predicts the cup's
position and draws a yellow "ghost" box — interpolating across the gap.

Visuals:
  green box + mask + conf   →  current YOLO detection
  yellow thin box           →  Kalman prediction (no detection this frame)
  green tail / yellow tail  →  recent observed / interpolated positions

Usage:
    python live_cup_detect.py
    python live_cup_detect.py --weights runs/segment/my_cup_v26/weights/best.pt
    # press q in any window to quit
"""
from __future__ import annotations

import argparse
import collections

import cv2
import numpy as np
import zmq
from ultralytics import YOLO

from cam_streams import load_streams
from pseudo_label import KalmanFilter2D

# COCO classes used when running a generic (non-fine-tuned) model.
CUP_LIKE_CLASSES = [39, 40, 41, 45, 75]  # bottle, wine glass, cup, bowl, vase


class StreamTracker:
    """Kalman tracker for a single object on one camera."""

    def __init__(self, max_miss: int = 15, tail_len: int = 40,
                 gate: float = 13.82):
        self.kf = KalmanFilter2D()
        self.initialized = False
        self.misses = 0
        self.max_miss = max_miss
        self.gate = gate
        self.last_w = 0.0
        self.last_h = 0.0
        self.tail: collections.deque = collections.deque(maxlen=tail_len)

    def step(self, detection):
        """detection = (cx, cy, w, h, conf, mask) or None.
        Returns a dict of track info, or None if no track is active."""
        if detection is None:
            return self._predict_only(already_predicted=False)

        cx, cy, w, h, conf, mask = detection

        if self.initialized:
            self.kf.predict()
            d = self.kf.mahalanobis((cx, cy))
            if d > self.gate:
                # Outlier — fall back to prediction-only this frame
                return self._predict_only(already_predicted=True,
                                          rejected_conf=conf,
                                          rejected_d=d)
            self.kf.update((cx, cy))
        else:
            self.kf.init((cx, cy))
            self.initialized = True

        self.last_w = float(w)
        self.last_h = float(h)
        self.misses = 0
        self.tail.append((cx, cy, False))
        return dict(cx=cx, cy=cy, w=w, h=h, conf=conf, mask=mask,
                    predicted=False, misses=0, rejected=None)

    def _predict_only(self, already_predicted: bool,
                      rejected_conf=None, rejected_d=None):
        if not self.initialized:
            return None
        self.misses += 1
        if self.misses > self.max_miss:
            self.initialized = False
            self.tail.clear()
            return None
        if not already_predicted:
            self.kf.predict()
        px, py = float(self.kf.x[0]), float(self.kf.x[1])
        self.tail.append((px, py, True))
        rejected = None
        if rejected_d is not None:
            rejected = dict(conf=rejected_conf, mahalanobis=rejected_d)
        return dict(cx=px, cy=py, w=self.last_w, h=self.last_h,
                    conf=None, mask=None, predicted=True,
                    misses=self.misses, rejected=rejected)


def best_detection(res):
    """Highest-confidence detection. Returns (cx, cy, w, h, conf, mask) or None."""
    if res.boxes is None or len(res.boxes) == 0:
        return None
    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    i = int(np.argmax(confs))
    x1, y1, x2, y2 = boxes[i]
    mask = None
    if res.masks is not None and len(res.masks.data) > i:
        mask = res.masks.data[i].cpu().numpy()
    return (float((x1 + x2) / 2), float((y1 + y2) / 2),
            float(x2 - x1), float(y2 - y1), float(confs[i]), mask)


def draw_track(frame, info, tail) -> np.ndarray:
    h_f, w_f = frame.shape[:2]
    out = frame.copy()

    # Trajectory tail (green where observed, yellow where interpolated)
    pts = list(tail)
    for j in range(1, len(pts)):
        xa, ya, pa = pts[j - 1]
        xb, yb, pb = pts[j]
        col = (0, 255, 255) if (pa or pb) else (0, 200, 100)
        cv2.line(out, (int(xa), int(ya)), (int(xb), int(yb)), col, 2)

    if info is None:
        return out

    cx, cy, w, h = info["cx"], info["cy"], info["w"], info["h"]
    x1, y1 = int(cx - w / 2), int(cy - h / 2)
    x2, y2 = int(cx + w / 2), int(cy + h / 2)

    if info["predicted"]:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 1)
        cv2.putText(out, f"predicted (miss {info['misses']})",
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if info.get("rejected"):
            r = info["rejected"]
            cv2.putText(out, f"rejected conf={r['conf']:.2f} d={r['mahalanobis']:.1f}",
                        (x1, min(h_f - 5, y2 + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    else:
        if info["mask"] is not None:
            m = info["mask"]
            if m.shape != (h_f, w_f):
                m = cv2.resize(m, (w_f, h_f), interpolation=cv2.INTER_LINEAR)
            sel = m > 0.5
            out[sel] = (0.5 * out[sel] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"my_cup {info['conf']:.2f}",
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--config", default=None)
    ap.add_argument("--weights", default="yolo11n-seg.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-miss", type=int, default=15)
    ap.add_argument("--gate", type=float, default=13.82,
                    help="Kalman Mahalanobis gate; loose=permissive")
    args = ap.parse_args()

    streams = load_streams(args.config)
    print(f"Connecting to {len(streams)} stream(s) on {args.host}: "
          + ", ".join(f"{n}:{p}" for n, p in streams.items()))

    print(f"loading {args.weights}...")
    model = YOLO(args.weights)
    classes = CUP_LIKE_CLASSES if len(model.names) > 10 else None

    trackers = {name: StreamTracker(max_miss=args.max_miss, gate=args.gate)
                for name in streams}

    ctx = zmq.Context()
    sockets = {}
    for name, port in streams.items():
        s = ctx.socket(zmq.SUB)
        s.connect(f"tcp://{args.host}:{port}")
        s.setsockopt_string(zmq.SUBSCRIBE, "")
        s.setsockopt(zmq.RCVHWM, 2)
        sockets[name] = s

    print("Connected. Press 'q' in any window to exit.")
    while True:
        for name, socket in sockets.items():
            raw = None
            while True:
                try:
                    raw = socket.recv(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
            if raw is None:
                continue
            frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8),
                                 cv2.IMREAD_COLOR)
            if frame is None:
                continue
            res = model(frame, classes=classes, conf=args.conf, verbose=False)[0]
            det = best_detection(res)
            info = trackers[name].step(det)
            disp = draw_track(frame, info, trackers[name].tail)
            cv2.imshow(name, disp)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
