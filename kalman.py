"""Single-object 2D Kalman filter (constant-velocity) and a per-frame
detection-filtering helper used by both `pseudo_label.py` (to drop bad teacher
predictions when building a dataset) and `evaluate.py` (to compute PR proxies
on cached detections).

The filter assumes the operator kept a single object visible throughout the
clip — it tracks one centroid and gates new detections by their Mahalanobis
distance from the KF's predicted position.

Both modules call `filter_detections()`; the helper is the single source of
truth for what counts as accepted, rejected, duplicate, phantom, or teleport.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# Chi-squared 2-DoF gates: 5.99=95%, 9.21=99%, 13.82=99.9%
GATE = 13.82
MAX_MISS = 30           # frames KF can coast before re-init from next candidate
SEPARATION_PX = 100     # extras within this distance count as NMS duplicates


class KalmanFilter2D:
    """Constant-velocity Kalman filter for a 2D centroid (state = x, y, vx, vy)."""

    def __init__(self, process_noise: float = 4.0, meas_noise: float = 8.0):
        dt = 1.0
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        q = process_noise ** 2
        self.Q = q * np.array([
            [dt**4 / 4, 0, dt**3 / 2, 0],
            [0, dt**4 / 4, 0, dt**3 / 2],
            [dt**3 / 2, 0, dt**2, 0],
            [0, dt**3 / 2, 0, dt**2],
        ])
        self.R = (meas_noise ** 2) * np.eye(2)
        self.x = None
        self.P = None

    def init(self, z):
        self.x = np.array([z[0], z[1], 0.0, 0.0])
        self.P = np.diag([10.0, 10.0, 100.0, 100.0])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def _innov(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return y, S

    def mahalanobis(self, z) -> float:
        y, S = self._innov(z)
        return float(y @ np.linalg.inv(S) @ y)

    def update(self, z):
        y, S = self._innov(z)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P


@dataclass
class FilteredFrame:
    """Result of filtering one frame's detections.

    accepted   the chosen detection dict, or None if no detection was accepted
    role       one of: accepted | predicted | rejected | uninit
    dup        extra detections within SEPARATION_PX of the accepted one
               (NMS duplicates on the same object)
    phantom    extra detections farther than SEPARATION_PX from the accepted
               one (real false positives on something else)
    tele       1 if a lone detection failed the gate and was rejected as a
               teleport, else 0
    """
    accepted: dict | None
    role: str
    dup: int = 0
    phantom: int = 0
    tele: int = 0


def _classify_extras(primary: dict, extras: list[dict], separation_px: float
                     ) -> tuple[int, int]:
    dup = phan = 0
    for d in extras:
        if np.hypot(d["cx"] - primary["cx"], d["cy"] - primary["cy"]) < separation_px:
            dup += 1
        else:
            phan += 1
    return dup, phan


def filter_detections(per_frame: list[list[dict]],
                       gate: float = GATE,
                       max_miss: int = MAX_MISS,
                       separation_px: float = SEPARATION_PX,
                       ) -> list[FilteredFrame]:
    """Run a single-object Kalman filter over `per_frame` and label each frame.

    Each frame is a list of detection dicts. Each dict must have `cx`, `cy`,
    `conf` keys; anything else (bbox, polygon, idx) is preserved and returned
    in `FilteredFrame.accepted` for the caller to consume.

    Returns one FilteredFrame per input frame, in order.
    """
    kf = KalmanFilter2D()
    initialized = False
    misses = 0
    out: list[FilteredFrame] = []

    for dets in per_frame:
        if not dets:
            if initialized:
                kf.predict()
                misses += 1
                out.append(FilteredFrame(None, "predicted"))
            else:
                out.append(FilteredFrame(None, "uninit"))
            continue

        if not initialized:
            top = max(dets, key=lambda d: d["conf"])
            kf.init((top["cx"], top["cy"]))
            initialized = True
            misses = 0
            dup, phan = _classify_extras(top, [d for d in dets if d is not top],
                                         separation_px)
            out.append(FilteredFrame(top, "accepted", dup=dup, phantom=phan))
            continue

        kf.predict()
        scored = sorted(((kf.mahalanobis((d["cx"], d["cy"])), d) for d in dets),
                        key=lambda s: s[0])
        best_d, best_c = scored[0]
        extras = [s[1] for s in scored[1:]]

        if best_d <= gate and misses <= max_miss:
            kf.update((best_c["cx"], best_c["cy"]))
            misses = 0
            dup, phan = _classify_extras(best_c, extras, separation_px)
            out.append(FilteredFrame(best_c, "accepted", dup=dup, phantom=phan))
        elif best_d <= gate:
            kf.init((best_c["cx"], best_c["cy"]))
            misses = 0
            dup, phan = _classify_extras(best_c, extras, separation_px)
            out.append(FilteredFrame(best_c, "accepted", dup=dup, phantom=phan))
        else:
            misses += 1
            if len(dets) == 1:
                out.append(FilteredFrame(None, "rejected", tele=1))
            else:
                out.append(FilteredFrame(None, "rejected", phantom=len(dets)))

    return out
