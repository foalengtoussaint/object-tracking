"""Load aniposelib-style TOML calibration into OpenCV world->cam form.

Translation units are millimeters (matches the iMOVE project).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import toml


@dataclass
class Camera:
    name: str
    K: np.ndarray           # 3x3 intrinsics (scaled to image_size)
    dist: np.ndarray        # 5-vec distortion (OpenCV)
    R: np.ndarray           # 3x3 rotation, world -> cam
    t: np.ndarray           # 3x1 translation, world -> cam (mm)
    image_size: tuple[int, int]   # (width, height) of frames we'll feed in
    calib_size: tuple[int, int]   # (width, height) at calibration time

    @property
    def P(self) -> np.ndarray:
        return self.K @ np.hstack([self.R, self.t.reshape(3, 1)])

    @property
    def center(self) -> np.ndarray:
        return (-self.R.T @ self.t).reshape(3)


def _scale_K(K: np.ndarray, calib_wh: tuple[int, int], target_wh: tuple[int, int]) -> np.ndarray:
    sx = target_wh[0] / calib_wh[0]
    sy = target_wh[1] / calib_wh[1]
    K = K.copy()
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] *= sx
    K[1, 2] *= sy
    return K


def load_calibration(
    toml_path: str | Path,
    cam_mapping: dict[str, str],
    target_image_size: tuple[int, int] = (1280, 720),
) -> dict[str, Camera]:
    """
    cam_mapping maps stream-name -> TOML section name, e.g.
        {"cam_left": "cam_0", "cam_center": "cam_1", "cam_right": "cam_2"}
    target_image_size is the (width, height) of frames you will pass in.
    Intrinsics are scaled from the calibration resolution to target_image_size.
    """
    data = toml.load(str(toml_path))

    cams: dict[str, Camera] = {}
    for stream_name, section in cam_mapping.items():
        block = data[section]
        K_calib = np.array(block["matrix"], dtype=np.float64)
        calib_wh = tuple(block["size"])
        K = _scale_K(K_calib, calib_wh, target_image_size)

        rvec = np.array(block["rotation"], dtype=np.float64)
        R, _ = cv2.Rodrigues(rvec)
        t = np.array(block["translation"], dtype=np.float64).reshape(3)
        dist = np.array(block.get("distortions", [0, 0, 0, 0, 0]), dtype=np.float64)

        cams[stream_name] = Camera(
            name=block.get("name", section),
            K=K,
            dist=dist,
            R=R,
            t=t,
            image_size=target_image_size,
            calib_size=calib_wh,
        )
    return cams


if __name__ == "__main__":
    import sys
    toml_path = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/imove/Documents/iMOVE/DATA/sub-test/20260428-152404/trials/" \
        "calibration_20260428_152404_calibration.filtered.1-2-3.toml"
    mapping = {"cam_left": "cam_0", "cam_center": "cam_1", "cam_right": "cam_2"}
    cams = load_calibration(toml_path, mapping)
    for name, c in cams.items():
        print(f"{name} ({c.name}): center={c.center.round(1)} mm  fx={c.K[0,0]:.1f}")
