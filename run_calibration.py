"""Calibrate 3 cameras from Charuco videos using aniposelib (same flow as iMOVE).

Reads cam-1.mp4 / cam-2.mp4 / cam-3.mp4 from <video_dir>, runs aniposelib's
CharucoBoard + CameraGroup.calibrate_videos, writes calibration.toml next to
the videos. The TOML is the same aniposelib format consumed by
calibration.py / visual_hull.py.

Usage:
    python run_calibration.py <video_dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

from aniposelib.boards import CharucoBoard
from aniposelib.cameras import CameraGroup

# Match the iMOVE/idrink default board (CharucoBoardDescription in
# iMove-SPZ/src/camera/calibration.py:91).
BOARD = dict(
    squaresX=7,
    squaresY=5,
    square_length=115,
    marker_length=92,
    marker_bits=4,
    dict_size=250,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dir", type=Path,
                    help="Directory containing cam-1.mp4, cam-2.mp4, cam-3.mp4")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output TOML path (default: <video_dir>/calibration.toml)")
    args = ap.parse_args()

    cam_names = ["cam1", "cam2", "cam3"]
    video_files = [args.video_dir / f"cam-{i}.mp4" for i in (1, 2, 3)]
    for v in video_files:
        if not v.exists():
            raise SystemExit(f"missing: {v}")
    print(f"videos: {[v.name for v in video_files]}")

    out_path = args.out or (args.video_dir / "calibration.toml")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    board = CharucoBoard(**BOARD)
    cgroup = CameraGroup.from_names(cam_names, fisheye=False)
    print("running aniposelib bundle calibration (this can take minutes)...")
    error, _rows = cgroup.calibrate_videos(
        [[str(v)] for v in video_files], board, verbose=True)
    print(f"calibration done. reprojection error = {error:.3f}")

    cgroup.metadata = {"error": float(error), "board": BOARD}
    cgroup.dump(str(out_path))
    print(f"wrote {out_path}")
    print(f"\nNext: edit visual_hull.py and set\n  CALIB_TOML = \"{out_path}\"")


if __name__ == "__main__":
    main()
