"""Calibrate N cameras from Charuco videos using aniposelib (same flow as iMOVE).

Reads all cam-*.mp4 files from <video_dir> (cam-1.mp4, cam-2.mp4, …),
runs aniposelib's CharucoBoard + CameraGroup.calibrate_videos, writes
calibration.toml next to the videos. The TOML is the same aniposelib format
consumed by calibration.py / visual_hull.py.

Usage:
    python run_calibration.py <video_dir>
"""
from __future__ import annotations

import argparse
import re
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
                    help="Directory containing cam-1.mp4, cam-2.mp4, … (any count)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output TOML path (default: <video_dir>/calibration.toml)")
    args = ap.parse_args()

    video_files = sorted(
        args.video_dir.glob("cam-*.mp4"),
        key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)),
    )
    if len(video_files) < 2:
        raise SystemExit(
            f"need at least 2 cam-N.mp4 files in {args.video_dir}, "
            f"found {len(video_files)}"
        )
    # Name each camera by the number in its filename (cam-7.mp4 -> "cam7"),
    # not by position — so a non-contiguous set (e.g. a camera dropped because
    # it never saw the board) keeps each remaining camera's true identity.
    # Identical to positional naming for the normal contiguous cam-1..N case.
    cam_nums = [int(re.search(r"(\d+)", v.stem).group(1)) for v in video_files]
    cam_names = [f"cam{n}" for n in cam_nums]
    print(f"videos ({len(video_files)}): {[v.name for v in video_files]}")

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
