"""Shared, machine-portable paths for the drink_study scripts.

Override on a different machine without editing any script:

    export OT_CLIPS_ROOT=/path/to/clips        # raw per-participant videos
    export OT_PYTHON=/path/to/env/bin/python    # (informational; for docs)

Defaults match the original workstation so nothing changes here.
"""
from __future__ import annotations
import os
from pathlib import Path

# Root holding per-participant clip dirs: <CLIPS_ROOT>/P01/P01_drinking_right_*.<cam>.mp4
CLIPS_ROOT = Path(os.environ.get("OT_CLIPS_ROOT", "/home/imove/Documents/clips"))

# In-repo cache (detections, 3D tracks, QTM mocap, etc.)
CACHE = Path(__file__).resolve().parent / "cache"

# Qualisys cup mocap: 772 labeled C3D drinking trials (4 cup markers, 100 Hz, mm).
# Copied off the Windows QTM partition; travels with the project.
QTM_C3D = Path(os.environ.get("OT_QTM_C3D", str(CACHE / "qtm_c3d")))
