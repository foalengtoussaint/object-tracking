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
