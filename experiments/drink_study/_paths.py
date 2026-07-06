"""Shared, machine-portable paths for the drink_study scripts.

Override on a different machine without editing any script:

    export OT_CLIPS_ROOT=/path/to/clips        # raw per-participant videos
    export OT_PYTHON=/path/to/env/bin/python    # (informational; for docs)

Defaults match the original workstation so nothing changes here.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

# The load-bearing spine modules live in ./lib/ (segment_cup_only, qtm_align, learn_seg, ...).
# Every entry-point script imports `_paths` (directly or transitively), so putting lib/ on
# sys.path here lets all scripts keep importing spine modules by BARE name
# (`import segment_cup_only`) regardless of which subdirectory the script itself lives in.
# This is the single hinge that keeps the flat-layout imports working after the reorg.
_LIB = Path(__file__).resolve().parent / "lib"
if _LIB.is_dir() and str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))

# Anchors, resolved from THIS file's location (so they are correct no matter which
# subdirectory a script lives in — the reason path bugs vanished after the lib/ reorg):
DS = Path(__file__).resolve().parent          # experiments/drink_study
ROOT = DS.parents[1]                           # repo root (…/object-tracking-master)

# Root holding per-participant clip dirs: <CLIPS_ROOT>/P01/P01_drinking_right_*.<cam>.mp4
CLIPS_ROOT = Path(os.environ.get("OT_CLIPS_ROOT", "/home/imove/Documents/clips"))

# In-repo cache (detections, 3D tracks, QTM mocap, etc.)
CACHE = DS / "cache"

# Qualisys cup mocap: 772 labeled C3D drinking trials (4 cup markers, 100 Hz, mm).
# Copied off the Windows QTM partition; travels with the project.
QTM_C3D = Path(os.environ.get("OT_QTM_C3D", str(CACHE / "qtm_c3d")))

# Cleaned re-export of the same 772 trials: identical 4 cup markers PLUS a 5-marker rigid
# HEAD cluster (FHD + L/R front/back head) on 737/772 trials. Used for the mouth-based
# drink-dwell truth (qtm_c3d.CupTrial.mouth_proxy / mouth_dwell.py). Same stems as QTM_C3D.
QTM_C3D_HEAD = Path(os.environ.get("OT_QTM_C3D_HEAD", str(CACHE / "qtm_c3d_cleaned")))
