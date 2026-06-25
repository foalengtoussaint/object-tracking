---
name: project_calibrate_all_participants
description: "Calibrated 22/23 drink_study participants from LOCAL Charuco footage (not the unmounted SSD); run_calibration.py per participant; only P21 has no footage; P03 needs session2, P24 is 9-cam (cam4 junk)"
metadata:
  type: project
---

The "calibrate the other participants — BLOCKED on unmounted SSD" task (see
[[session_20260618_cache_and_outliers]]) was a false premise: the per-participant
Charuco footage is **local** at `/home/imove/Documents/clips/P<NN>/P<NN>_calibration_<ts>.<cam>.mp4`.

Calibrated **22/23** participants (was 4: P01/P06/P19/P23) via `recording/run_calibration.py`
on per-participant `data/calib/<P>/` dirs (cam-1..10.mp4 symlinks to the footage). Then
`cache_track3d.py` produced 3D tracks for all of them. Reproj errors 0.55–1.78px (P05
loosest at 3.70).

- **Only P21 cannot be calibrated** — no calibration footage exists anywhere locally.
- **P03**: first session `20240112_150724` is a false-start (3MB files, 0 boards) — use the
  second session `20240112_150823`.
- **P24**: cam-4 detects only partial board slivers (0 views ≥9 Charuco corners) and crashes
  the aniposelib solver; calibrate from its **9 good cameras** (drop cam-4) → 0.683px.
- Batch tool: `experiments/drink_study/calibrate_all.py` (idempotent, picks earliest session,
  per-participant progress). `run_calibration.py` now names cams by FILE number (cam-7.mp4→
  "cam7") so dropping a bad camera keeps the others correctly identified.
- aniposelib calibration is CPU bundle-adjust ~9 min/participant, ~1.5GB RAM; ran 3-wide
  safely on the 24-core box (NOT the torch-decode OOM path).

Calib quality feeds the 3D cup track that the iMOVE drink-task phase segmentation consumes.
