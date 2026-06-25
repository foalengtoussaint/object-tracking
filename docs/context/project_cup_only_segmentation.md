---
name: project_cup_only_segmentation
description: "Cup-ONLY drink-task phase segmentation from the 3D cup track (no pose): speed + displacement-from-rest; 5 phases (no reaching); 98% drink-detect on 355 reps; 3 rotation-invariant trajectory-weirdness detectors"
metadata:
  type: project
---

Goal of the object_tracking work = feed the iMOVE drink-task phase segmentation
(`iMOVE/DEV/imove_extensions/drink_task_segmentation.py`, which normally needs cup 3D +
PersonKeypointFast pose). We have **no body pose** for these reps, so built a **cup-only**
segmenter: `experiments/drink_study/segment_cup_only.py`.

- Inputs: the cached 3D cup track (`cache/track3d/*.json`, RTS). Two rotation-invariant
  signals (calibration world frame = per-session Charuco frame, no fixed "up"):
  **cup speed** (mm/s) and **displacement-from-rest** (‖cup − median(first 0.5s)‖ mm).
- 5 phases (no `reaching` — that's hand-toward-still-cup, invisible to the cup; folds into
  rest_pre): rest_pre → forward_transport → drinking → back_transport → rest_post.
  Motion window = speed hysteresis (onset 150, offset 80 mm/s); drinking = near peak disp +
  slow (<120mm/s) ≥0.2s.
- 355 reps: **98% drink-dwell detected**. High rate = robustness of fusion+smoothing, NOT
  proof of accuracy (agreement ≠ correctness still applies).

**Trajectory-weirdness detectors** (a good drink = one clean up-and-down, planar):
- up-down cleanliness (`trajectory_cleanliness.py`): n_peaks (5% prominence) + detour ratio
  (arc/2·peak; cohort median 1.07 = near-minimal). 
- lateral/planarity (`trajectory_lateral.py`): PC3 (out-of-plane) extent = left/right wander;
  median 20mm. PC3=medio-lateral VALIDATED: lift(rest→apex) ⟂ PC3 to 89.6° median.
- shape PCA+DTW (`flag_trials.py` consolidates all). NOTE DTW over-flags whole minority
  participants (P23/P20 clusters) — trust only when a geometric method agrees.
- Coverage/gap (cup lost = kept<3 cams) is the most direct detector for cup-at-mouth losses.

Most flagged failures were P10 (5-cam-cache artifact, see [[project_dets_collision_5cam_bug]])
and KF divergences (see [[project_consensus_anchored_kf]]) — i.e. pipeline bugs, not bad data,
plus a few genuinely truncated clips. Videos: render_tracking_video.py / render_grid_video.py
(all-camera grid, NVDEC). Per-trial viz: viz_flagged.py.
