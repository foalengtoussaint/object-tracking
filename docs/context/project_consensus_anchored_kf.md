---
name: project_consensus_anchored_kf
description: "drink_study 3D-track divergences are the 2D-gated EKF being hijacked by a single bad detection; feeding the KF the CONSENSUS as a 3D measurement (no gating) fixes it — 346→354/355 clean, all 7 divergences gone"
metadata:
  type: project
---

The big 3D-track failures (P24 "flying to 2.4m", several P10 reps) are **KF divergence,
not detection or confident-wrong**. Diagnosed on cached track3d (consensus/kf/rts stored
per frame):

- At the divergence the **consensus stays correct** (~10mm from cup, ~5px, 9 cams agree),
  but the **KF runs away to 1m+** and reprojects 200–700px from *every* camera's detection.
- Mechanism: the current KF (`KalmanFilter3D` in `kalman_3d.py`) is an EKF updated by raw
  per-camera **2D detections with a Mahalanobis gate**. A single bad detection (e.g. one
  camera's transient outlier — cam_3 in P24_105712) kicks the state; once off, the gate
  **rejects every true detection** (now "too far") so it coasts ballistically and never
  recovers. My earlier "9 cams agree on a far object" read was WRONG — I'd measured RTS
  displacement (follows the runaway KF) + the kept/px fields (describe the consensus).

**Fix (validated, `kf_consensus.py`):** keep the KF's dynamics/interpolation but drop the
2D filtering — feed the **consensus as a direct 3D measurement to a no-gate linear KF+RTS**.
It's anchored (every consensus frame pulls it back → cannot diverge) yet still smooths and
interpolates gaps with the constant-velocity model.

Cohort result (355 reps), cup-only-segmentation cleanliness:
- RTS_2d (current 2D-gated): drink 347, clean 346, median detour 1.073
- consensus + linear interp: drink 344, clean **274** (linear kinks fail cleanliness — worse)
- **consensus→KF→RTS: drink 352, clean 354, detour 1.049 — best; all 7 divergences fixed**

So it's the **gating** that's harmful, not the filter; crude linear interp is NOT the answer.
Remaining failures after the fix: ~2 truncated clips (P20/P14) + borderline fast sips
(P24_105328, P10_153258) that sit on the drink-dwell threshold. TODO: port consensus-anchored
KF into `viz_replay.run_pipeline` / `cache_track3d`. Relates to [[project_failure_modes_confident_wrong]].
