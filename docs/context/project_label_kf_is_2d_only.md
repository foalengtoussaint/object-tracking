---
name: project_label_kf_is_2d_only
description: "teacher→student labels are filtered by a 2D per-camera KF (teleport/phantom gate) — blind to wrong-but-smooth objects, static FPs, and cross-camera 3D disagreement"
metadata: 
  node_type: memory
  type: project
  originSessionId: bdb7162a-6c96-4391-8f59-078e4c60445f
---

The teacher→student label path DOES filter with a Kalman filter, but it's `kalman.py::KalmanFilter2D` / `filter_detections` — **2D, single-object, per-camera, per-clip** (GATE=13.82 chi-2 99.9%, MAX_MISS=30). Called via `pseudo_label.label_clip(use_kf=True)` ← `label_clip_cached`. It gates each detection by Mahalanobis distance from the per-camera predicted centroid: rejects teleports, NMS duplicates, phantoms within one camera's track. This is the KF the "obvious we need it" ablation refers to — it genuinely cleans labels.

**But it is structurally blind to the three failure modes found this session (P06 baseline):**
- **cam_4 wrist-marker bracelet (smooth wrong object):** bracelet is near the cup and moves smoothly to the mouth → passes the gate; not a teleport. [[project_agreement_cam4_dominates]]
- **cam_10 static side-glass (stationary FP):** a static object has ~zero innovation → the KF *blesses* it, never gates it. [[project_cam10_distractor_cup]]
- **~4s drinking-peak correlated occlusion:** per-camera KFs coast independently; no cross-camera reconciliation, so 4 cams mislocalizing at once goes uncaught.

Note this 2D label KF is SEPARATE from `kalman_3d.py::KalmanFilter3D` (the 3D multi-view filter used in the live tracker `live_track.py`/`viz_rerun.py`). The agreement *metric* (`agreement.py`) uses neither KF — raw per-frame DLT triangulation by design (so it can expose the disagreement spikes).

**Why:** explains why bad labels still reached the student despite "KF filtering" — the filtering is 2D/single-view; the failures are 3D/cross-camera or wrong-object.

**How to apply:** the concrete fix is to **promote label-time filtering from 2D-per-camera to 3D multi-view** — triangulate during labeling (cameras are synchronized + calibrated) and reject detections that don't agree across cameras, which would drop the cam_10 static glass and down-weight the cam_4 bracelet BEFORE the student trains. Pairs with the parked inlier-gated triangulation for the metric. Relates to [[feedback_keep_all_experiment_data]] (all of this diagnosed from saved checkpoints + cached dets, zero retrain).

**GATE FIX + why the 3D KF is required (the synthesis).** The first gate had `min_cams=2` floor: it stopped dropping cameras at 2, but 2 cams are never vetted (loop exits before testing them), so 54 cam_10 glass frames survived as degenerate pairs disagreeing by THOUSANDS of px (e.g. cam_10 vs cam_2 at 3910px "kept"). Fix: drop ANY camera with reproj>thr regardless of count, then require **≥3 genuinely-agreeing cams** for a valid consensus. With the fix, cam_10 glass = **100% rejected** (647/647).
BUT the fixed gate then leaves **223/870 frames "no-consensus" (<3 cams detect the cup)** — the deep-occlusion frames during the drink. Per-frame gating has NO memory, so it can't fill these. **This is the concrete proof the temporal 3D KF (`kalman_3d.py::KalmanFilter3D`) is necessary**: gating answers "is this detection geometrically consistent NOW" (rejects glass/bracelet); the 3D KF answers "where must the cup be given its recent motion" and coasts through the no-consensus frames. Two halves of one filter: gate spatially per-frame, then track temporally. Not redundant. Ties to the ~4s occlusion window in [[project_agreement_cam4_dominates]].

**VALIDATED 2026-06-05 on P01.** Calibrated P01 (`recording/run_calibration.py` on the seq2 calibration video `P01_calibration_20231220_142559`, 3413 frames → `data/calib/P01/calibration.toml`, reproj error **1.207px**; NOTE seq1 `..142219` is a 458-frame truncated take that detects 0 boards — use seq2). Ran teacher on rep `P01_drinking_right_20231220_141546`, 3D inlier-gated at thr=30px (need ≥3 cams/frame). Result: cam_10 static glass (center 0.87,0.69 std=0, detected 870/870) is **92% REJECTED** (593/647), while the real-cup cameras are kept (cam_1/2/6/7/9 = 0% rejected; cam_3/4/5 = 3–6%; cam_8 = 33%). So cross-camera geometry surgically removes the static FP that the 2D-KF blessed, without any region heuristic. Teacher dets cached at `experiments/drink_study/cache/P01_..._teacher__c0.25.json`. Only TEST participants (P06/P19/P23) were calibrated before; now P01 too.
