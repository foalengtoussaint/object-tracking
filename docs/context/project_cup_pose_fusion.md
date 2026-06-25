---
name: project_cup_pose_fusion
description: confidence-weighted cup+pose fusion for drink phase — distance-to-head(mouth proxy), near=evidence/slow=gate, log-odds fusion; fixes the occluded-apex dwell
metadata:
  type: project
---

Drink-dwell phase segmentation by FUSING the cup track and the pose, in
`experiments/drink_study/fuse_phases.py` (+ `render_fused.py` for video). Built
because the cup-only dwell gate ([[project_cup_only_segmentation]]) is blind at
the mouth — the cup is occluded there, so its track is interpolated/rescued
([[project_consensus_anchored_kf]], 2-cam rescue) and the speed-threshold dwell
test whipsaws on ~1cm of apex jitter.

**Core idea — one physical signal, two sources, confidence-weighted.** Both
trackers are projected onto distance-to-mouth and fused by how sure each is:
- `cup → head` distance (rescued refill cup track vs head joint), weight = cup
  tracker confidence = `min(kept_cams/3,1)·tightness(median_px)`, 0 on
  interpolated frames, 0.35 on a 2-cam rescue.
- `wrist → head` distance (dominant wrist 85/77 vs head, by npz `side`), weight =
  pose confidence = `sqrt(wrist_conf · head_conf)` from `keypoints3d[...,3]`.
The confidences are ANTI-correlated in the right way: cup confidence COLLAPSES at
the occluded dwell while pose holds; during fast transport pose blurs while cup
is clean. So the handoff is automatic — no hard winner rule.

**Mouth = HEAD joint 67 as a proxy** — there is NO dedicated mouth/nose joint in
bml_movi_87. The container's own `drink_task_segmentation.py` does the same
("Head keypoint as the mouth proxy", line ~245). The constant head→lips offset
(~10–15cm, so cup→head plateaus at ~135mm at a sip, not 0) is cancelled by
self-normalising distance to each trial's own 5th-pct minimum. So it's a proxy
and works, but it is NOT true cup-to-lip distance. This REPLACES the old
rest-relative "near peak displacement" hack with an anatomically-meaningful,
self-normalised distance-to-mouth.

**Fusion math (two fixes after a first wrong version):**
- evidence per source `e = near · gate(speed)` — distance-to-mouth IS the signal
  so NEAR is the evidence; SLOW is only a GATE (a fast frame can't be a dwell),
  it must not DILUTE a clear near-mouth reading. (First version multiplied two
  soft memberships → product capped ~0.5, never crossed threshold.)
- fuse in LOG-ODDS, confidence-scaled: `L = w_cup·logit(e_cup) +
  w_pose·logit(e_pose); E = sigmoid(L)`. Votes ADD, so two confident sources
  agreeing push E ABOVE either alone (true agreement reinforcement) — a plain
  weighted average can never exceed the max and averages to the middle.
- hysteresis on E (on 0.55 / off 0.40, min 12 frames) → drinking interval.
  Transport/rest onset/offset still come from cup motion window (clean
  end-effector); only the drink dwell is fused so far.

**Validated on the 4 cached biomech trials** (P01/P06/P19/P23 — see
[[project_calibrate_all_participants]] for the biomech npz recipe). Fused drink
interval lands BETWEEN cup-only (too short) and pose-only (over-extended) on all
4, anchored where both are confident. P23 dwell: E rises to a 0.78 plateau while
cup confidence is ~0 — pose carries it.

**Validated on the 20 WORST no-drink trials** (cohort's highest 2-cam-rescue
counts = cup struggled most; P07/P14/P15/P24/P23/P10). Fusion gives a confident
(>=0.25s) drink dwell on **19/20**, RECOVERING **13** that cup-only missed (9 of
which cup-only found literally NONE). **0 fusion-destroyed dwells.** The 1 miss
(P23_151925, pose conf 0.60) is a borderline-short dwell: fused E peaks 0.64 (>
on-thr) but doesn't sustain past the 12-frame duration gate — a threshold
brittleness both methods share, not a fusion failure. Montage:
`cache/fuse_montage_worst.png`; per-trial `cache/fuse_<trial>.png`.

**Standalone pose recipe (was never saved before — THIS is the reusable bit):**
pose comes from a DataJoint pipeline in `isr-containers-dev-1`; the npz were made
ad-hoc. Reproduced WITHOUT DataJoint via
`DATA/ot_pose_scratch/run_pose.py` (in the DATA mount, no container code
modified) + host driver `experiments/drink_study/run_pose_batch.sh`:
- MUST start BOTH containers: `docker start isr-containers-db-1` (MySQL) AND
  `isr-containers-dev-1`. pose_pipeline imports DataJoint AT IMPORT TIME and
  connects to host `db` — without db up, even loading the MeTRAbs model fails.
- in-container paths: `sys.path` needs `/home/vscode/workspace/packages/imove_extensions`
  AND `/.../PosePipeline` (note double-nest imove_extensions/imove_extensions).
- recipe: `metrabs_multicam_batched({cam:vid}, skeleton="bml_movi_87")` -> per-cam
  keypoints2d -> `triangulate_keypoints_multicam(kp2d, {cam:{K,dist,R,t}})` with
  our `data/calib/<P>/calibration.toml` -> keypoints3d in W0 (NOT MeTRAbs's own
  metric frame — re-triangulate the 2D, like PersonKeypointFast does). Saves npz
  with keypoints3d (T,87,4) + side; no phase_intervals (skipped segment_drink_task).
- ~5.5 min/trial (10-cam 1080p cross-cam MeTRAbs decode dominates; num_aug=2).
  GPU = RTX 3060 Ti 8GB, ran fine at num_aug=2, ~5GB peak, no OOM on 20 trials.

Open: extend fusion to transport/rest boundaries (currently cup-only); validate
fused boundaries vs the container 7-phase intervals quantitatively; relax the
duration gate / lower off-thr to catch the P23_151925-style short dwells; fuse
cohort-wide (pose now cached for 24 trials = 4 original + 20 worst).
