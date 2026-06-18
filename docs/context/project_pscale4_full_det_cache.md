---
name: project_pscale4_full_det_cache
description: "Full pscale_4 student detection cache exists for 370 drinking_right reps, all participants — don't re-run the 12h GPU job"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7857798a-7e7a-40b9-bc77-a4a5a69893a9
---

The pscale_4 student (`experiments/drink_study/runs/pscale_4/weights/best.pt`) has
been run on **every drinking_right rep, all 10 cams, all 23 participants** (P01–P24,
no P22). Cached as per-cam per-frame cup centroids in
`experiments/drink_study/cache/student_dets/{P}_{stem}__pscale_4__c0.25.json`
(**370 distinct reps**, integrity-verified, 0 corrupt). Built 2026-06-18 by
`experiments/drink_study/cache_all_dets.py --workers 1 --batch 64` (~12.5h GPU,
serial batched). Format is byte-identical to the per-frame cache, so it's directly
reusable by `viz_replay.py`, `kf_accuracy.py`, and the agreement/robustness analyses
with **no GPU**.

**Why:** this is an expensive, reusable asset — re-running is a ~12h GPU job.
**How to apply:** before any pscale_4 inference over drinking clips, reuse these JSONs
(per [[feedback_keep_all_experiment_data]] / always-use-the-cache). 3D consensus/KF
still needs calibration TOMLs — only P01/P06/P19/P23 have them ([[calibration_source]]);
the other ~19 participants are 2D detections awaiting calib.

Gotchas:
- `cache_all_dets.py` batched path (`agreement.cup_centroids_batched`, batch=64) is
  ~2.5x faster than per-frame and verified identical (None-pattern exact, centroids
  <0.06px). GPU ~2-5GB, RAM-safe single process.
- DON'T use the parallel-decode (`--workers >1`) path: spawn workers re-import torch,
  load CUDA each, buffer 1080p clips → OOM'd and killed the VSCode extension host on
  this 29GB box. Real GPU-decode (NVDEC) isn't available in the `.venv` (torchvision
  0.27 dropped VideoReader; no decord/ffmpeg-cuvid).
- `(1)` clip dirs (P03 (1), P10 (1), …) are the SAME participant, other trials; they
  fold to the base id in the cache key, so 407 raw jobs → 370 distinct files.
