---
name: project_fill_refill_recache
description: "fill/refill (reproject-fill) models re-cached on ALL drink reps, all 10 cams, via NVDEC GPU decode; caches at cache/student_dets_clean3d_{fill,refill}/; gpu_decode.py is the reusable NVDEC helper"
metadata:
  type: project
---

The reproject-fill models (the "best model" — clean3d lineage with reject-then-fill labels):
- **fill**:   `/home/imove/drink_study_models/pscale_1_clean3d_fill/best.pt`
- **refill**: `/home/imove/drink_study_models/pscale_1_clean3d_refill/best.pt`
- single-class (`{0: 'cup'}`), heavier backbone than the pscale_* nano students.

Re-cached detections for BOTH on **all 724 deduped drink reps** (both hands, all trial dirs),
**all available cameras** (10 where present), conf 0.25:
- `experiments/drink_study/cache/student_dets_clean3d_fill/{pid}_{stem}__clean3d_fill__c0.25.json`
- `.../student_dets_clean3d_refill/{pid}_{stem}__clean3d_refill__c0.25.json`
- 724 files each; P10 now 10-cam (was 5 — see [[project_dets_collision_5cam_bug]]); P12/P21
  genuinely 5-cam. refill detects MORE per rep (~5500 vs fill ~3800) — reproject-fill taught
  it to find the cup more often. ~5h/model (~10h total) on RTX 3060 Ti.

**`gpu_decode.py`** (new, reusable): NVDEC H.264 decode via ffmpeg (`-hwaccel cuda -c:v
h264_cuvid`) piping BGR frames to numpy — this cv2 build has no CUDA, decord/PyNvVideoCodec
absent, but ffmpeg has NVDEC. `frames(path)` / `dims(path)`; CPU cv2 fallback; OT_FORCE_CPU_DECODE=1
to force CPU. `cache_dets_model.py` uses it (streaming batched inference, serial → no OOM,
unlike the cache_all_dets parallel path that OOM'd VSCode). Run with conda env python directly
(NOT `conda run`, which block-buffers stdout and hides live progress).

Next: rebuild 3D tracks for both models with the consensus-anchored KF ([[project_consensus_anchored_kf]]),
then re-run flagging to measure improvement over pscale_4 (the glass-contaminated model,
see [[project_pscale4_full_det_cache]]). Cup-only segmentation = `segment_cup_only.py`;
quality detectors = trajectory_cleanliness/lateral + PCA/DTW shape (`flag_trials.py`).
