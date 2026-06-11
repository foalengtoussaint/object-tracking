---
name: project_e6_camera_transfer
description: "drink_study E6: single-camera transfer is a property of viewpoint not data volume; cam8 is a hub (707 frames→7/10 cams), cam10 row=1.0/0.0 = static-glass memorization"
metadata: 
  node_type: memory
  type: project
  originSessionId: bdb7162a-6c96-4391-8f59-078e4c60445f
---

E6 camera-transfer matrix (drink_study, 2026-06): trained 10 single-camera students on P01 cam_N reps 1-10 (sampled to 3000-frame budget where possible), tested per-camera on held-out P01 reps 11-15 (same person, isolates pure cam-to-cam transfer). Matrix at `experiments/drink_study/e6/transfer_matrix.json`; runner `experiments/drink_study/run_e6.py` (all cache-hit, no teacher; eval_gate's `per_camera` table = one matrix row).

**Findings:**
1. **Diagonal always strong (own-view recall 0.64–1.0), even cam8 trained on only 707 frames** → the 3000-frame budget is well past saturation; **training-data VOLUME is not the bottleneck** (corroborates the main sweep's early saturation). Per-cam frame yield from 10 reps varied wildly (cam8=707, cam6=1780, cam9=1188 … cam2=3966, cam10=5003) but didn't drive the results.
2. **Transferability is a property of the source VIEWPOINT, not volume.** Hub cameras generalize broadly: **cam8 (707 frames) → 7/10 cameras** (cam2=.99, cam10=.93, cam3=.80), cam2 → ~5. Loner cameras cover only themselves ± one neighbor: cam1, cam3, cam7, cam10. Best single camera to train on = cam8; worst = cam10/cam1/cam3.
3. **Rig viewpoint-adjacency structure:** reciprocal pairs cam2↔cam4 (.44/.72) and cam7↔cam9 (.47/.61); cam1 is a common target (cam5→.61, cam6→.76). The distant cam10 is reachable only from cam6 (.36) and cam8 (.93).
4. **cam10 row = 1.0 own / 0.0 all others = static-glass memorization signature** — the cam10 student learned a fixed FP (the side-desk glass at a constant pixel), trivially perfect on its own held-out view, useless elsewhere. End-to-end confirmation of [[project_cam10_distractor_cup]].

**Why:** quantifies WHY multi-view training is needed (one view ≈ near-zero transfer for loner cams) and that you need ≥1 camera per cluster, not more frames. A perfect-own/zero-transfer row flags a memorized static FP.

**How to apply:** pick training cameras for viewpoint coverage (a hub like cam8 + one per isolated pair) rather than maximizing frames; treat a 1.0-own/0.0-transfer signature as a distractor-memorization red flag. Relates to the main sweep (participant diversity >> clip count) and [[feedback_keep_all_experiment_data]] (entirely cache-hit, no retrain of the teacher).
