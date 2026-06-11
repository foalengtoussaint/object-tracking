---
name: project_3dclean_vs_dropcam
description: "drink_study: 3D-gate label cleaning (reject-only, thr30, ≥3 cams) recovers cam_10 (0.00→0.51) but over-prunes (40% dropped) so mean recall (0.738) loses to crudely dropping cam_10 (0.789); needs looser gate or reproject-fill"
metadata: 
  node_type: memory
  type: project
  originSessionId: bdb7162a-6c96-4391-8f59-078e4c60445f
---

Three P01 students, same 3000-frame budget, eval per-camera on held-out P06/P19/P23 (cached, `run_clean3d.py` / `run_drop_cam10.py`):

| variant | mean recall | cam10 | cam4 |
|---|---|---|---|
| pscale_1 (raw, glass labels) | 0.700 | 0.00 | 0.78 |
| drop_cam10 (train on cams 1-9) | **0.789** | 0.13 | 0.92 |
| clean3d (3D inlier-gate ALL cams, reject-only) | 0.738 | **0.51** | 0.77 |

**Key result:** 3D-gating **recovers the hard cam_10 view (0.00→0.51)** — drop_cam10 can't, because removing the camera means the student never learns that viewpoint (stays 0.13). So per-detection cleaning > dropping a whole camera *for the cleaned camera itself*.

**But reject-only over-prunes:** the gate dropped **40% of all labels** (glass + bracelet + every frame with <3-cam consensus — conservative), so cleaned training lost many GOOD hard frames → other cameras regressed (cam2 0.94→0.80, cam7 0.84→0.65, cam9 0.91→0.63). Net mean recall (0.738) < surgical drop_cam10 (0.789).

**Why:** the right idea (geometric per-detection cleaning, no camera dropped) executed too aggressively. ≥3-cam consensus throws away verifiable-enough frames.

**How to apply:** loosen the gate (accept 2-cam agreement) OR use **reject + reproject-fill** (add labels by reprojecting the 3D-KF/RTS track into cameras that missed/were-rejected) instead of reject-only — recover cam_10 WITHOUT the data loss. The reject-only version is a floor, not the ceiling. Relates to [[project_pscale_glass_confound]], [[project_label_kf_is_2d_only]], [[project_cam10_distractor_cup]].
