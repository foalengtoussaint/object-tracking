---
name: project_pscale_glass_confound
description: "drink_study: participant-scaling F1 gain is ~2/3 real generalization + ~1/3 cam_10 glass-dilution; cam10 recall 0.00→0.67 is the outlier, but 8/10 cams improve"
metadata: 
  node_type: memory
  type: project
  originSessionId: bdb7162a-6c96-4391-8f59-078e4c60445f
---

Checked whether the sweep's participant-scaling F1 gain (pscale_1→pscale_4, recall 0.78→0.88) is real generalization or just dilution of P01's cam_10 static-glass labels. Per-camera held-out recall on P06/P19/P23 (cached, `experiments/drink_study/percam_recall.py`):

- **cam10 is the dominant mover: 0.00 → 0.67 (+0.67)** — pscale_1 NEVER detects the held-out cup in cam10 (glass-memorization), participants fix it. ~4× the average gain.
- **But 8/10 cameras improve** (cam3 +0.18, cam5 +0.21, cam8 +0.19, cam9 +0.24 — glass-unrelated). One regressed: cam6 −0.07.
- **Decomposition:** mean recall gain all-cams = +0.170; **excluding cam10 = +0.114**. So ~2/3 of the headline gain is genuine cross-participant generalization, ~1/3 is the cam10 glass-dilution bonus.

**ABLATION (drop cam_10 from training, P01 cams 1-9, same 3000 budget): the confound is BIGGER than the per-camera split suggested.** Mean held-out recall pscale_1=0.700 → **pscale_1_no10=0.789 (+0.089)** — just removing the glass-contaminated camera from P01 recovers most of what adding 3 participants did (pscale_4=0.870). Critically the gain is GLOBAL not localized: cam9 +0.25, cam5 +0.14, cam4 +0.13, cam3 +0.11 — cameras unrelated to cam_10 all improve. So P01's static-glass labels were poisoning the student's cup representation EVERYWHERE, not just on cam_10. The "participant effect" was substantially the glass; the true residual participant gain (no10 0.789 → pscale_4 0.870) is only ~+0.08. (pscale_4 still has its own cam_10 glass, so even this understates the clean participant effect — needs the 3D-cleaned re-run.)

**Why:** the "participants >> clips" headline was substantially CONFOUNDED — a single bad camera in training dragged recall down across all views. Don't trust the raw sweep magnitudes; the glass must be cleaned at source before the scaling curves are valid.

**How to apply:** report the gain BOTH ways (all-cams +0.17 / excl-cam10 +0.11). Once the 3D-gate label cleaner removes the glass at source (see [[project_label_kf_is_2d_only]]), re-run pscale to get the un-confounded participant-scaling curve. Relates to [[project_cam10_distractor_cup]], [[project_e6_camera_transfer]] (cam10 row=1.0/0.0 same glass).
