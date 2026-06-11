---
name: project_agreement_cam4_dominates
description: drink_study agreement-px variation across checkpoints is ~90% caused by cam_4 alone (left-wrist marker bracelet ≈ cup); metric needs inlier-gated triangulation
metadata: 
  node_type: memory
  type: project
  originSessionId: bdb7162a-6c96-4391-8f59-078e4c60445f
---

On the P06 rep, the inter-camera agreement median-reproj-px swings a lot across checkpoints (std 5.5, range 3.1–21.9px; ep5=21.9, ep24=10.7). Cache-backed leave-one-out (debug_cam10/dets_cache + analyze_dets.py) shows **dropping cam_4 collapses the swing to std 0.64, range 3.0–5.0px** — cam_4 explains ~90% of the variation. ep5 delta 18.4px and ep24 delta 6.2px are entirely cam_4.

**Two distinct per-camera pathologies (don't conflate):**
- **cam_4 = precision failure (confident but wrong):** its view confuses the cup with the **marker bracelet on the participant's left wrist** ~half the frames. Median own-reproj 16–26px even on calm epochs (worst real camera every epoch), blowing up to 65–66px on loose checkpoints (ep5/ep24). Detects plenty of frames, so it poisons the all-cam DLT triangulation.
- **cam_10 = coverage failure (barely detects):** 350–1250px error but only a handful of detections (distant view; trained-on static side-glass, see [[project_cam10_distractor_cup]]). High error, near-zero weight.

**Why:** the agreement metric's apparent instability is NOT the model's global precision wobbling — it's one rogue camera dragging a non-robust triangulation. "Agreement got worse this epoch" can mean "cam_4 localized worse," not "the model degraded."

**cam_4 is the HARDEST view for BOTH models, but capacity sets the severity:**
- Teacher (yolo26x) cam_4 median 2.5px but it is the **only** teacher camera with any >20px frames (6%, 14 frames, max 40px) — every other teacher cam is flat 0%. Those 14 bad frames are NOT the bracelet: they cluster at frame-center (0.58,0.48) = the real cup near the mouth, in time-contiguous runs (fr109–123, 376–378) = mild localization jitter during the fast drinking motion / hand occlusion. A precision nudge, right object.
- Student (yolo26n) cam_4: 86% >20px, errors to 200px, locked onto the **wrist marker bracelet** (different location, bottom-right ~0.88) = wrong object.
- So cam_4's viewpoint is intrinsically hard (cup passes near hand/mouth with occlusion; wrist marker in frame). Capacity decides whether "hard" = 25px jitter (teacher) or grabbing the wrong object (student). Labels are NOT poisoned (teacher hits the right cup); it's a capacity / hard-negative problem — bigger student or bracelet hard-negatives, not a labeling fix. Teacher is also conservative on cam_4 (248/541 detected vs student ~460).

**How to apply:**
- Make triangulation robust (RANSAC / inlier-gated; `inlier_frac` at 30px is already computed but the reported median uses ALL cams). With cam_4 rejected per-frame, the metric reads ~3–5px stably across epochs — the real precision signal.
- **VALIDATED: inlier-gating at thr≈30px is the sweet spot.** Iteratively drop the worst camera whose reproj>30px and refit. On P06 it improves the cam_10-reprojected cup position by **up to 34px during the ~4s drinking-peak window** (drops cam_4 28× + cam_3 16×, keeps a healthy median of 5 cams). Stricter thr8/thr5 over-prune to a fragile 3-camera consensus (tighter residual but fewer votes — not necessarily more accurate). Caveat: gating helps ONLY in the hard window; over the whole clip the median shift is 0px (95% of frames drop nothing) — judge by the windowed effect, NOT the clip-wide median (that median-hides-the-tail trap bit the analysis twice).
- Gating fixes the single-rogue-camera case (cam_4) but NOT a frame where the MAJORITY of cameras are wrong at once (the 4s correlated occlusion) — that needs the temporal `KalmanFilter3D`, not stricter per-frame gating.
- When a per-camera reproj error is huge, check detect count first: huge+few = coverage; moderate+many = a confidently-wrong camera that actually corrupts triangulation.
- This whole analysis ran with ZERO re-inference off cached detections — [[feedback_keep_all_experiment_data]] in practice. Relates to the drink_study agreement metric.
