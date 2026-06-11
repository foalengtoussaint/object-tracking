---
name: project_cam10_distractor_cup
description: "drink_study cam_10 labels are a STATIC side-desk glass, not the task cup — teacher labels any cup-like object; KF can't filter a static false positive"
metadata: 
  node_type: memory
  type: project
  originSessionId: bdb7162a-6c96-4391-8f59-078e4c60445f
---

In the drink_study baseline (train P01), **all 525 cam_10 training labels sit at a fixed point x=0.877 y=0.687 with std=0.001** — a static object in the right-third of the wide cam_10 view. Rendering confirms it's a **glass/cup on the side desk by the operator's monitor**, NOT the cup the participant is drinking (which moves hand→mouth→table, center of frame). So the student trained on cam_10 learned a stationary distractor; on held-out P06 (no such glass there) it detects **0/541** frames.

Root cause = **label semantics**, not training mechanics (loss curves, mAP=0.994, YOLO-seg polygon format all healthy) and NOT apparent size (an earlier wrong guess). The COCO teacher (`yolo26x-seg`, cup-like classes) has no notion of *the task cup* — it labels ANY cup-like object, including bystander glasses, especially in wide/distant cameras.

**KF blind spot:** a *static* false positive is the easy case for the single-object KF — low residual, never gated out, tracked perfectly. So "KF kept it" is NOT evidence of correctness; the KF locks onto a static distractor more reliably than the real moving cup.

**Why:** changes how we trust the teacher labels and the agreement metric — a confidently-tracked static object can be the wrong object.

**Static-glass FP is P01-TRAIN-specific; the teacher's CALIBRATED-participant labels are clean.** On P06 the teacher detects the REAL cup in cam_10 (center 0.37,0.48 std~1px, reprojects 1.9px vs other-cam 3D consensus, 0% >30px). So the distractor-glass mislabel was in P01's training clips, not a universal teacher failure.

**Inlier-gating CANNOT clean these training labels — a calibration-coverage blocker:** only the held-out TEST participants (P06/P19/P23) have `data/calib/<P>/calibration.toml`; the TRAIN participants (P01–P05,P08–P10) have NONE. 3D cross-camera gating needs multi-cam calibration, so it can only run on the participants we don't train on. To use cross-camera agreement to reject teacher FPs at label time, the **training participants must first be calibrated** (`recording/run_calibration.py`).

**Gating vs 3D-KF for FILTERING are complementary, not redundant:** inlier-gating = spatial (reject a camera that disagrees with others THIS frame: cam_4 bracelet, a glass at a wrong 3D spot). 3D KF = temporal (reject/coast a detection that disagrees with where the cup was a moment ago: the ~4s occlusion where the MAJORITY of cams are wrong at once, which gating can't fix — no good majority to gate to). Want both for label-cleaning. [[project_agreement_cam4_dominates]]

**How to apply:**
- Constrain the teacher to the task cup: gate detections to a region near the active hand/body, or seed the KF from the cup that is moving + near the hand, so static bystander glasses are rejected. (2D/heuristic — works without calibration, unlike 3D gating.)
- When auditing labels, check label-center **std over time** per camera; near-zero std on a "drinking" cup = a static distractor, not the task object.
- Diagnosing this needed saved `epoch*.pt` + untouched clips + the cached label .txt — concrete proof of [[feedback_keep_all_experiment_data]]. Relates to the drink_study agreement metric and [[live_rig_cam_mapping]].
