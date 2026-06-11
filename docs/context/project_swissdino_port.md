---
name: project_swissdino_port
description: "Porting SwissDINO (DINOv2 one-shot) into our pipeline — plan, decisions, build order"
metadata: 
  node_type: memory
  type: project
  originSessionId: 6dbe7eaf-84a9-438a-96e2-3dbb2b1ed064
---

Started 2026-06-09. Porting the one-shot DINOv2 method from [[reference_theinle_objectdetection]] into our object_tracking pipeline. Motivation user gave: faster onboarding, training-free fallback detector, **robustness to environment changes** (frozen web-scale DINOv2 features generalize across lighting/background far better than our small env-overfit YOLO student — directly counters cam_10 glass memorization, see [[project_cam10_apparent_size]], [[project_e6_camera_transfer]]).

**Decisions:** vit_b backbone default (flag-switchable). Clean-room reimplement (CC-BY-NC-SA, don't copy their files). Env = `idrink` (torch 2.5.1+cu121, CUDA ok, cv2 4.10).

**Planned modules (all additive, don't touch trained-YOLO live path):**
1. `swissdino_lib.py` — load_dino / extract_feature_map / build_prototype / detect / mask_patch_to_polygon (our `"0 x y ..."` format) / score_crop (for label gate). BUILDING FIRST + smoke test.
2. `swissdino_onboard.py` — clip + few SAM clicks → prototype → detect across clip → YOLO-seg seed dataset in our layout (feeds KF pseudo-labeler / finetune.py).
3. `pseudo_label.py --detector swissdino` hook + `score_crop` appearance gate for the future 3D multi-view label filter ([[project_label_kf_is_2d_only.md]] is the weakness it targets).

**Our detection currency:** YOLO-seg polygon `"0 x1 y1 x2 y2 ..."` normalized; teacher emits `{frame_idx: polygon}`; KF `filter_detections` takes per-frame `{cx,cy,conf,_idx}` dicts. SwissDINO patch mask → polygon = drop-in for both.

Smoke-test data: `data/datasets/cube_smoke_seed/` has cube frames + existing SAM labels (use as GT to validate prototype). Validation benchmark: SwissDINO-teacher vs YOLO-teacher recall on held-out P02 + cam_10.

**Module 1 = `swissdino_lib.py` built + tested 2026-06-09.** Clean-room reimpl correct, chain works end-to-end.

**CRITICAL CORRECTION (user caught this):** The repo's onboarding builds the prototype from a WHOLE VIDEO, not one frame. Mechanism (their onboard.py L1166-1185): Track-Anything propagates 1 click → per-frame masks across the clip; for each frame pull object patches `fm[mask]`; **np.concatenate object patches from ALL frames → np.mean → one prototype**; threshold = mean of per-frame adaptive thresholds. My first smoke test used ONE frame = degenerate worst case.

**Re-test on cube_smoke_seed (vit_b, 448px), eval on 6 held-out val frames:**
- single-frame onboard: thresh 0.989, mean IoU 0.16, 0% frames >0.3
- **VIDEO onboard (pool 24 train frames): thresh 0.642, mean IoU 0.48, 100% frames >0.3**
Pooling ~tripled IoU and fixed the pathological threshold. So video onboarding WORKS on the tiny cube even at 448px. 0.48 IoU = good localization, not tight enough as a standalone training label → refine via SAM (SwissDINO localizes → SAM tightens, we already have SAM).

KEY: their onboarding needs per-frame object masks (TA provides them). **We already generate equivalent per-frame masks** (KF pseudo-labeler + SAM flow) → can build a video-prototype from our own data, NO Track-Anything/XMem/WSL needed.

Still-true notes: appearance-gate (score_crop) remains a strong fit for the cam_10 glass problem. Going to 896px alone (single-frame) made it worse — resolution isn't the lever, multi-frame pooling is.

**XMem PROPAGATION TEST (2026-06-09):** Installed XMem (hkchengrex/XMem, ~62M params, 238MB ckpt XMem-s012.pth) at /tmp/XMem — inference runs in `idrink` with ZERO extra pip installs (missing deps progressbar2/thinplate/hickle/gdown are training/demo-only). Idea tested: drop SwissDINO, use detector/SAM for frame-1 only + XMem to propagate (= what theinle onboarding does minus the prototype). Test: SAM 1-click seed (via our sam_label_server.py) on frame 0 of `data/clips/drink_p02_val/cam_R0C4...mp4` (cup, 541 frames 1080p) → XMem propagate at 720px → overlay video `_swissdino_viz/xmem_cam4_overlay.mp4`.
RESULT: clean track while cup static (~4500px); mask shrinks then **fully lost (0px) ~frames 200-250 — user confirmed this is when the cup is AT THE MOUTH (drinking), not the initial grab**; 102/541 frames (19%) had NO mask, concentrated in the drinking phase; then **XMem RE-ACQUIRED the cup cleanly once it returned to the desk** (full ~4500px by f400) = long-term memory recovery. WHY mouth-phase fails: cup occluded by hand+face+chin AND tilted = max appearance change, while visible pixels (skin/face) don't match cup memory → 0; no motion prior to coast through. KEY for drink_study: the dropout lands exactly on the drinking action = the phase we most care about. Confirms tradeoff: XMem = pixel-tight masks + recovers from total loss, but NO motion/position prior so heavy-occlusion+appearance-change = dropout. Pairs naturally with our 3D/KF geometric prior which XMem lacks (KF would coast through the mouth phase).

**cam_10 XMem test (2026-06-09):** Same setup, seeded task cup on cam_R0C10 P02 (542 frames). RESULT: even worse occlusion failure — **31% frames no mask (169/542), one long continuous blackout ~f130-340 = the whole drinking phase**, then recovered on desk return. cam_10's higher/further viewpoint hides the cup more completely/longer during drinking. IMPORTANT: it went to 0px, NOT onto a wrong object = **failed SAFE (blank, KF-recoverable) not failed-confident-wrong.** CAVEAT on the distractor-glass premise: user + I looked and there is NO visible distractor glass in this P02 cam_10 clip — the [[project_cam10_apparent_size]] "static side-desk glass" note was about the YOLO labels / a different context, do NOT assume it transfers to this clip. So the look-alike-drift case was NOT actually tested here (no look-alike present); what's confirmed is severe occlusion dropout from this viewpoint.

**MULTI-VIEW XMem RESULT (2026-06-09, KEY FINDING):** Ran XMem on ALL 10 cameras of drink_p02_val to frame 300, seeded per-cam frame-0 cup masks. Per-cam hold rate: C1/C2/C5/C6/C7/C9=100%, C4=66%, C3=55%, C8=56%, C10=44% (worst, drinking-occlusion view). **CRUCIAL: per-frame, ALWAYS ≥6 cameras held the cup (min 6, mean 8.2/10); ≥3 cams 100% of frames; ZERO frames where all cams lost it.** No two cams are occluded at the same instant (hand/face blocks only some viewpoints). Since triangulation needs only 2-3 good views ([[project_kf_accuracy_budget]]) and we always have ≥6, **multi-view fully covers the single-cam occlusion blackout.** CONFIRMED ARCHITECTURE: XMem (tight per-cam masks) + our 3D triangulation (fuse + survive per-cam occlusion) — each covers the other's weakness. ~10s/cam at 720px on 3060Ti. CSV: _swissdino_viz/multiview_coverage.csv. NOTE: user pointed out seeding all 10 should use triangulate-one-point-then-reproject, not 10 manual clicks — do that next time.

**MASK-vs-BOX STABILITY (2026-06-09, user's hypothesis tested on cam_4):** User observed XMem mask point jumps more under occlusion than a bbox would. CONFIRMED: on frames where BOTH XMem & YOLO fire, partial-occlusion frame-to-frame jitter = XMem mask center **0.8px vs YOLO box 0.4px** (~2× steadier box). Mask-centroid≈mask-boxcenter (7.0 vs 7.2 in deep occ) so it's the MASK itself unstable, not the centroid-reduction. Mask area collapses 397→207px under occ (mask shrinks/wobbles as cup peeks past hand). BUT: I wrongly described a cam_4 'occluded window where YOLO finds nothing' — that was a single-view artifact; user correctly noted YOLO detects in SOME cameras most frames.

**ENV CORRECTION:** I wrongly ran everything in `idrink` (ultralytics 8.3.40, can't load our yolo26 student). Project env is **`object_tracking`** (ultralytics 8.4.49, torch cu118, CUDA) — [[conda-envs]] said so and I ignored it. Student `drink_p01` = yolo26n-seg, **imgsz 640**, 10 ep, trained on P01 data (P02=held-out). imgsz matters a lot: detects 0.82-0.93 @640, often MISSES @1280.

**REAL student vs XMem multi-view COVERAGE (P02, 10 cams, 300 frames, object_tracking env):** Student drink_p01 @imgsz640 @conf0.10: min 4 cams, mean 8.0, ≥2 100%, ≥3 100%, 0-cam 0%. XMem(seeded): min 6, mean 8.2, ≥2 100%, 0-cam 0%. **= essentially EQUAL coverage.** My earlier 'generic COCO YOLO mean 1.9, 37% blackout' was GARBAGE (wrong model in wrong env + wrong imgsz1280) — disregard it.

**REVISED VERDICT (this overturns the XMem case for tracking):** For the KNOWN cup, the trained student BEATS XMem on BOTH axes that matter — full multi-view coverage (≥4 cams/frame, equal to seeded XMem) AND steadier point (box 0.4 vs mask 0.8px under occlusion) — plus ~5× faster, no per-cam occlusion blackout. So XMem/SwissDINO add NOTHING to the tracking of a known object. Their only surviving niche = COLD START (new object, before a student exists) = the original 'faster onboarding' goal, nothing more. Script: /tmp/xmem_propagate.py. Viz lives in gitignored _swissdino_viz/.
