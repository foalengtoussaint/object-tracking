---
name: project_reproject_fill_filter
description: "drink_study 3D label filter, reproject-fill variant (run_clean3d_fill.py): strict 30px reject for real detections + bbox reproject-fill for non-detecting cams in >=3-cam consensus frames. P01 student, held-out F1 early-stop. Wins on 3D PRECISION (3.15px vs 4.2-4.4) but cam10 recall 0.21 < reject's 0.51 -- fill did NOT beat reject on the hard camera."
metadata:
  type: project
---

New training-data filter `experiments/drink_study/run_clean3d_fill.py` (vs reject-only `run_clean3d.py`).
Idea: reject-only over-pruned (~28% of labels) so its mean recall trailed crude drop_cam10. Fix = keep strict
30px reject for cameras that DID detect (so cam_10's static glass, 942px off consensus, stays rejected) BUT
reproject-FILL a synthetic bbox (square polygon sized by reprojecting a 35mm offset = cup apparent radius; YOLO
is a SEG model so labels must be polygons not boxes) for cameras that did NOT detect in a >=3-cam-consensus frame.

**Label pool (P01, 2 reps):** kept_real=4576, filled=3604, dropped=1747. Fills concentrate on sparse cams
(cam6/8/9 each gained ~700-800). cam_10 = 0 real / 0 fill / 1357 drop (it always detects the glass -> always hits
the reject branch, never the "didn't detect -> fill" branch).

**Trained P01 student, held-out F1 early-stop (best ep3, F1 0.842).** Per-epoch curve: recall climbs 0.30->0.80
while GATED 3D precision stays tight throughout (px 2.5-4, ~8 cams agree) -> training buys COVERAGE, precision
was always good. Used the corrected GATED agreement metric (post-gate >=3-cam consensus, eject worst-reproj cam),
NOT the old raw inlier_frac.

**Held-out (P06/P19/P23) comparison (gated 3D precision + per-cam recall):**
| variant | mean recall | cam10 | 3D-prec px | tri_rate |
|---|---|---|---|---|
| raw (glass) | 0.700 | 0.00 | 4.41 | 0.898 |
| drop_cam10 | 0.789 | 0.13 | 4.40 | 0.956 |
| reject | 0.738 | 0.51 | 4.21 | 0.834 |
| **fill** | 0.757 | 0.21 | **3.15** | 0.897 |

**Findings (honest, mixed):**
1. **Fill WINS on 3D precision** (3.15px, clearly tightest inter-camera agreement) and restores coverage
   (tri_rate 0.897, vs reject's over-pruned 0.834) -> it did fix reject's coverage drop without losing precision.
2. **Fill did NOT beat reject on cam_10** (0.21 vs 0.51) -- the counterintuitive result. cam_10 is never filled
   (always detects the glass), and the crude square fills elsewhere may dilute the cup signal vs reject's cleaner
   sparser set. So reject-only remains the cam_10 champion; fill trades a bit of cam_10 for big precision + coverage.
3. Mean recall: fill (0.757) > reject (0.738) > raw (0.700), still < drop_cam10 (0.789).

**UPDATE — REJECT-THEN-FILL wins everything (run_clean3d_fill.py, CFG=pscale_1_clean3d_refill):**
Changed the failed-gate branch: a camera that DETECTS but disagrees with consensus (cam_10 glass, 942px off)
is no longer dropped -- its wrong detection is SUPPRESSED and REPLACED by the cup reprojected from the >=3-cam
consensus. cam_10 went 0 -> 912 refilled labels. Trained held-out-F1 early-stop; per-epoch re-scored for 3D-F1
(recall x consensus-inlier precision) + train-F1; 3D-F1 plateaus ~0.86-0.88 from ep5 (variance), picked ep6 as
best_3df1.pt. Final 5-way held-out (P06/P19/P23):
| variant | mean recall | cam10 | 3D-prec px | tri_rate |
|---|---|---|---|---|
| raw | 0.700 | 0.00 | 4.41 | 0.898 |
| drop_cam10 | 0.789 | 0.13 | 4.40 | 0.956 |
| reject | 0.738 | 0.51 | 4.21 | 0.834 |
| fill | 0.757 | 0.21 | 3.15 | 0.897 |
| **refill** | **0.803** | **0.74** | **2.99** | 0.917 |
**refill WINS every axis**: cam10 0.74 (>> reject 0.51, the old champion), mean recall 0.803 (> even crude
drop_cam10), tightest 3D precision 2.99px, coverage restored. Overturns the earlier "reject still cam10 champion"
finding -- reject-THEN-fill gets BOTH the cam10 recovery AND precision/coverage. Also: 3D-F1 ~= 2D-F1 throughout
(3D-prec ~0.95 -> not confident-wrong); train-F1 0.88-0.90 sits modestly above held-out 0.84-0.88 (healthy gap).
Early-stop drives on 2D-F1 but best re-picked by 3D-F1 (recall x consensus-inlier precision) post-hoc.

**Still open:** (a) better fill mask than a square (reproject real cup polygon shape); (b) make 3D-F1 the LIVE
early-stop signal (currently 2D, re-picked post-hoc); (c) cup apparent-size world coords are MM (bug fixed: was
/1000 -> 6px floor everywhere).

**Box-size idea (PARKED 2026-06-15, user undecided):** on held-out cam_10 the refill student's box is ON the cup
but TOO SMALL/square -- because fill labels are sized as a 35mm SPHERE, while the cup is taller than wide. User
suggested sizing the fill box from the OTHER cameras' detections (reproject their 3D cup extent into the fill cam).
BUT: cached teacher dets are CENTROIDS ONLY (no box dims) -> needs a teacher re-run to cache xyxy (detect_rep
already computes xyxy at agreement.py:57 but discards size). KEY REframe: the 3D pipeline (triangulate/consensus/KF)
only ever uses the CENTROID -- box size is never consumed downstream, so the small box is COSMETIC (doesn't affect
recall/precision/tri_rate/track). So box-sizing only matters for viz or a future non-centroid consumer. Cleanest
alternative = switch student to a KEYPOINT/pose head so the label IS the centroid (no box, matches what's used).
Options when revisited: (1) teacher re-run -> cache boxes -> reproject 3D extent from other cams; (2) keypoint head;
(3) do nothing (centroid suffices). Verify tools: viz_consensus.py (per-frame >=3-cam banner), viz_grid.py,
viz_sidebyside.py. P06 tri_rate 1.0, P19 0.986 (8/573 no-consensus), P23 0.765 (88/374 -- short sparse trial).

**'2 epochs enough' VERIFIED (verify_2epoch.py):** mean recall 0.803 + cam10 0.74 are FULLY reached at EPOCH 1,
flat through ep6; ep6 only sharpens 3D precision (3.27->2.99px) + coverage (tri_rate 0.87->0.92). One epoch solves
presence incl. the cam10 recovery; later epochs just tighten localization.

**SELF-DISTILLATION (run_selfdistill.py, 2 epochs):** trained a fresh student on the REFILL student's own
detections through the SAME 3D-gate+refill filter (student dets, not teacher). Label pool transformed: kept_real
4576->11229, filled 3604->1688, refilled 940->93, dropped 1747->106. **cam_10: 0 real/912 refill (teacher) ->
1050 real/0 refill (student)** -- the student now detects cam10's REAL cup natively; the glass is gone, no
geometric refill needed. **CORRECTED RESULT (a BUG invalidated the first pass):** `per_cam_eval` (a) cached by CFG only, NOT checkpoint,
so the 4-epoch self-distill run REUSED the 2-epoch run's cached detections -> reported stale 0.803/0.74; AND
(b) it measures DETECTION RATE (any box fired), NOT recall -- no GT, no cup/FP check. FIXED: cache key now includes
a checkpoint hash; metric renamed detection-rate. Re-scored with the fix (held-out P06/19/23, det-rate + gated 3D):
| student | det-rate | cam10 | tri_rate | med_px |
|---|---|---|---|---|
| refill (parent) | 0.803 | 0.74 | 0.917 | 2.99 |
| self-distill 2ep | 0.857 | 0.94 | 0.987 | 3.52 |
| self-distill 4ep | 0.927 | 0.92 | 0.993 | 3.14 |
**Self-distillation IMPROVES over the parent (NOT just matches -- that was the bug).** cam10 0.74->0.94 is the big
win: native cam10 detections (1050 real labels, 0 refill, vs teacher's 0 real/912 refill) train a much stronger
cam10 student. Extra detections are REAL not FPs -- confirmed by the 3D precision metric (tri_rate 0.987-0.993,
med 3.1-3.5px; if FPs, tri_rate would drop / px blow up). 2ep ~= 4ep (within noise) so "2 epochs enough" holds, but
at a HIGHER level than the buggy 0.803 suggested. cam10 recovery is SELF-SUSTAINING (student generates clean cam10
labels itself). LESSON: detection-rate has no correctness bound (FPs inflate it) -> always read it WITH the 3D
gated precision. The original 2-epoch model is gone (4ep run overwrote same run-dir/CFG); 2ep numbers above are the
4ep-schedule's epoch1 (different LR schedule, but a real verifiable early point).
Caching: teacher dets, 8180 fill labels (.done marker), per-epoch ckpts, eval_by_epoch.json,
heldout_3dprecision.json all cached. agreement_eval hardcoded clips path FIXED -> CLIPS_ROOT.
Relates to [[project_3dclean_vs_dropcam]], [[project_cam10_apparent_size]], [[project_label_kf_is_2d_only]],
[[project_failure_modes_confident_wrong]], [[project_robustness_envelope]].
