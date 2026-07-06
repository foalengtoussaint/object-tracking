---
name: project_failure_modes_confident_wrong
description: "drink_study robustness: failures are overwhelmingly CONFIDENT-BUT-WRONG (213/228 bad conditions: cov>80% yet median 1588mm, up to 35m), not graceful track-loss. Predictor false positives cluster where det_rate/tri_rate look great but cameras don't actually agree. inlier_frac as currently measured does NOT separate good/bad (median ~0.01 for BOTH) -> likely a metric bug worth fixing."
metadata:
  node_type: memory
  type: project
  originSessionId: bdb7162a-6c96-4391-8f59-078e4c60445f
---

From `robustness.py` false-positive analysis (the thing we care most about: when does the pipeline output a confident WRONG position).

**Failure-mode split (of 228 bad conditions): 213 catastrophic vs 15 graceful.** Catastrophic = coverage>80% yet p99>200mm (confident, far-wrong — locked on a distractor / degenerate fusion). Median catastrophic error **1588mm**, max **35432mm**. So when this pipeline fails it almost always fails LOUD and WRONG, not by quietly losing the cup. A wrong-but-confident track is worse than no track — this is the deployment risk.

**Predictor false positives (said good, was bad): 8/26 of "predicted-good" calls (~31% FP rate among positives).** 7 of 8 share one signature: det_rate~0.92, tri_rate=1.0, n_cams=10 (all look great) BUT inlier_frac~0.01 — every camera detects & triangulates SOMETHING, but to a point none agree on. These are the corrupt>=60% and sigma>=40px conditions. The predictor is fooled because det_rate/tri_rate carry positive weight and look fine.

**Combined gate works anyway:** keep if `pred_good>=0.5 AND inlier_frac>=0.1` drops FPs 8 -> 1.

**CAVEAT / likely bug:** inlier_frac as stored does NOT discriminate — GOOD conditions have median 0.014, BAD have 0.051 (good is LOWER!). Its solo AUC is ~0.51 (coin flip). A standalone inlier_frac veto costs ~56% of good to catch ~55% of bad. My initial "every FP has low inlier_frac" was right but non-discriminative (good has it too) — I jumped on 8 rows without the base rate. **Inter-camera agreement SHOULD be the best confident-wrong early-warning, but as measured it isn't** — suspect it's computed on the post-gate camera set / saturated. Fixing how inlier_frac (or a pre-gate agreement metric) is measured is the open task.

**Open next steps (not yet done):** (1) fix the agreement metric so it actually separates good/bad; (2) build a runtime confident-wrong guard (post-fusion reprojection-residual spike detector) so deployment can SUPPRESS a wrong-but-confident output; (3) validate the predictor on REAL checkpoints (pscale_0..4) not just synthetic degradations of one trial.

**UPDATE (2026-06-11 re-analysis, `false_positives.py`):** re-ran on cached rows. Confusion @0.5: TP=9 FP=11 FN=35 TN=214 -> **55% FP rate among predicted-good** (worse than the earlier "8/26 = 31%"; the count depends on the cached CV split). FPs split into TWO families: **(A) camera-disagreement = 8 FPs, median 1588->1196mm — the dangerous ones** (det~.92, tri=1.0, n=10 all look great, inlier~.015; corrupt>=40% or sigma>=40px); **(B) too-sparse = 3 FPs, median 84mm** (extreme dropout, KF coasts just past cup — mild). **Correction to earlier note:** inlier_frac isn't merely non-discriminative, its solo AUC is ~0.55 and within the high-quality-looking subset (det>.5,tri>.8) it's **INVERTED (AUC 0.32 — good has LOWER inlier_frac)**. reproj_px looks weak globally only because exactly-determined 3-cam GOOD tracks show reproj up to ~1150px (residual meaningless when not over-determined). **No cheap raw-det threshold cleanly separates** — every guard that kills the disagreement FPs also kills good sparse/3-cam tracks (best combined inlier>=.1 & tri>=.1 still loses 6/9 good). Confirms the fix must be a POST-GATE agreement metric (RANSAC inlier set), not the raw-camera metric. Also built `explore_robustness.py` (interactive slider: hold one degradation axis, watch the other 4 envelope curves lift) — shows corrupt>=60% collapses tolerance to ALL other axes (whole envelope -> FAIL), i.e. corruption is not independent.

Relates to [[project_robustness_envelope]], [[project_cam10_apparent_size]], [[project_kf_accuracy_budget]], [[project_agreement_cam4_dominates]].
