---
name: project_robustness_envelope
description: "drink_study: how degraded can a detector be and still track? Pipeline shrugs off 90% dropout / 50% corrupted cams / 15fps individually; ONLY hard floor is >=3 geometrically-agreeing cams. Predictor (logistic AUC .81): inter-camera AGREEMENT (inlier_frac/det_rate) is the dominant success driver, not precision."
metadata:
  node_type: memory
  type: project
  originSessionId: bdb7162a-6c96-4391-8f59-078e4c60445f
---

`experiments/drink_study/robustness.py` — predictive usefulness model. **Ground truth = pscale_4 clean gating->KF->RTS on ONE visually-verified trial (P01 141546, 870/870 coverage, confirmed on-cup in the Rerun replay), frozen as X_truth(t).** Degrade THAT trial's cached student dets along 5 axes; outcome = worst-1% drift of the **post-pipeline** estimate from frozen truth (good = p99<35mm cup radius AND cov>0.5). 263 conditions (per-axis sweeps + random multi-axis combos). No GPU.

**Outcome is scored AFTER gating->KF->RTS** (that's what we care about — the filter's best rescue), but the predictor INPUTS are measured on the RAW degraded dets (pre-pipeline, no-GT — what you'd actually have on a new model): det_rate, n_cams, reproj_px, inlier_frac, tri_rate, fps_eff.

**Recoverable-quality envelope (single axis, others clean — the pipeline is REMARKABLY robust):**
- **dropout -> 90%**: still 100% on-cup (11mm). KF predict() coasts on 1-in-10 frames.
- **corruption -> 50% of cams shoved 120px off**: still 100% on-cup. The >=3-cam inlier gate ejects outliers cleanly.
- **frame-rate -> 15fps (decim 4)**: 100% on-cup. Motion model handles the drink reach/retract.
- **sigma jitter**: holds to ~10px (23mm @ 20px) — same ±20px budget as [[project_kf_accuracy_budget]].
- **n_cams = THE ONE HARD FLOOR**: 2 cams = 0% (9494mm, relocks onto cam_10 glass); **3 cams = 100% (14mm)**. Cliff is exactly the >=3-consensus gate. (2-cam seeded from frozen truth so this is a TRACKING failure not a seeding one — degenerate depth ray.)

**So the "minimum useful model" bar: deliver >=3 geometrically-agreeing cameras per frame.** It can be sparse (10% det rate), imprecise (±10-20px), low-fps, and have HALF its cams lying — the filter recovers as long as >=3 agree. Precision/density are cheap; cross-camera AGREEMENT is the currency.

**Predictors (5-fold CV, sklearn):**
- LOGISTIC P(good) **AUC 0.806**: standardized weights inlier_frac +1.22, det_rate +0.91, fps_eff +0.71, tri_rate +0.50, n_cams +0.38, reproj_px -0.22. Agreement-type metrics dominate.
- LINEAR log10(p99_mm) **R2 0.426** (heavy-tailed/bimodal — good~12mm then >1000mm cliff, so log-linear underfits the cliff; logistic is the right tool, linear only confirms direction). inlier_frac -0.49 most reduces error.
- SINGLE-metric go/no-go (Youden threshold): best is **tri_rate >= 0.67 (AUC .688)**, then n_cams>=6 (.668), reproj_px<=816 (.647). No clean solo gate — failures are multi-causal — but the combination predicts well and agreement carries it.

**How to apply:** to vet a NEW detector for this rig cheaply (no GT, no full sweep), measure its raw-det inlier_frac / tri_rate / det_rate on a rep; feed the logistic to predict P(track good). Hard precondition: >=3 cams must agree per frame (it's the only cliff). reproj_px alone is a weak solo gate here (the gate already rejects outliers, so global reproj doesn't separate good/bad until it's catastrophic). Relates to [[project_kf_accuracy_budget]], [[project_agreement_cam4_dominates]], [[project_label_kf_is_2d_only]]. Built alongside the offline Rerun replay `experiments/drink_study/viz_replay.py`.
