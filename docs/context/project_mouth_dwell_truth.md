---
name: project-mouth-dwell-truth
description: Head mocap (cleaned C3Ds) gives a mouth-based dwell truth that VINDICATES the hybrid segmenter on the DISAGREE reps
metadata:
  type: project
---

The cleaned QTM C3Ds (`cache/qtm_c3d_cleaned/`, same 772 stems as `qtm_c3d/`) add a
rigid 5-marker HEAD cluster — `FHD, L_FHd, R_FHd, L_BHd, R_BHd` — on 737/772 trials,
fully gap-filled. This closes the open frontier in [[project_learned_segmenter]] /
[[project_cup_pose_fusion]]: we now have a real MOUTH landmark, not a speed proxy.

- `qtm_c3d.CupTrial` gained `has_head()`, `head_frame()`, `mouth_proxy()`, `cup_to_mouth()`.
  Mouth = front-head centre offset down+fwd in the rigid head frame, DOWN offset
  **apex-calibrated per trial** so min(cup→mouth)=~50mm (cup rim at lips). Mild circularity
  (uses cup to set mouth depth) but the dwell keys off the distance RANGE, not the floor.
- `mouth_dwell.py`: van Andel dwell = cup→mouth stays within 15% of steady-state range above
  its min. Cached for all trials in `cache/mouth_dwell.json` — 762 trials, median dwell 1.96s
  [IQR 1.35–2.38]. New root constant `_paths.QTM_C3D_HEAD`; `QTM_C3D` (cup-only) left untouched
  so the accuracy pipeline is unaffected.
- **THE RESULT** (`validate_mouth_vs_hybrid.py`, uses `qtm_align.json` for the video↔c3d pair +
  sync lag): on all 3 DISAGREE reps the mouth truth shows the hybrid was RIGHT and the old
  speed-proxy was ~1s too short. P16 mouth 1417ms / speed 217 / hybrid 633; P23 1267 / 350 /
  833; P24 1767 / 550 / 1000. |hybrid−mouth| < |speed−mouth| on EVERY rep. The speed proxy
  caught only the motionless core; real drinks last 1.3–1.8s at the lips.

**Why it matters:** the DISAGREE slide flips from a caveat ("model might be right, can't tell")
to a measured finding — the hybrid's disagreements are corrections; the speed-gate label is
systematically late.

**Head as a FEATURE (`learn_seg_mouth.py`, cache/learn_seg_mouth.json):** cup→mouth distance +
approach-velocity + normalised-distance + present-flag added as 4 channels (via
`mouth_features.py`, which puts the mocap signals on the 60Hz track grid using the qtm_align
pairing + sync lag). LOPO 662 reps, TWO clf trained per fold against the MOUTH truth: base17
(13 fused + 4 occ, no head) vs mouth21 (+4 head). Result — the head feature is DECISIVE:
mean 390→130ms, p50 133→33, p90 1300→333; **562/662 reps better, 62 worse, 38 tie**. Unlike the
old 3D-direction/occlusion channels (tail-only), the head feature moves the whole distribution.
Biggest gains are P03/P08/P19 LEFT-side occluded-at-mouth drinks (−1.6 to −2.4s each) — the cup
is occluded so kinematics guessed; distance-to-mouth says "at the lips" directly.
CAVEATS: (1) `tuned` scores 885ms here only because it's measured against the mouth truth it was
never built for — not a regression, the truth moved; base17/tuned numbers are NOT comparable to
the old speed-proxy deck. (2) mouth21 tail still ~2.5s max, 62 reps regressed (likely sync/pairing
leakage) — inspect before headlining. Head feature is realistic (not oracle): plan is to get head
POSE from video later, so mouth21 is a deployable design with mocap head as the stand-in.

**Proxy failures are a FEATURE problem, not truth/sync:** audited the ~62 reps where the head
feature regressed — sync_corr median 0.98 (same as all reps), corr(sync,err)≈0; the mouth-TRUTH
dwell brackets the independent cup-apex window on every failing rep (truth robust: keys off curve
SHAPE not absolute value). Cause = the mouth proxy's FIXED down/fwd offset swings to the wrong
place under head TILT (P15 29°, P07 18°) → corrupts the fed distance, not the label.

**Head-representation comparison** (`learn_seg_mouth.py`, cache/learn_seg_mouth.json,
perrep_cols in file). Two rounds testing how to feed the head signal, all vs MOUTH truth, LOPO 662:
- Round A (cup-in-head-FRAME, rotated basis): proxy21 mean 121; headfrm21 alone worse (320) but
  recovers proxy failures; both25 (proxy+headfrm) best (mean 119, p95 700, max 2033).
- Round B (RAW points in SHARED rest/basis space + cup-to-marker DISTANCES, `mouth_features.
  points_channels`/`dist_channels`, `qtm_c3d.head_marker_pts`): proxy21 mean 134/p50 33/max 2533;
  **points33 (cup+5 markers, shared space) mean 403** and **dists23 (5 cup→marker distances) mean
  369** — BOTH barely beat base17 (372) as whole-distribution features (raw geometry is HARD to
  learn "at mouth" from). BUT on the reps the proxy HURT, points/dists recover them strongly (P03
  1383→333/200, P07 650→17/33, P12 ~625→~150/230).
- Round C (COMBO27 = base + proxy + distances, the design both A/B seemed to point at):
  combo27 mean 278/p50 50/p95 1566 — WORSE than proxy21 alone (134/33/816). vs proxy: 351 reps
  worse, 183 better. Naive concatenation POISONS the clean proxy signal — a single TCN can't
  "use proxy when good, distances when bad", it blends and the blend loses on the many easy reps.
  BUT combo still fixes the proxy's tilt-failures (P03 1383→83, P12 ~625→117, P07 650→17).

CONCLUSION (all three rounds): **SHIP proxy21 alone** — best single feature by far (mean 134 vs
base17 372 vs tuned 885). The computed mouth PROXY is a ready-made "distance to mouth" scalar the
TCN learns easily; raw geometry (head-frame / points / distances) is WEAK standalone AND degrades
the proxy when naively concatenated (round C). The ~12 tilt-failure reps ARE fixable by raw
geometry, but the fix is NOT more channels on one model — it needs a better (tilt-aware) proxy OR a
GATED/ensemble that uses distances only where the proxy is unreliable. NOTE round A's both25 (mean
119) looked like a win but round C (proxy+dists, mean 278) shows concat is fragile — treat both25's
edge as within-noise, not a robust "combo wins". Bug fixed en route: head markers ~25% gap-filled →
per-MARKER interp (not all-5-present). Residual ~2s max on P10-right/P19 resists EVERY rep (hard).

**CENTROID BUG + TRUTH REDESIGN (SUPERSEDES everything below — the numbers under here were on
a corrupted signal).** `CupTrial.centroid()` did `nanmean(ALL markers)`; the cleaned C3Ds add 5
HEAD markers, so the "cup" centroid was pulled ~318mm toward the head → cup_to_mouth was really
(cup+head avg)→mouth. The "double-minima / multi-apex drinks" that motivated the bridging fix were
THIS ARTIFACT (cup track swinging through the head markers), not real two-swallow drinks. Caught by
projecting markers into video (overlay_markers.py) — everything shifted. FIX: centroid() averages
only CUP_MARKERS. Verified on video (P06, 0.9mm Kabsch): cup markers on cup, one clean dip per drink.
TRUTH REDESIGNED (drops the mouth proxy entirely, for the future 1-landmark biomech head model):
dwell = cup-centroid → HEAD-centroid distance within 15% of the apex→rest range; rest = cup→head
distance at the reach's start+end (walk out from apex to 70%-of-max), averaged; plain longest-run,
NO bridging (verified: 1 clean run/rep now, bridging was patching the artifact); 7-frame smoothing.
`qtm_c3d.head_centroid()/cup_to_head()`, `mouth_dwell.dwell_truth` rewritten. Truth median dwell
1.60s. CLEAN LOPO (learn_seg_mouth.json, 666 reps, cup→head signal): **proxy21 (cup→head dist)
mean 34ms / p50 17 (≈1 frame) / p99 173, beats base17 (114) on 510/666**; points33 106, dists23 130
(raw geometry weak standalone, same as before). proxy21 max 2417 is essentially ONE rep
(P14_R_092230, base133→proxy2417; points/dists recover it) — not systematic. Prior-signal caches
archived as *_oldsignal.json / *_buggycentroid.json / *_oldmouth.json.

**DEPLOYMENT-REALISTIC feature (current learn_seg_mouth.json):** proxy21's cup→head distance is
now TRACKED cup (d['fused'], the noisy video track) → mocap head-centroid, both in W0 via per-rep
robust Kabsch (`mouth_features.tracked_cup_to_head_channels`). Only the head is a mocap stand-in
(for the future VIDEO head landmark); the cup is the real track. TRUTH stays mocap→mocap.
Realistic LOPO 666 reps: **proxy21 mean 77ms / p50 50 / p90 150 / p99 368, beats base17 (118) on
408/666** (mocap-cup ORACLE was 34ms/510-better, archived learn_seg_mouth_mocapcup.json — the
34→77 gap is the tracking-noise cost, ~5mm median cup diff). points33 108, dists23 115. **77ms is
the honest number to report** (tracked cup + head landmark = deployment); base17 118 = video-only.
P14_R_092230 = the one robust outlier (proxy 2417, base 333) across all runs. Head-distance
feature is the lever. ⬇ everything below is PRE-CENTROID-FIX (corrupted signal) ⬇

**METRIC BUG FOUND + FIXED (the big one) [PRE-CENTROID-FIX, on corrupted signal]:** the worst proxy21 reps weren't model failures —
the mouth TRUTH was chopping MULTI-APEX drinks. A drink can dip to the mouth, rise over a small
HUMP (re-approach / 2nd swallow), dip again = ONE drink; `_longest_run` split it at the hump and
kept one apex. 8/9 worst reps were chopped (P10 R kept 1.04s of a 3.45s drink). Verified via
`plot_worst.py` (slides/worst_proxy21_grid.png) + disputed-tail check: in the frames TCN called
drink past the (chopped) truth end, cup was 44–72mm from mouth (apex ~35) = STILL AT THE LIPS →
TCN right, truth wrong. Fix = `mouth_dwell._bridged_span` (leave_frac=0.40): bridge below-thr runs
whose separating hump stays below the "cup left the mouth" level. Population-safe: median dwell
1.96→2.09s, p90/max unchanged, 25% reps grow +0.37s mean, 0 reps >4s. Truth regenerated
(`cache/mouth_dwell.json`; old → `mouth_dwell_prebridge.json`). RE-SCORE vs bridged truth
(`rescore_bridged.py`, cache/learn_seg_mouth_bridged.json): **proxy21 mean 134→53ms, p50 33→17,
p90 415→67, max 2533→2067** — the model was ~2.5× better than the chopped metric reported; 17ms ≈
1 frame. base17 372→144; tuned 885→993 (worse — gate targets speed-proxy, further from the longer
correct truth). Also found: some "worst" reps (P03 L, cached err 1383) were LOPO training NOISE —
retrain to ±1 frame, not real failures. This SUPERSEDES the attention/combo modeling thread: the
lever was the METRIC, not the architecture. (proto_attn.py TCN+attn edge was single-split noise.)

**How to apply:** headline proxy21 vs bridged truth (mean 53ms). Deck slide 9 + `cache/rrd/README.md`
should cite it. Next: add cup→mouth / head-frame to the Rerun viewers so the supervisor sees the
signal; eventually swap mocap head for a video head-pose proxy (same `cup_in_head_frame` def).
