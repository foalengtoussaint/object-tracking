---
name: project-biomech-omc-head-cup-align
description: "Aligned markerless biomech (W0) to OMC/QTM per-rep via head+cup Kabsch. Per-participant SCALE fix (P24/P16/P23/P02 ~5% compressed calib) rescues head to 4.3mm (pop 5.1mm); drink-apex cup still the open floor. Renders use the DUMB-PLAYER pattern (precomputed draw_points npz) so pixels==reported numbers. 6 wrist-swap OMC defects excluded."
metadata: 
  node_type: memory
  type: project
  originSessionId: 25d11f20-b722-49a2-b616-7d5262e468ad
---

Per-rep rigid mocap->W0 Kabsch on the COMBINED head+cup point cloud (user asked to align
biomech to OMC using BOTH markers). Script: scratchpad `align_head_cup.py`; output
`experiments/drink_study/cache/align_head_cup.json` (per-rep R,t + combined rms + head/cup
residuals; per-participant summary). Reuses the [[handoff_wire_session_R_into_pipeline]] sync:
`features.mocap_to_w0` convention — resample(video→VIDEO_FPS, mocap→rate) to COMMON_HZ, apply
qtm_align `lag`, Kabsch with hard-exclude+readmit (EXCLUDE_MM=20 for the combined cloud).

Sources: biomech head = keypoints3d[:,67,:3] (biomech_<stem>.npz, W0 mm); biomech cup =
track3d_clean3d_refill[...]['rts']; OMC head = mocap.head_centroid() (5-marker cluster);
OMC cup = mocap.centroid() (4 cup markers). qtm_align.json gives rep→c3d/lag/side (22 pids).

## RESULT (2026-07-09): 146 reps aligned, 0 fit failures, overall MEDIAN 6.4mm
Head resid ~2-10mm (med ~4), cup resid ~3-16mm (bigger: drink-phase cup tilt makes the mocap
4-marker centroid vs video track measure slightly different points — the exclude-readmit handles
it). KEY WIN: head+cup TOGETHER removes the symmetric-round-cup rotation ambiguity that forced the
session-R workaround — the old degenerate reps converge per-rep now: P16 3.9, P19 6.9, P24 2.5;
P23 highest at 12.4 but fine. So for biomech→OMC, per-rep both-marker fit is enough; no session-R
needed. Supersedes the cup-only-alignment premise in [[project_session_R_ab_confound]].

## P05 = biomech head is DEAD (data bug, not alignment)
P05 biomech head67 is 100% all-zero / conf=0 on every rep (vs P04 median conf 0.79) → P05 aligned
on CUP ONLY (n_head=0, cup resid 1-4mm excellent). Likely person-pick/joint-67 dropout in P05
footage. Flag for the biomech re-run. All other 21 pids have proper head+cup fits.

## OMC quality filtering — FLAG-ONLY (user choice 2026-07-09)
Some QTM C3Ds are corrupt; aligner now MEASURES per-C3D quality and TAGS (never drops) each rep:
`omc_flags` ([]=clean) = frozen_cup (cup bbox travel <20mm, the known [[project_session_R_ab_confound]]
frozen-cup corruption), gappy_head (OMC head-cluster valid frac <0.90), bad_sync (qtm sync_corr <0.90),
cls_broken. Per-rep also stores omc_head_valid_frac, omc_cup_travel_mm, sync_corr, qtm_cls, used_head.
Top-level `flagged_omc` lists them; summary has n_flagged/n_cup_only per pid. Filter downstream yourself.
ONE non-negotiable: when OMC head is too gappy the HEAD CHANNEL is dropped from that rep's fit (cup-only,
rep still kept) — a dead head cluster would corrupt the rotation (fit-correctness, not filtering).
Current 146-rep set: only 2 flagged (bad_sync 0.84/0.90); 0 frozen-cup/dead-head YET (those reps'
biomech npz not computed) — gates will auto-catch P08/P12/P19 frozen cups as the run fills them in.

## Per-phase residual + the static-cup tail (2026-07-09)
Residual by phase (146 reps, ALL frames): HEAD flat ~4mm in static/moving/drink (seated subject =
always-static anchor). CUP climbs 7→23→46mm static→moving→drink (~4-6x, drinking-worse in 93% of
reps) — matches all cup-only variants in lopo_fused (kf/hard/fused all ~4x, drink ~16-17mm vs true).
So the drink-phase cup concentration is INTRINSIC to cup tracking (tilt+occlusion at the mouth), NOT
caused by biomech or the alignment; adding biomech didn't move where the cup fails.

STATIC-phase question ("2 stationary points → tighter fit"): TRUE for 82% of reps (static cup ~7mm,
head ~4mm) — but static cup p90=82mm from an ~18% tail. Diagnosed: NOT phase-mislabel (bad frames
4441 interior vs 276 boundary) and NOT corrupt OMC (cup travels 470-780mm, no flags) and NOT track
glitch (P24 rest: OMC cup spread 1.7mm, biomech cup spread 0.8mm — both dead still). It's a CONSTANT
CUP OFFSET (~80mm) that survives even at rest: the markerless cup track and the OMC 4-marker centroid
lock onto DIFFERENT physical points on the cup body (cup-definition/calibration offset), so the cup is
the weak partner. The HEAD (4mm, same physical landmark) carries the fit — which is exactly why
head+cup beats cup-alone: the head rescues reps whose cup has an offset. The exclude-readmit hides this
by dropping ~half the frames (n_used≈0.5·n_total on offset reps) → clean 6.4mm median but fat all-frame
tail. Reps affected skew to P08/P13/P15/P19/P24. TODO if it matters: reconcile the video-cup point vs
OMC cup-centroid definition (a fixed transform), or trust the HEAD channel for validation and treat the
cup as advisory.

## Outlier policy: NOW NO-EXCLUSION (user choice 2026-07-09)
fit_head_cup gained `exclude` (default FALSE): plain robust(Huber) Kabsch on ALL head+cup frames,
no hard 20mm drop, no readmit — honest whole-rep fit. The old exclude-readmit flattered the number:
overall median RMS 6.4mm (excluded) → **18.9mm (all frames)**; offset reps P07/P12/P13/P24 jump to
35-45mm. HEAD stays ~4-6mm even with no exclusion (genuinely good, not propped up); CUP is what blows
up (P13 48, P24 29, P19 28mm) = the constant cup-centroid-definition offset. Downside of no-exclusion:
on reps with a LARGE cup offset (P16/P24) the offset now drags the rotation so the head fit degrades too
(head 15-18mm) — exclusion had protected the head by dropping bad cup frames. Cache align_head_cup.json
now holds the no-exclusion fit. So: 19mm is the true agreement; the cup offset is real, not a one-frame
artifact; head is the trustworthy channel.

## Head vs cup: bias AND variance BOTH decomposed (2026-07-09) — corrects "head=clean anchor"
Measured at rest (140 reps), separating |mean residual vector| (BIAS) from scatter (VARIANCE):
- HEAD: bias 4.9mm, var 3.9mm; rest JITTER 0.61mm/frame (~30x the cup), rest wander ~9mm.
- CUP : bias 7.3mm (up to 80 on offset reps), var 2.1mm; rest jitter 0.02mm/frame (rigid).
KEY: the HEAD bias is NOT a fixed offset — direction-consistency 0.32 (mean head-bias vector only
2.3mm; each rep's ~5mm offset points a different way, they cancel). Cause = OMC head-cluster centroid
depends on which of 5 markers are visible + subject RE-SEATS between reps → head "definition" shifts
PER-REP (user was right: head definition is variable). The CUP bias IS structural/fixed (video cup
point vs mocap 4-marker centroid) → correctable with one offset transform.
COROLLARY inverts "trust the head": HEAD = per-rep irreducible bias+variance (can't calibrate away,
handle by averaging/down-weighting); CUP = low-variance + FIXED bias (subtract it, then cup is the
better anchor). Both carry ~5-7mm definition offset — neither is clean; they fail in opposite ways,
which is why head+cup together constrains better than either alone.

## Rest-only fit test (2026-07-09) — judge by GOOD-FRAME FRACTION, not median (I got this wrong first)
Fit transform on REST frames only, predict whole rep (rest=in, move/drink=out) vs all-frames; both
plain robust Kabsch. FIRST reported medians only and declared rest-only "worse" — VIOLATED
[[feedback_dont_claim_definitive]] + [[feedback_good_frame_fraction_not_average]]; user called it out.
Redone as %frames-below-threshold (restfit/allfit):
  HEAD good@15mm: rest 81/84  move 80/84  drink 82/93
  CUP  good@15mm: rest 85/80  move 33/39  drink 0/4   (@40mm drink 13/44)
CORRECTED conclusions: (1) rest-only fit is BETTER on rest-cup by good-frame count (85 vs 80% @15mm) —
the median hid this; for rest-phase cup accuracy, fitting on rest WINS. (2) head transfers from rest
with only mild loss (~5-9pts). (3) drink-phase cup is unusable under EITHER fit (0-13% good) → it's a
cup-TRACKING-at-the-mouth floor (tilt+occlusion), NOT a fit-choice difference — the median 56-vs-46mm
made it look like fit choice mattered; good-frame view shows both fail. So rest-only = genuine TRADEOFF
(better rest-cup, slightly worse head elsewhere), not strictly worse. Cup-offset-vs-tilt still plausibly
orientation-dependent but NOT yet proven (don't claim it from these numbers alone).

## Does head+cup rescue the degenerate reps? (2026-07-09) — PARTIAL, conditional on head quality
Prior work: [[handoff_wire_session_R_into_pipeline]] / [[project_session_R_ab_confound]] — per-rep
CUP-ONLY Kabsch flips the rotation branch on round-cup-symmetry reps (P16/P19/P23/P24), needing a
per-SESSION R rescue. Tested whether HEAD+CUP per-rep removes the flips (branch-flip = per-rep rotation
>20° from head+cup chordal-mean consensus): cup-only 71 flips → head+cup 39. BUT read per-rep residuals,
not the flip flag:
- RESCUED by head+cup (were cup-only degenerate, now agree per-rep, NO session fit): P11 5→0, P12 6→0,
  P17 6→0, P20 5→0 (maxdev 65→9°). Head anchors them (head resid ~5-7mm). Real win.
- STILL BAD: P16/P19/P23/P24 — head+cup does NOT rescue; because on THESE reps the HEAD is ALSO degraded
  (P16 head 17.9mm, P24 15.0, P23 11.4, rms 25-43mm). A noisy head can't anchor → still need session
  rescue or they're occlusion-broken. The original worst-4 are unchanged.
- FALSE ALARMS in the flip count: P07/P14 flagged >20° but head 4.9/5.9mm + cup 6.4/9.2mm = fits are
  FINE; the chordal-consensus metric misfired. So "39 flips" overstates failures.
RULE: head+cup rescues iff the head is good (~5mm) on that rep; where the head is also bad, no rescue.
Conditional, not universal. [[feedback_dont_claim_definitive]] — verified via per-rep residual split,
not the single flip count.

## REST-ONLY head+cup fit is BEST for rescue (2026-07-09) — but flip-count lied, use good-frame%
Fitting head+cup on REST frames ONLY (exclude drink/move where cup tilt-offset + occlusion corrupt):
flips>20° = 140 (rest cup-only) / 80 (all-frames head+cup) / **17 (rest-only head+cup)**. Principled
version of exclude-readmit: fit on the physically-clean phase. BUT the flip count is a confident-wrong
metric — a rotation can agree with consensus yet place points badly. Verified with REST good-frame%
(<15mm): GENUINELY rescued = P07 (head100/cup100), P23 (98/100), + P11/P12/P14/P17 clean. NOT rescued
despite flip=0 = P16 (head23/CUP 0% good — flip count FALSELY called it rescued). Still broken = P19
(0/0, frozen-cup participant), P05 (head dead/nan, cup 100 — cup-only would work). So rest-only rescues
MOST previously-hard reps with NO session fit, but P16/P19/P05 remain data-broken. Lesson (again):
flip-count/consensus-agreement ≠ good fit; confirm with good-frame fraction per [[feedback_good_frame_fraction_not_average]].

## Windows QTM operator bad-list cross-check (2026-07-09) — STALE for dead-takes
`~/Documents/qtm_takes_to_inspect.txt` (also cache/) = operator's bad-list, 36 takes: 14 step/marker-
swap, 14 dead-take(lift 0mm), 3 despiked, 6 broken(passes gate, won't align to video). Everything else
implicitly GOOD. 9 overlap my aligned set. FINDINGS:
- dead_take P14_0031/33/34/35/36/42/43: Windows says "lift 0mm" but the CLEANED C3Ds
  (qtm_c3d_cleaned/) I load show 375-426mm cup Z-lift, clean head, my fit head 4-7/cup 7-12mm = GOOD.
  The bad-list was made from the OLDER frozen-cup export ([[project_session_R_ab_confound]]); the
  cleaned re-export FIXED the cup. => Windows dead-take flags are STALE; DON'T exclude them. Trust cleaned C3D.
- broken P13_0029/0030: my omc_flags=[] (gates missed) but cup resid 88-96mm, rms 61-64 = my FIT caught
  them. My frozen/sync/head gates have a BLIND SPOT for the "broken" class (subtle GT defect).
FIX identified: add a POST-FIT residual gate (cup_resid_med>40 or head_resid_med>15mm). It flags 23 reps
incl the 2 broken + P08_0028(cup115). BUT most of the 23 are P16/P19/P24/P15 = MARKERLESS/occlusion
failures, not OMC defects. KEY DISTINCTION: "OMC-good" (Windows: is the C3D clean) ≠ "good fit" (biomech
↔OMC agreement). For validation: exclude Windows OMC-bad takes FIRST, then measure agreement on clean-OMC.

## WRIST-SWAP OMC defect confirmed by video + motion-correlation (2026-07-09)
Watched overlays: on P08_0028, P13_0029/0030, P19_0028/0031/0032 the OMC "cup" markers move like the
WRIST, not the cup. User's precise framing: it doesn't SIT on the wrist (fit is wrong so nothing lands
right) — it FOLLOWS the wrist's MOTION. My first test (distance "does OMC cup land on wrist" through a
head-fit) was WRONG (muddled "cup ok") — measured absolute position through a broken transform. Correct
FIT-INDEPENDENT test = correlate SPEED profiles (|vel| per frame, invariant to R,t): OMC-cup~bioWRIST
0.93-0.96 vs OMC-cup~bioCUP 0.72-0.85 on ALL 6 → confirmed wrist-swap. [[feedback_dont_claim_definitive]]
[[feedback_shared_code_metric_and_render]] — eye was right, first number measured the wrong quantity.
IMPLICATION: P08/P13/P19 are CORRUPT OMC (cup markers mislabeled onto wrist), NOT markerless/occlusion
failures — EXCLUDE from validation. This SPLITS the old "P16/P19/P24 hard-to-rescue" bucket: P19 = OMC
wrist-swap defect; P16/P24 = good cup GT but bad HEAD landmark. Different problems, different fixes.
overlay.py now draws biomech head67 (WHITE dot) vs OMC head (blue X) so the head disagreement is visible.

FULL SWEEP (146 reps, speed-corr wrist-swap detector, cache/wrist_swap_sweep.json): exactly 6 wrist-swaps
= P19_0028/0031/0032, P13_0029/0030, P08_0028 (all LEFT side; wrist_corr 0.93-0.96 vs cup_corr 0.72-0.85).
No others hidden. CRUCIAL SPLIT: of 20 high-cup-resid(>25mm) reps, only 6 are wrist-swaps → the OTHER 14
(P16/P24-type) have GOOD cup GT but bad HEAD/occlusion = genuine markerless cases to KEEP+report. Do NOT
just drop all high-resid reps — that would discard 14 good-GT reps that validation should test on.

## ROOT CAUSE of the "hard 14" reps = PER-PARTICIPANT SCALE error, not tracking (2026-07-09)
User hunches ("rotation should be better", "maybe a zoom thing") — BOTH right. The 14 high-resid reps
have head+cup BOTH 100% valid, low jitter (0.5-0.8mm/fr), both BIASED off — signature of scale, not
dropout/noise. Test = cup→head vector LENGTH (rotation/translation-INVARIANT), video vs mocap, per pid:
  P24 0.952±0.004  P16 0.949±0.017  P23 0.964±0.002  (= ~5% COMPRESSED, tiny std = systematic)
  most others 0.99-1.02 (fine). OVERALL 1.001 → scale error is PER-PARTICIPANT, NOT global.
The markerless 3D for P24/P16 is uniformly ~5% smaller = a CALIBRATION/triangulation depth-scale error
in THOSE participants' calib. Pure-rotation Kabsch can't absorb scale → shows as 20-30mm "residual" that
LOOKS like tracking failure but is metric-scale. Rotation is actually fine (~10° head-cup angle residual).
This is 9 of the 14 reps (P24×8, P16). WHY global umeyama FAILED earlier: applied scale to ALL reps incl
correct ones (rms went UP). FIX = PER-PARTICIPANT scale (~1.05x for P24/P16), not global, not rotation.
Ties to [[calibration_source]] (P24=9-cam cam4-junk) + [[project_calibrate_all_participants]]. So the
"hard-to-rescue" bucket is now: 6 wrist-swap (OMC bad) + ~9 per-pid-scale (P24/P16 calib) + a few genuine.

## PER-PARTICIPANT SCALE APPLIED + VERIFIED (2026-07-09) — head FIXED, cup-drink still open
Fit s_p per pid = median MMC/OMC chord-length ratio (fit-INDEPENDENT, invariant to R,t; can't be
faked by a bad fit). EXCLUDE the 6 wrist-swap reps from s_p estimation (their cup follows the wrist =
broken track, poisons the ratio — P13 s_p 1.042→0.982 once excluded). Correct by dividing the W0/video
side (cup + biomech head) by s_p about a shared centroid, THEN refit head+cup Kabsch (no exclude).
Scale-compressed pids: P24 0.947, P16 0.943, P23 0.963, P02 0.969 (|s_p-1|>3%); all others ~1.00 = no-op.
RESULTS (good-frame% <15mm, scale-compressed group P02/P16/P23/P24, 29 reps): 41.7%→76.3%; ALL improve,
none regress. Per-rep at rest the fix is decisive: P24 rest CUP resid 27.2mm→2.4mm (10x). HEAD across
ALL reps now median 5.1mm; scale-fixed group head 4.3mm (was ~20mm) = MATCHES the population → HEAD IS
FIXED. Scripts: scratchpad per_participant_scale.py (→cache/per_participant_scale.json), scale_x_restfit.py.
CAVEAT don't overclaim: (1) rms barely moves (dominated by drink-apex cup, scale-independent) — good%
is the honest metric here, median hides it. (2) Scale HELPS static/reach; for the drink apex it AMPLIFIES
the existing tilt/occlusion cup displacement (P24 drink cup 70→114mm WORSE) — cup-at-mouth is still the
open floor, separate from scale. (3) rest-only fit stacks on scale only for P24 (+11pts, long tilted
drink); P16/P23/P02 already ~80% with scale alone, rest-only redundant. So "hard 14" = 6 wrist-swap
(OMC bad) + P24/P16 scale (FIXED for head/static) + P15-type (good%=0 even after everything, a DIFFERENT
markerless failure scale doesn't touch). Ties [[calibration_source]] P24 9-cam cam4-junk.

## good% is BOTH channels pooled; SPLIT it + report BIAS vs VARIANCE (2026-07-09)
The good-frame% I quoted (76% scale-fixed) is HEAD+CUP residuals POOLED (concatenate([head_gap,
cup_gap]) < 15mm). Split from the drawn arrays (cache/draw_points): SCALE-FIXED group head 93% /
cup 59% / both 76%; ALL pids head 91% / cup 59% / both 75%. So "head fixed" is stronger than the
pooled number said (93%, near-100% for P02/P16/P23); cup 59% is the weak channel EVERYWHERE (drink
apex). Don't quote pooled good% as if it describes the head. BIAS/VAR decomposition (bias=|mean
residual vector|, var=scatter about it) — ALL pids: HEAD bias 4.0/var 6.1mm; CUP bias 13.9/var 32.6mm
→ the cup's problem is VARIANCE (frame scatter from occlusion), not a fixable offset; head is tight
on both. REPORT end-numbers as head/cup SEPARATELY and bias/var SEPARATELY, never one pooled scalar
([[feedback_good_frame_fraction_not_average]] [[feedback_dont_claim_definitive]]).

## P15-right is a ~20mm BIAS offset, NOT a failure — I misread a threshold artifact (2026-07-09)
CORRECTS an earlier wrong claim ("P15 = 0% good on both = broken markerless failure scale can't fix").
User: "Im seeing pretty good tracking on the P15 reps you rendered" — eye vs number, number was wrong.
Truth: P15 SPLITS by side. drinking_LEFT (6 reps, cls=clean, sync 0.985) = head 100%/cup ~68%/both 84%
= FINE. drinking_RIGHT (3 reps, cls=localized, sync 0.906-0.96) reads good@15mm=0% — but that is a HARD-
THRESHOLD ARTIFACT: head_med 20mm, cup_med 36mm sit just over the 15mm line. Relax threshold: P15-right
good@25mm=44%, @40mm=86%. The tracked cup (magenta) mean is ~30mm from OMC cup (yellow), bio head ~20mm
from OMC head — visually on-target, drawn markers land right. BIAS/VAR nails it: P15-right HEAD bias 20.2
/ var 3.8mm = almost PURE constant offset (rock-steady, rigidly shifted), signature of sync/lag not
tracking; P15-left head bias 2.4/var 4.4. A constant bias is CORRECTABLE (fix lag / subtract offset).
So P15-right = graceful ~20mm offset from cls=localized OMC pairing, NOT catastrophic, NOT the wrist-swap
bad-OMC bucket, NOT a markerless failure. LESSON (again, [[feedback_dont_claim_definitive]]
[[feedback_good_frame_fraction_not_average]]): a single hard-threshold good% turned "a bit worse (20mm)"
into "0% broken"; report good% across 15/25/40mm or the median+bias/var, and the eye was right.
The 3 rendered P15 reps are all drinking_RIGHT (the offset ones); render a drinking_LEFT to show clean P15.

## Render MUST equal the reported number — DUMB-PLAYER pattern (2026-07-09)
User caught render↔number inconsistency ("head is completely wrong" on a rep the analysis said 2.7mm).
ROOT CAUSE was NOT the fit: overlay.py paired the biomech-head and OMC-head at a crude integer frame
`mi=round((fr-lag)*ratio)`, a DIFFERENT correspondence than the analysis's resample+lag sync — inflated
the on-screen gap 2.7mm→18.7mm. Also overlay's fit was cup-ONLY (couldn't place the OMC head at all).
FIX (user chose "precompute in analysis"): emit_draw_points.py writes, PER VIDEO FRAME, the exact 3D
points to draw (OMC head/cup R,t-transformed, biomech head + tracked cup scaled — all one scaled-W0
space, ONE correspondence) → cache/draw_points/<video>.npz. overlay.py `--pscale` is a DUMB PLAYER:
load npz, project, draw — NO fit/sync/scale in the renderer. Verified drawn bio↔omc head gap = 2.7mm =
reported. This is [[feedback_shared_code_metric_and_render]] taken to its limit: the number and the
pixels are literally the same array, so they CANNOT diverge. save_headcup_fits.py stores R,t + the
scale_center used (render scales about the SAME center → bit-identical). Renders: drink_dwell/renders/
OVERLAY_*.mp4, each prints the head/cup/good% it draws. RULE: if a render can recompute anything the
number depends on, they WILL drift — precompute the drawables, make the renderer dumb.

## Coverage / re-run
523 reps skipped = biomech npz not computed YET ([[project_imove_fast_biomech_pipeline]] overnight
run mid-flight, ~156 npz when this ran). Re-run align_head_cup.py after the batch finishes to fill in.
