# Drink-study methods — full explainer

A single reference for every method in the video cup-tracking pipeline and its
validation against QTM optical mocap, written to be *explained out loud*. Each
section: what the method does, why we chose it, and the verdict / caveat we found.

The one-line story: **markerless video tracks the cup to a few millimetres almost
everywhere; the only place it breaks is the drinking apex, where the cup at the mouth
occludes the cameras and the multi-view consensus becomes confidently wrong. From that
tracked cup we segment the drinking movement into phases, and the drink-dwell onset/offset
is where the current work lives.**

---

## 0. The data

- **Task**: the van Andel drinking protocol — reach, grasp, lift to mouth, drink, place
  back. Repeated many times per participant.
- **Video**: up to 10 calibrated cameras (Logitech BRIO), per-participant. Calibration is
  a Charuco-board intrinsics + extrinsics solve; a few participants have fewer usable cams.
- **Mocap ground truth (QTM)**: 4 reflective markers on the cup, 100/120 Hz, sub-millimetre
  6-DoF pose. 772 labelled C3D drinking trials. This is the *reference*, recorded
  simultaneously with the video.
- **Scale of the validation**: 669 valid drinking reps across 22 participants.

Two things to keep straight because they are easy to conflate:
- **The cup track** (a 3D position time-series) — the output of the tracking pipeline.
- **The phase segmentation** (rest / transport / drinking labels) — computed *from* a cup
  track (either the video track or the mocap track).

---

## 1. Detection — per-camera cup finding

**Method.** A YOLO detector run on every frame of every camera produces 2D cup boxes.
The production detector is a *distilled student*: we seeded it from ~30 SAM clicks on one
object, propagated dense pseudo-labels with a Kalman filter, and trained the student on
those. Participant-scaling ("pscale") augmentation improved cross-participant recall.

**Why.** Markerless, one-shot onboarding of a new object without hand-labelling thousands
of frames. Detections are cached per rep so the ~12h GPU job never re-runs.

**Verdict / caveats.**
- A **static side-desk glass** in `cam_10` is a persistent false positive the detector
  fires on; a 2D per-camera Kalman filter can't remove a *static* FP.
- Fix that matters downstream: **reject-then-fill** — drop the glass detections, then
  refill that camera from the multi-view consensus reprojected back into the image. This
  beat every alternative (cam_10 recall 0→0.74, mean recall 0.803). The tracks used for
  validation are the `clean3d_refill` variant.

---

## 2. 3D fusion — from 2D detections to one cup position

**Method.** Each frame's per-camera 2D detections are triangulated into a single 3D cup
point by **≥3-camera gated triangulation (consensus)**: only cameras that geometrically
agree contribute, and we require at least 3 of them. That consensus point is then fed to
a **Kalman filter + RTS smoother** to produce a clean 3D trajectory in millimetres.

**Why.** A single camera can't give depth; naive triangulation is hijacked by one bad
detection. Requiring geometric agreement across ≥3 views rejects the odd wrong detection,
and the KF/RTS turns a noisy per-frame point into a smooth track.

**Verdict / caveats (the core failure mode).**
- **Gating on a per-camera 2D KF is harmful.** A 2D gate can be fooled into locking onto a
  smoothly-moving *wrong* object (e.g. the hand/wrist marker, which in `cam_4` looks like
  the cup and alone drives ~90% of cross-checkpoint disagreement). The better design is a
  **consensus-anchored KF**: feed the 3D consensus point as the measurement to a *no-gate*
  KF + RTS. This fixed essentially all the 3D divergences (clean reps 346→354/355).
- **Accuracy budget**: the 3D KF tracks the cup faithfully up to ~±20 px of detection
  jitter and needs ≥3 well-spread cameras. Below that (σ≥40 px, or only 2 cameras) it fails
  catastrophically (>1 m error, typically grabbing the glass).
- **The apex is not a fillable gap.** At the mouth, occlusion is near-universal *and* the
  surviving cameras often agree on the *wrong* thing (hand/glass) — a confident-wrong
  consensus, not a data gap. No interpolator or learned prior fixes it (see §5).

---

## 3. Validation — scoring the video track against mocap

This is the accuracy story of the deck. We measure how close the video cup track lands to
the sub-mm mocap cup, per rep.

### 3a. Pairing rep ↔ mocap take
Mocap recorded more takes than the video log (extra/aborted captures), and drinking
speed-curves are near-identical, so **duration matching mis-registers**. Instead we pair
each video rep to its take by the **order-preserving assignment that minimises the 3D
Kabsch fit residual** (2 mm for the right pairing vs 30 mm for a wrong one — unambiguous).

### 3b. Temporal sync
Cross-correlate the two cup **speed curves**; the lag at peak correlation aligns them in
time. Reps below 0.7 correlation are rejected. A ground-truth quality gate also drops mocap
takes that are dead/static or have non-physical jumps, so a bad GT can never be charged as
tracking error.

### 3c. Error metric
After sync, fit a **robust (Huber-weighted) rigid Kabsch transform** so we measure
trajectory *shape*, not absolute placement. Then per frame: the mm distance between the
transformed video track and the mocap cup. Two summary numbers per rep:
- **inlier RMS** — fidelity on the agreeing frames (tracking accuracy).
- **frac_fail** — fraction of the trajectory >50 mm off (how much of it fails).

A rep is **clean / localized / broken** by how much of it exceeds the 50 mm threshold.

**Result.** Median agreement a few mm, p90 still small, consistent across all 22
participants; 0 broken reps after the pairing fix. Every rep tracks <15 mm on the good
frames — the class only reflects the brief occluded apex.

### 3d. Where and why it fails
- Failure is **one continuous arc**: ~0% in the rest/transport phases, rising to a peak
  **mid-drinking** (the apex), then decaying. It is confined to the drinking dwell.
- **Driver analysis** (per-frame logistic regression with cluster-bootstrap CIs, resampling
  reps not frames): **cup-near-mouth dominates** and is the least confounded factor;
  **hand-on-cup is a gate** (failure only when the hand is on the cup); **speed is
  non-monotonic** (slow fails at the mouth, fast fails in transport). The sign-flip survives
  in the *raw* (unsmoothed) track, so the interaction is real, not a KF artifact.

---

## 4. Phase segmentation — the geometric baseline

**Method (`segment_cup_only.py`).** Segment the drinking movement into phases **from the
cup 3D trajectory alone** (no body pose). Steps:
1. 6 Hz Butterworth low-pass (filtfilt, zero-phase) on the cup track.
2. Compute **speed** and **displacement-from-rest**.
3. Gate the **drinking dwell**: frames where `speed < DRINK_SPEED` *and* displacement is
   near its peak (`disp > peak − DRINK_DISP_PAD`), with hysteresis (`FWD_ON`/`BACK_OFF`) on
   the transport boundaries and a minimum phase length.
4. Phases: rest → forward-transport → **drinking** → back-transport → rest.

**Why.** The cup alone is enough — the drinking apex is exactly "cup lifted and briefly
still near its peak displacement". No pose estimator needed.

**Verdict.** 98% drink-detection. But the dwell *onset/offset* had a systematic bias.

### 4a. Tuning the gate (parameter search)
**Method (`tune_seg.py`).** Leave-one-participant-out grid search over the gate parameters
(`DRINK_SPEED`, `DRINK_DISP_PAD`, hysteresis, filter cutoff, min-phase). Truth =
`segment_cup_only(mocap_track)` at default parameters.

**Verdict.** The dwell under-estimate was a **segmenter-gate bias, not a track bug**.
Retuning `DRINK_SPEED/DISP` from 120/90 → **150/150** fixed it: bias −95 → +14 ms, mean
|dur| error 206 → 168, better on 20/21 folds, and it's an **interior optimum** (not a
grid-edge artifact — we widened the search to confirm). This is the shipped production gate.

---

## 5. Gap-fill — can we improve the apex track itself?

Before touching segmentation, we asked whether the ~20 mm apex error is a *fillable gap* in
the track.

**Methods tried (all on matched reps, LOPO):**
- KF re-tuning and occlusion-aware hold → plateau at the ~20 mm apex floor.
- A Gaussian process → broke the time-sync.
- A learned drink-shape prior → replaced good short-gap KF interpolation with a generic
  template, ~40% *worse* where it fired.
- **Velocity-fill TCN** — a temporal CNN that predicts the cup's **movement (velocity),
  not its position**, and integrates it across the gap. This one *worked*: −13% median /
  −33% p90 error at the apex vs KF coast.

**Verdict.**
- Predicting **movement, not position** is the key idea — the network models how the cup
  *continues* rather than guessing where it is.
- Regularisation and global context helped; **more layers, endpoint-loss, and attention
  were single-split noise** that evaporated on LOPO. (Recurring lesson: only LOPO wins are
  real here.)
- The residual apex error is **gaps-only** occlusion, not a modelling failure — the KF is
  excellent (~4.5 mm) on *visible* frames. Beware selection bias: always compare methods on
  the *same* matched reps.

---

## 6. Learned phase segmentation — the current work

The tuned gate (§4a) is a single scalar threshold, so it structurally can't adapt: some
reps want a *lower* gate (P20's slow place-down gets read as dwell), others want *higher*
(P10/P16 have a noisy in-dwell track). That leaves a heavy error **tail** (max 3617 ms) a
scalar cannot fix. So we replaced the threshold with a model that reads the local *shape*.

Everything here is **leave-one-participant-out** — the only trustworthy eval, because every
single-split "win" in this project evaporated under LOPO.

### 6a. The model: a per-frame classifier
A small **dilated TCN** (`ch=48`, 5 layers, dilations 1/2/4/8/16 → ~2 s receptive field)
outputs **P(drinking) per frame**. The dwell = the longest run where P > threshold, with the
threshold itself LOPO-tuned on the training folds.

We also tried two other heads and dropped both: a **boundary regressor** (regress onset/offset
directly — fragile, worst of all) and a **residual gate-corrector** (predict a per-rep delta
on the gate — netted back to baseline; it *looked* best at a short smoke run because the most
constrained model wins when everything is undertrained, then collapsed at full convergence).
**Never rank models from a short run.**

### 6b. What you feed it — the whole story
The classifier's quality is entirely about its input features. Progression (mean |dur| ms):

| input | ch | mean | what it added |
|---|---|---|---|
| filtered scalars (speed, disp, accel, gap, window) | 6 | 161 | beats the gate; fixes the P10 noisy-dwell tail |
| + raw speed + **3D velocity/disp vectors** (rep-local basis) | 13 | 162 | **solves P20 place-down** (3617→250) |
| raw multi-camera **detection** tensor (per-cam kept, ncams, occ, mpx) | 17 | 203 | worse alone, but occlusion **directly marks cup-at-mouth** |
| **HYBRID = 13 fused + 4 occlusion** | 17 | **162** | inherits both wins |

Key insights:
- **Speed should not be pre-filtered, and features should be 3D.** The gate's blurred scalar
  speed can't tell a genuine dwell (hover near mouth) from a slow place-down (moving *down/
  away*) — both have low speed *magnitude*. The **velocity direction** can. Adding raw
  (unfiltered) speed + 3D velocity/disp vectors in a sign-consistent rep-local basis is what
  solves P20's place-down (3617 → 250 ms).
- **Occlusion is a direct cup-at-mouth cue.** When the cup is at the mouth it occludes
  cameras, so `ncams`/`occ` spike exactly during the dwell. On its own the raw detection
  state is too noisy (worse than the gate), but *concatenated onto* reliable kinematics it
  sharpens the tail.

### 6c. The hybrid segmenter (current SOTA)
**Method (`learn_seg_hybrid.py`).** 13 fused-track kinematic channels + 4 occlusion channels
(`present, mpx, ncams, occ` from the detection tensor), same TCN, per-frame P(drink) head.

**Verdict — a genuine trade, not a clean win:**
- **Mean −14%** vs the tuned gate (162 vs 185), and the real win is the **tail**: p99
  1056 → ~700, max 3617 → ~2067. It keeps the P20 direction-solve *and* sharpens the P10
  occlusion tail harder than kinematics alone. Per-rep: better on 364, worse on 265.
- **Cost:** it creates ~26 *new* regressions >200 ms on reps the gate handled perfectly, by
  over-extending the dwell.
- **Ship decision is a judgment call:** hybrid if worst-case/mean matters; keep the tuned
  gate if breaking currently-working reps is unacceptable. The hybrid is **not yet wired into
  production** (`segment_cup_only.py` still ships the tuned gate).

### 6d. The critical caveat: the "truth" is itself imperfect
Our truth is `segment_cup_only(mocap_track)` — the geometric segmenter run on the *mocap*
cup. That is a **speed proxy**: it defines drinking as the cup being nearly *still*
(speed < gate). There is **no mouth/face marker** in the mocap set, so it clips to the
motionless *core* of the drink, not the true cup-at-mouth *envelope*.

**Video-verified finding:** rendering TRUE/tuned/hybrid dwell bands on the footage
(`render_dwell_compare.py`) shows that on several of the hybrid's "regressions" the cup is
**already at the mouth in drinking posture** in the disputed frames — the hybrid is *right*
and the speed-gate label is *late*. So (1) the hybrid's mean/tail wins are real and its
"regressions" are partly inflated by an imperfect label; (2) to *truly* validate dwell
onset/offset we'd need a mouth/face-marker distance (the van Andel 15%-of-steady-state gate)
or hand-labelled video onsets. This is the honest open frontier.

---

## Recurring methodology lessons (worth stating in the talk)
1. **LOPO or it isn't real.** Every single-split winner in this project (velocity-fill
   attention/depth, the residual gate) evaporated under leave-one-participant-out.
2. **Never compare models at unequal epochs.** The most-constrained model wins when
   everything is undertrained; flexibility wins at convergence. Smoke runs mislead.
3. **Compare on matched reps.** Selection bias (a method that only fires on easy reps) fakes
   improvement; always score on the same set.
4. **Ground truth can be wrong.** The mocap speed-gate is a proxy; a model beating it can be
   more correct than its own label. Validate the label before trusting the score.

---

## File map
- `segment_cup_only.py` — production geometric segmenter (tuned 150/150 gate).
- `tune_seg.py` — LOPO gate parameter search.
- `learn_seg.py` — learned segmenter harness (clf / boundary / residual heads).
- `learn_seg_clf.py`, `learn_seg_det.py`, `learn_seg_hybrid.py` — feature-set variants.
- `render_dwell_compare.py` — burns TRUE/tuned/hybrid dwell bands on footage.
- `make_qtm_slides.py` — the accuracy-validation figures + starter deck.
- Caches: `cache/lopo_fused/` (per-rep fused tracks + truth), `cache/learn_seg*.json`
  (per-rep predictions), `cache/qtm_align.json` (validation), `cache/dwell_compare/` (videos).
