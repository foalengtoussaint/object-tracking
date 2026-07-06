---
name: project-dwell-truth-failures
description: drink_dwell worst reps are TRUTH failures not model — head-marker gaps/jumps + asymmetric cup→head reach (start close, end far) break the van-Andel dwell; filter both
metadata:
  type: project
---

The worst-error reps in `experiments/drink_dwell` (proxy21 vs mocap cup→head dwell) are
TRUTH-quality failures, not model failures. Confirmed by video overlay (P19) + measurement.
Two distinct mechanisms, BOTH to be filtered:

**1. HEAD-marker gaps / jumps (DONE — despike + exclude).** The rigid CUP is clean, but the
HEAD cluster occludes/drops out and QTM gap-fills with teleports (P19_0042: head-centroid
missing 172/978 frames = 18%, 308mm/fr jump). That spikes cup→head → corrupts the dwell.
Fix shipped: `mocap.head_centroid(despike=True)` — per-marker TEMPORAL outlier reject (NaN a
marker that jumps >30mm/fr from its OWN prev position, NOT vs the cluster — head markers are
legitimately far apart) + residual excursion excise. Plus `truth.head_quality` EXCLUDES reps
the despike can't save: head-missing >20% over the CUP-MOTION window OR residual jump >30mm/fr.
**GATE BUG CAUGHT BY USER:** first version measured head-missing over the DWELL span = circular
(dwell is computed from cup→head, which is NaN where head missing, so those frames can NEVER be
in the dwell → always 0%). Correct = measure over `mocap.cup_motion_window` (cup displacement
>40% of peak), head-free. That found P19 55% / P07 26-29% missing. 11 reps excluded, 655 kept.
Result: p99 533→408; base17 126→116; proxy21 mean ~84 unchanged.

**2. ASYMMETRIC cup→head reach (OPEN — filter next).** User spotted: on P14_R_092230 and
P11_L_145909 the cup is CLOSER to the head BEFORE the drink than AFTER. Measured: rest-before vs
rest-after cup→head = P14 349 vs 649 (+300mm), P11 227 vs 659 (+432mm); a NORMAL rep (P02) is
~symmetric (−16mm). The person starts with the cup near their face (on the table close to them)
and puts it down far away after. This breaks `truth._reach_rest`, which walks out from the apex
to 70%-of-max on BOTH sides and AVERAGES them — with a 300-430mm asymmetry the averaged rest is
skewed, the 15% threshold sits wrong, and the dwell lands wrong (P14 250ms→2400ms error).
The dwell definition ASSUMES a symmetric reach (rest→lift→drink→lower→rest). TODO: detect
asymmetric reach (|rest_before − rest_after| large, e.g. >150-200mm) and either filter the rep
or make `_reach_rest` use the higher/farther side (or per-side thresholds) instead of the mean.

**3. CUP gap-fill teleport (OPEN — despike blind spot).** Same class as (1) but on the CUP:
P07_142730 (c3d P070029) the mocap cup drops out for 3 frames at 285-288 and 296-299 and QTM
fills across with a ~440mm teleport — at the dwell's LEADING EDGE (dwell starts 302), so it
shifts the dwell onset. `centroid(despike=True)` did NOT remove it: the cup despike only excises
excursions bracketed by TWO jumps within ~1s; a single gap-crossing teleport slips through. FIX
needed: also NaN a single >Nmm/fr gap-crossing step (or reject cup markers whose gap-fill spans
>k frames). The MMC (video) cup stays continuous through these (overlay.py now draws it magenta),
so these reps' truth is mocap-only-wrong.

NOTE: despiking does NOT touch P14/P11 (their head is clean, 0 NaN) — those are pure asymmetry
cases, orthogonal to mechanism 1. Don't confuse the two. P14/P11 local video is NOT present
(only 092404/092441, 145743/145758 timestamps on disk; the exact bad-rep clips are on the
unmounted SSD) — diagnose them from mocap only. See [[project_drink_dwell_experiment]],
[[project_mouth_dwell_truth]].
