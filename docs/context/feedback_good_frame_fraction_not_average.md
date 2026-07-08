---
name: feedback-good-frame-fraction-not-average
description: "score fixes by the FRACTION OF GOOD FRAMES (frames crossing a usable threshold), not by the median/mean — the average hides fixes that move nothing usable"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 378d7b89-6fd6-44de-8fe5-5344c9335e39
---

The user repeatedly insists: judge an alignment/track fix by **how many frames become GOOD**
(cross a usable threshold — e.g. velocity-direction angle < 20°, or distance < 15mm), NOT by the
mean/median error. "the metric should be about the amount of good frames because that's what
we're trying to maximize."

**Why:** the median lies. In drink_dwell, a lag-retune showed +40° median-angle "improvement" on
several reps — but the good-frame fraction went 0% → 0%: it shuffled a pile of 130° frames down to
90° (moved the average) without pushing a SINGLE frame under the 20° usable bar. Optimizing the
average "improved" a number while making nothing actually usable. Only the good-frame count
exposed that the fix was empty.

**How to apply:** define a physical "good" threshold, report FRACTION of (drink-window / relevant)
frames below it, per rep. When testing a fix, also hold out (choose the parameter on one half of
frames, score good-frame% on the other) so an in-sample average gain can't fool you. Beware
scoring on a too-narrow window (a fit can be right everywhere but spike inside the dwell span where
the cup tilts) — widen to the cup-motion window if the narrow metric contradicts the visual.
See [[feedback_shared_code_metric_and_render]], [[project_dwell_truth_failures]].
