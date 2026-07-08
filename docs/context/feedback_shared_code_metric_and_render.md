---
name: feedback-shared-code-metric-and-render
description: metric/number code and the video/plot that visualizes it MUST call the same function — never reimplement the transform in the renderer
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 378d7b89-6fd6-44de-8fe5-5344c9335e39
---

When a NUMBER (a metric, a fit, an alignment) and a VISUALIZATION of that same thing (video
overlay, plot) are produced by SEPARATE code paths, they WILL drift and disagree — and the user
catches it by eye ("the yellow point is not on the cup, so the translation is wrong") while the
metric still reports a good value. This burned a whole session on drink_dwell: the overlay
reimplemented the mocap→W0 translation with a length-mismatched fallback that silently skipped the
sync, so the metric said 4mm but the rendered marker was nowhere near the cup.

**Why:** the render is the ground-truth sanity check on the number. If they use different code,
a bug in either one is invisible — the number looks fine, the render looks fine-ish, and you
"reconcile" two implementations forever instead of fixing one.

**How to apply:** factor the transform/fit into ONE function (e.g. `alignment_for(video, mode) ->
(R,t)` in session_align.py) and have BOTH the metric scoring AND the overlay/plot import and call
it. No duplicated math in the renderer. When the user asks for a graph AND a video of the same
quantity, wire them to the same source before rendering either. The user stated the rule directly:
"why don't you use the same functions for both the actual computations for getting the numbers and
the video generation — they shouldn't differ." See [[project_dwell_truth_failures]],
[[feedback_good_frame_fraction_not_average]].
