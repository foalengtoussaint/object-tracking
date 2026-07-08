---
name: feedback-dont-claim-definitive
description: "don't say \"definitively/settled/conclusive\" from metrics alone — verify by render/measurement first; hedge until then"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 378d7b89-6fd6-44de-8fe5-5344c9335e39
---

Stop declaring things "definitive / conclusive / settled / this proves" on the strength of a
NUMBER when the claim hasn't been visually or independently verified. This session I did it
repeatedly and was wrong each time: "durations match → not temporal", "they don't correspond",
"translation is fine", "P16 is a spatial non-correspondent" — every one got overturned by the
next render or check. The user called it out directly: "stop saying definitively when you
actually have no idea."

**Why:** confident wrong conclusions send the work down dead ends and erode trust faster than
saying "I don't know yet." The metric is often measuring the wrong window / wrong quantity (see
[[feedback_good_frame_fraction_not_average]]), or a second code path disagrees with it (see
[[feedback_shared_code_metric_and_render]]).

**No single number tells you what's happening — there are only TWO ways to understand a rep
(the core point).** The user: "there's no single number that will tell you exactly what's
happening ... it's either watch the video or look at a big combination of numbers to understand."
So the two valid tools are: (a) WATCH THE VIDEO (render + actually view frames across the clip), or
(b) look at a BIG COMBINATION OF NUMBERS TOGETHER — the full panel viewed jointly: per-frame angle
timeline + speed + distance + which phase (reach/apex/lower) + scale + missing-frame count, side by
side. A single scalar (good-frame%, RMS, median angle, scale s) is a LOSSY projection that hides
where/when/why it fails — drink-window% hid that the reach tracked fine; RMS hid the apex blow-up;
a stable scale s=0.9 was real but invisible to the direction metric. NEVER conclude from one
number. Either watch the video, or build/read the multi-number panel.

**How to apply:** (1) When a number and the video/eye disagree, the number is suspect — go look,
don't rationalize the number. (2) Before saying a fix "works" or a hypothesis is "confirmed",
render it (and actually extract+view a frame) or cross-check with an independent metric. (3) Until
verified, hedge: "the metric suggests X, but I haven't confirmed it visually." (4) One still frame
does NOT prove behavior across a clip — don't over-read it either. (5) Prefer per-frame timelines +
video over a single aggregate; report what was actually checked, nothing more.
