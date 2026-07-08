---
name: handoff-wire-session-R-into-pipeline
description: HANDOFF for another machine/instance — wire the session-R alignment into features.build_rep and re-run the drink_dwell LOPO; exact steps + gotchas
metadata:
  type: project
---

# HANDOFF: wire session-R into the drink_dwell feature pipeline + re-run LOPO

Written 2026-07-08 to be picked up on ANOTHER computer by a fresh Claude instance. Everything
needed travels in git. This is the pending payoff of a long alignment investigation (see the
result section below and [[project_dwell_truth_failures]]).

## What was established (don't re-derive — it's done)
The mocap→W0 alignment used to be a PER-TRIAL rigid Kabsch. On ~10% of reps (P16/P19/P23/P24)
the round-cup rotational symmetry let the per-trial fit pick the WRONG rotation branch. The fix:
the rotation is a SESSION CONSTANT (participant+date) — fit it robustly over all the session's
trials and freeze it, re-fit only translation per trial. This rescues the degenerate reps
(P16 24→62%, P19 90→94% all-moving good-frame agreement with the mocap cup GT). Ruled out and
archived (don't revisit): lag-retune, DTW time-warp, velocity-vector fit, exclude-threshold
sweep, scale/similarity. The residual disagreement is bad-MMC (occlusion) frames, not alignment:
gating to good-MMC frames (≥4 cams, ≤8px) gives 96% agreement.

## The one code change to make
`experiments/drink_dwell/features.py` → `head_distance()` currently fits the mocap→W0 transform
PER-TRIAL:
```python
    fit = mocap_to_w0(np.asarray(fit_cup if fit_cup is not None else cup_world, float),
                      tr.centroid(), tr.rate, r["lag"])
    R, t, _ = fit
```
Change it to use the SESSION rotation via the shared `alignment_for` (already written, tested):
```python
    from session_align import alignment_for      # lazy import (session_align imports features)
    al = alignment_for(video, "session", tr=tr)   # session R + per-trial t; falls back to
    if al is None:                                #   per-trial position fit if <3 session trials
        return None
    R, t, _ = al
```
Notes:
- `alignment_for(video, mode, npz=None, tr=None)` lives in `session_align.py`. Modes:
  position | session | velocity | scale | simrot. Use **"session"**.
- LAZY import inside the function — `session_align.py` imports `features`, so a top-level import
  in features.py is circular.
- `alignment_for` loads the npz itself if not passed; passing `tr=tr` avoids a reload. It uses the
  RAW consensus (`cons`) internally for the per-trial rotations and translation, matching the old
  `fit_cup=raw` intent — so you can drop the `fit_cup` plumbing for the fit (distance is still
  measured to `cup_world`/fused, unchanged).
- Keep everything downstream identical: `head_w0 = _mocap_to_track(tr.head_centroid() @ R.T + t, ...)`
  then distance to `cup_world`.
- OPTIONAL: add a `use_session=True` kwarg to `head_distance`/`build_rep` so you can A/B
  per-trial vs session in one run rather than hard-swapping.

## Then re-run the LOPO and compare
```bash
cd experiments/drink_dwell
conda activate <env>            # see [[conda_envs]]; on the dev box it was the repo .venv (py3.12)
python -u run.py                # LOPO over participants, writes cache/results.json
```
`run.py` trains base17 + proxy21 (METHODS at top). It's cache-first for features. The headline is
median |dwell duration error| in ms per method. BEFORE this change (per-trial fit) the numbers to
beat are ~proxy21 83-85ms / base17 116-126ms on ~655-666 reps (see [[project_drink_dwell_experiment]]).
Compare the new proxy21 (session-R truth) against that. EXPECT: the degenerate reps had corrupted
head-distance channels; session-R should improve proxy21 on P16/P19/P23/P24 folds specifically —
check per-participant, not just the median (the median may barely move since most reps were fine).

## Gotchas / rules (from memory — HOLD THESE)
- **No single number tells the story** — don't declare the change good/bad from the median alone;
  look at per-participant errors AND, if in doubt, render a rep. [[feedback_dont_claim_definitive]]
- **Metric and render must share code** — if you render to verify, use `overlay.py --session`
  (already calls the same `alignment_for`). Don't reimplement the transform. [[feedback_shared_code_metric_and_render]]
- Announce long runs + tail a log; never run >30s silently. Keep all caches. Work on a branch.
- Session grouping key = participant + recording DATE (`_sk` in session_align.py). P03 has 2 dates.
- `session_rotation`/`session_similarity` are cached per session in-process; a full run warms them.

## Files that matter
- `features.py` (edit here), `session_align.py` (alignment_for — the API), `agreement.py`
  (shared metric math), `overlay.py` (--session render), `run.py` (LOPO entry).
- Branch: `reorg/drink-study-lib`. All of the above is committed there.
