---
name: project-session-r-ab-confound
description: session-R head_distance A/B — RESULT: session-R HURTS unless guarded; only P12/P19 sessions are degenerate (ninl=0) and blow up. run.py-vs-committed-baseline is confounded — use same-process seeded use_session A/B.
metadata: 
  node_type: memory
  type: project
  originSessionId: 25d11f20-b722-49a2-b616-7d5262e468ad
---

Picked up the [[handoff-wire-session-R-into-pipeline]] task (branch `wire-session-R`, commit
18ada0e): wired session-R into `drink_dwell/features.head_distance` behind a `use_session=True`
kwarg (lazy import of `session_align.alignment_for` to dodge the circular import).

## RESULT (2026-07-08): session-R HURTS as written — do NOT merge without a degeneracy guard
Clean same-process, same-seed A/B (`use_session` False vs True, only head channels differ):
- proxy21 **per-trial**: mean **109ms**, p50 67, p90 183
- proxy21 **session**:  mean **231ms**, p50 83, p90 455  ← worse, fat tail
Per-participant: 12 worse / 7 better / 2 flat. The mean is dragged by TWO sessions only:
**P19 167→1867ms**, **P12 83→325ms**.

## Root cause — narrow, not a refutation of the idea
`session_align.session_rotation` is HEALTHY on 19/21 sessions (ninl≈ntr, self-dev 1–8°) and
rescues/tightens them. **P12 and P19 are DEGENERATE: ninl=0** (dev 25°/58°) — the per-trial
rotations are so scattered (round-cup symmetry, the exact thing this fix targets) that none fall
within the 20° inlier band, so it falls back to `R0` = the chordal mean of GARBAGE rotations,
worse than any single per-trial fit. Those two sessions blow up; that's the whole regression.
NOT side-related (P12/P19/P16/P23 all have both L+R). This is the OPPOSITE of the handoff's
"P19 90→94%" claim — on the locally-rebuilt lopo_fused cache P19's session R doesn't converge.

## THE FIX (designed, NOT applied — user said stop & report)
Guard `head_distance`: use the session R only when robust (`ninl>=3` and dev below a threshold),
else per-trial for that rep. `alignment_for(...,"session")` already returns `dict(ntr,ninl,dev)`.
Matches the handoff's own premise ("fit it ROBUSTLY, freeze it") — if it can't be fit robustly,
per-trial is correct. Expect: keeps the 7 wins, removes the P12/P19 blow-ups.

## Method note — why run.py-vs-committed-baseline was the WRONG test
The first comparison (fresh session-R `run.py` vs committed `cache/results.json` per-trial baseline)
was CONFOUNDED: **base17 regressed too** (116→195) though base17 never uses head_distance — proof
the shift is TCN training nondeterminism (run.py has no seed) + locally-rebuilt cache, not the
alignment. base17 is the tell. Judge session-R ONLY from the same-process seeded A/B.
[[feedback_dont_claim_definitive]] [[feedback_good_frame_fraction_not_average]]

## Artifacts / setup on THIS machine
- Cleaned C3Ds (head cluster) copied from Windows partition `Qualisys/Data/_drinking_all` →
  `drink_study/cache/qtm_c3d_cleaned/` (772 files, ALL have the head cluster). `ezc3d` installed
  into `object_tracking` env. `lopo_fused/` npz rebuilt on GPU (669 reps) — needed a ROOT-path fix
  in `archive/gapfill/lopo_fused.py` (reorg moved it deeper, doubled the cache path).
- A/B script: scratchpad `ab_session.py`; outputs `cache/ab_session.json` (untracked).
  Per-trial baseline preserved as `cache/results_pertrial_baseline.json` (committed `results.json`
  restored to the dev-box value).
