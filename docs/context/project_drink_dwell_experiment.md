---
name: project-drink-dwell-experiment
description: experiments/drink_dwell/ — standalone stage-by-stage rebuild of the proxy21 dwell pipeline; base17=proxy21 minus head-distance; entry run.py/summary.py/plot.py
metadata:
  type: project
---

`experiments/drink_dwell/` (created 2026-07-06, branch `reorg/drink-study-lib`) is the legible
rebuild of the proxy21 drink-dwell pipeline, split out of drink_study because the original
(`learn_seg_mouth.py` + `learn_seq_kf.build_seq`) was an 11-concern tangle no one could debug a
single stage of — the reason the cup→head "spike" took a whole session to localise.

**Structure (each file runs standalone on ONE trial as a smoke test):**
- `mocap.py` — cup+head C3D loader + `cup_to_head()` (THE signal, one function) + Kabsch/sync.
- `truth.py` — van-Andel dwell in 5 visible steps (smooth→apex→rest→thr=apex+15%(rest−apex)→longest run).
- `features.py` — proxy21 = `kinematics`(13) + `occlusion`(4) + `head_distance`(4), each a named fn.
- `model.py` — TCN + resample + span_from_prob + errs (frozen primitives, COPIED).
- `run.py` — LOPO train/score → `cache/results.json` (the entry point).
- `summary.py` → `slides/dwell_summary.png` (base17 vs proxy21: CDF + paired scatter + table).
- `plot.py` → `slides/worst_proxy21_grid.png` (worst reps).

**base17 vs proxy21:** SAME inputs (13 kinematics + 4 occlusion); proxy21 just ADDS the 4
head-distance channels (tracked cup → mocap head-centroid: dist, approach vel, norm, present).
base17 = video-only ceiling; the base17→proxy21 gap = what cup→head distance buys.

**Result (LOPO 666 reps):** proxy21 mean ~85ms / p50 67 / p90 158 beats base17 ~123 / 83 / 258.
(The drink_study docs' 77ms was the same pipeline; 77→85 is TCN run-to-run variance, no seed;
the old max-2417 outlier was training noise → ~1433 here.) Reproduces the SAME 666-rep set as
[[project_learned_segmenter]] / the old learn_seg_mouth.

**Copies the reachable slice (~600 lines), SHARES drink_study's DATA caches** (lopo_fused,
track3d_clean3d_refill, qtm_align.json, qtm_c3d_cleaned) — no duplicated caches, no GPU for
detection/tracking. Old `learn_seg_mouth.py`+`plot_worst.py` RETIRED to
`drink_study/archive/dwell_legacy/` (one live copy, no drift). Canonical grid name is
`worst_proxy21_grid.png` (was worst_trackedcup_grid). Stale suffixed caches → `cache/_archive/`.
Cost accepted: kinematics/head_distance primitives now duplicated between drink_dwell and lib/.
See [[project_repo_layout]].
