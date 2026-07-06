---
name: project-repo-layout
description: drink_study reorganized 2026-07-06 — pipeline.py entry point, lib/ spine, grouped leaves, archive/; paths anchored in _paths.py
metadata:
  type: project
---

`experiments/drink_study/` was reorganized 2026-07-06 (branch `reorg/drink-study-lib`) from a
flat 113-script directory into a legible structure. **Why:** the flat dir was unreadable — no
one could tell the ~15 load-bearing modules from ~95 one-shot probes, and there was no
end-to-end entry point.

**Structure:**
- `pipeline.py` — THE entry point. Whole DAG (clips→dets→3D track→segment→dwell) in one file,
  **cache-first** (reuses `student_dets*/`, `track3d*/`, `lopo_fused/`, `learn_seg_mouth.json`;
  computes only cold stages; never launches GPU just to inspect). `--rep <stem>` / `--summary`.
- `lib/` — the spine, imported everywhere by BARE name: `segment_cup_only qtm_align qtm_c3d
  learn_seg learn_seq_kf learn_correction kf_consensus kf_accuracy tune_interp mouth_dwell
  mouth_features cache_track3d cache_track3d_consensus gpu_decode agreement metrics
  render_phase_compare qtm_video_map`.
- `analysis/` live probes+scorers (learn_seg_mouth, plot_worst, tune_seg, robustness,
  fuse_phases, flag_trials, validate_mouth_vs_hybrid, overlay_markers, rescue_*, …).
- `cache_scripts/` (cache_dets*, run_clean3d*, calibrate_all) · `viz/` · `render/`.
- `archive/{gapfill,phaseseg,segvariants,prefix_pipeline,agreement_iter,early}/` — SETTLED /
  superseded threads, kept for provenance, NOT deleted. Archived files' cache paths were left
  broken-in-place (they aren't run); fix lazily if ever revived. See [[project_gapfill_variants_lopo]].

**Import mechanism (how bare imports survive the move):** each script has a small top-of-file
shim that walks up to the dir containing `lib/`, then inserts `lib/`, the drink_study root, and
the repo root onto `sys.path`. So `import segment_cup_only` (spine) and `from kalman_3d import …`
(parent repo) both resolve from any subdirectory. `_paths.py` (stays at drink_study root) also
adds `lib/` and now exports anchors: `CACHE`, `ROOT`, `DS` — **use these, never recompute
`Path(__file__).parents[N]`** (that arithmetic broke on the move and is the anti-pattern this
reorg removed).

**Verified after move:** all files byte-compile; spine imports resolve; leaves run as scripts;
`analysis/plot_worst.py` reproduces from cache ("all served … no training"); `pipeline.py
--summary` = 669 reps dets+track, 666 dwell-scored. No algorithm/data changes; no cache deleted.
