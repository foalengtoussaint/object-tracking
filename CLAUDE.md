# Project context for Claude

Persistent context for this repo lives in [docs/context/](docs/context/) — a
git-tracked snapshot of the working memory (project findings, the rig setup, and
the user's working preferences). **Read [docs/context/MEMORY.md](docs/context/MEMORY.md)
first** — it's a one-line index of every note; open the individual files for
detail.

To make these live memory on this machine, symlink them into your Claude memory
dir (optional):
```bash
ln -s "$(pwd)/docs/context"/*.md ~/.claude/projects/<this-project>/memory/
```

## Key working preferences (from docs/context/feedback_*.md)
- Never chain `kill … ; … ; launch …` in one shell call — split into separate calls.
- Announce what's running + ETA before long commands; never run >~30s silently
  (give a tailable log + incremental prints; pair background tasks with a monitor).
- **Keep ALL experiment data** — never auto-delete checkpoints/datasets/caches.
- **Always use the cache** — don't re-run GPU inference when a cached detection
  JSON exists (`experiments/drink_study/cache/`).
- Verify "stuck/slow" by measurement (etime/file-mtime/GPU), don't reassure from expectation.

## Where the work stands (drink_study)
See [docs/context/MEMORY.md](docs/context/MEMORY.md). Most recent thread:
- **Robustness/usefulness model** (`experiments/drink_study/robustness.py`): the
  filtered pipeline shrugs off 90% dropout / 50% corrupted cams / 15fps individually;
  the only hard floor is ≥3 geometrically-agreeing cameras.
- **Confident-wrong failures** (open): 213/228 failures are confident-but-wrong
  (median 1588mm), and `inlier_frac` as measured doesn't separate good/bad — likely
  a metric bug. Open tasks: fix the agreement metric, build a runtime confident-wrong
  guard, validate the predictor on real checkpoints. See
  [docs/context/project_failure_modes_confident_wrong.md](docs/context/project_failure_modes_confident_wrong.md).

## Repo layout (drink_study reorganized 2026-07-06)
`experiments/drink_study/` is no longer flat. Entry point is
[pipeline.py](experiments/drink_study/pipeline.py) — the whole DAG (clips→dets→3D→seg→dwell),
cache-first. Spine modules live in `lib/` (imported by BARE name via a per-script path shim +
`_paths.py`); leaves are grouped `analysis/ cache_scripts/ viz/ render/`; settled threads are in
`archive/` (kept, not deleted). Paths are anchored in `_paths.py` (`CACHE`, `ROOT`, `DS`) — do
NOT recompute `parents[N]`. See the README's Layout section and
[docs/context/project_repo_layout.md](docs/context/project_repo_layout.md).

## Setup on a new machine
See [experiments/drink_study/README.md](experiments/drink_study/README.md): conda
env from `environment.yml`, copy `clips/` + `runs/` (not in git), set
`export OT_CLIPS_ROOT=/path/to/clips`. Cached analyses run with no videos/GPU.
