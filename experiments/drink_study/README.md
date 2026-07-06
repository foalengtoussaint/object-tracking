# drink_study

Cup-tracking + drink-dwell study: raw clips → per-cam detections → 3D cup track →
phase segmentation → drink-dwell (validated against Qualisys mocap).

## Layout (read `pipeline.py` first)

```
pipeline.py       ← START HERE. The whole DAG in one file; cache-first, no GPU to inspect.
                    `python pipeline.py --rep <stem>`  /  `--summary`
_paths.py         Single source of truth for paths (CACHE, ROOT, DS, CLIPS_ROOT).

lib/              The load-bearing spine — imported by everything, by BARE name.
                  segment_cup_only, qtm_align, qtm_c3d, learn_seg, learn_seq_kf,
                  kf_consensus, kf_accuracy, tune_interp, mouth_dwell, mouth_features,
                  cache_track3d(_consensus), gpu_decode, agreement, metrics, …
analysis/         Live probes + scorers (learn_seg_mouth, plot_worst, tune_seg, robustness,
                  fuse_phases, flag_trials, validate_mouth_vs_hybrid, overlay_markers, …).
cache_scripts/    Detection/track caching entry points (cache_dets*, run_clean3d*, calibrate_all).
viz/              Rerun/matplotlib viewers + deck builders (make_qtm_slides, build_*_nb, …).
render/           Video renderers (render_fused, render_tracking_video, …).
archive/          SETTLED / superseded threads, kept for provenance (NOT dead-deleted):
                  gapfill/ phaseseg/ segvariants/ prefix_pipeline/ agreement_iter/ early/
cache/            All committed analysis artifacts (detections JSON, 3D tracks, mocap, scores).
```

Scripts import spine modules by bare name (`import segment_cup_only`); a small shim at the
top of each script + `_paths.py` put `lib/` and the repo root on `sys.path`, so any script
runs from any directory. Paths are anchored in `_paths.py` (never recompute `parents[N]`).

## 1. Environment
```bash
conda env create -f ../../environment.yml      # creates env "object_tracking"
conda activate object_tracking
# or: pip install -r ../../requirements.txt
```

## 2. Data not in git (copy manually, e.g. from the SSD)
| What | Where it goes | Needed for |
|---|---|---|
| `clips/P*/P*_drinking_right_*.mp4` | any dir (see env var below) | re-inference, retraining, viz on new trials |
| `runs/` (model weights `.pt` + metrics) | `experiments/drink_study/runs/` | reusing trained students (e.g. pscale_4) |

On the SSD these are under `object_tracking_transfer/{clips,runs}/`.

## 3. Point scripts at the videos (no code edits)
Scripts default to `/home/imove/Documents/clips`. Override with one env var:
```bash
export OT_CLIPS_ROOT=/your/path/to/clips
```
(`_paths.py` reads it; `lib/kf_accuracy.py`, `analysis/robustness.py`, `viz/viz_replay.py`,
`cache_scripts/run.py` all use it.)

## 4. What runs WITHOUT videos
The cached analyses run straight from `cache/` (JSON detections, committed):
```bash
python experiments/drink_study/pipeline.py --summary          # DAG coverage across all reps
python experiments/drink_study/analysis/robustness.py         # usefulness/robustness model
python experiments/drink_study/analysis/plot_robustness.py    # the figure
python experiments/drink_study/lib/kf_accuracy.py             # KF accuracy budget
```
Anything touching a *new* participant/trial or retraining needs the videos + the
`pscale_4` weights in `runs/`.
