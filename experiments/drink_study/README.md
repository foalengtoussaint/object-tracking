# drink_study — setup on a new machine

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
(`_paths.py` reads it; `kf_accuracy.py`, `robustness.py`, `viz_replay.py`, `run.py` all use it.)

## 4. What runs WITHOUT videos
The cached analyses run straight from `cache/` (JSON detections, committed):
```bash
python experiments/drink_study/robustness.py        # usefulness/robustness model
python experiments/drink_study/plot_robustness.py   # the figure
python experiments/drink_study/kf_accuracy.py        # KF accuracy budget
```
Anything touching a *new* participant/trial or retraining needs the videos + the
`pscale_4` weights in `runs/`.
