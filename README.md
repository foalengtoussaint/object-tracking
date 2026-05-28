# Object Tracking — Distillation Pipeline

Teacher → pseudo-label (Kalman-filtered) → fine-tune a small student → evaluate.

```
[teacher YOLO]  ──pseudo_label.py──▶  [dataset]  ──finetune.py──▶  [student]
                                                                       │
                                                                       ▼
                                                       evaluate.py (PR on train + test)
```

The Kalman filter in [kalman.py](kalman.py) drops the teacher's bad detections (anything trajectory-inconsistent) and the no-detection frames, so the dataset is just clean labels. The same filter is used at eval time to count what's a real localization error vs. an NMS duplicate.

## Layout

```
.
├── kalman.py / pseudo_label.py / finetune.py / evaluate.py    ← the 4 pipeline entrypoints
├── recording/        camera capture + Charuco calibration scripts (incl. teleimager/)
├── data/
│   ├── pretrained/         stock YOLO weights (yolo26n/s/m/l/x-seg)
│   ├── clips/{train,test}/ recorded mp4s
│   ├── datasets/{gen1,...}/  pseudo-labeled YOLO-seg datasets
│   ├── detections_cache/   per-frame JSON dets for evaluate
│   ├── runs/segment/<run>/ ultralytics fine-tune outputs (weights/best.pt etc.)
│   └── viz/{train,test}/   rendered mosaic mp4s for visual comparison
├── experiments/      markdown logs of significant runs (the recipe lives in git)
└── trash/            archived one-off scripts and old artifacts
```

## Pipeline (4 files)

| File | Role |
|---|---|
| [kalman.py](kalman.py) | `KalmanFilter2D` class + `filter_detections()` — shared by pseudo-label and evaluate |
| [pseudo_label.py](pseudo_label.py) | Run a teacher on a clip dir, KF-filter, write a YOLO single-class seg dataset |
| [finetune.py](finetune.py) | Train a YOLO-seg student on that dataset |
| [evaluate.py](evaluate.py) | Cache per-frame detections + compute PR proxies for any model/clip set |

## End-to-end example

Reproduce a gen2-style run: yolo26x labels `data/clips/train`, train a yolo26n student, evaluate on training + test sets.

```bash
# 1. Pseudo-label the training clips with a COCO teacher
python pseudo_label.py data/clips/train --out data/datasets/gen1 \
    --weights data/pretrained/yolo26x-seg.pt --conf 0.10

# 2. Fine-tune a small student on the resulting dataset
python finetune.py --data data/datasets/gen1/data.yaml --name cup_run --epochs 10

# 3. Evaluate on the training set (overfit check)
python evaluate.py --weights data/runs/segment/cup_run/weights/best.pt --clips data/clips/train

# 4. Evaluate on the held-out test set (real generalization)
python evaluate.py --weights data/runs/segment/cup_run/weights/best.pt --clips data/clips/test
```

### Self-distillation (use the student you just trained as the next teacher)

```bash
python pseudo_label.py data/clips/train --out data/datasets/gen2 \
    --weights data/runs/segment/cup_run/weights/best.pt \
    --conf 0.25 --classes 0

python finetune.py --data data/datasets/gen2/data.yaml --name cup_run_gen2 --epochs 10
python evaluate.py --weights data/runs/segment/cup_run_gen2/weights/best.pt --clips data/clips/test
```

`--classes 0` is needed because the student is single-class (only knows `my_cup`), so the COCO cup-like default won't match.

## Metrics

`evaluate.py` prints recall / P_loose / P_strict / F1 for both. Definitions and the loose-vs-strict rationale are in [experiments/2026-05-26_cup_5cam_demo.md](experiments/2026-05-26_cup_5cam_demo.md) under "Metric definitions" — short version: P_loose ignores NMS duplicates on the same object (the right metric for tracking), P_strict penalizes them too (useful for NMS-quality diagnostics).

## Recording your own clips

Camera capture is in [recording/](recording/). Quick path:

```bash
# 1. Edit recording/teleimager/cam_config_server.yaml so each camera's video_id
#    matches `v4l2-ctl --list-devices` on your machine
python recording/cam_server.py        # publishes 5 ZMQ streams on ports 55555..55559

# 2. Record a clip (one per camera, ~30 s)
python recording/record_clips.py      # writes clips/cam_N_<timestamp>.mp4
```

For Charuco calibration: `recording/record_calibration.py` then `recording/run_calibration.py` — only needed for 3D reconstruction, not the distillation pipeline.

## Setup

```bash
conda create -n object_tracking python=3.11 -y
conda activate object_tracking
pip install -r requirements.txt
```

## Tracking fine-tune runs

Datasets and weights are kept out of git (see `.gitignore`). Each significant run gets a markdown file in [experiments/](experiments/) using [experiments/TEMPLATE.md](experiments/TEMPLATE.md) — recipe in git, artifacts on disk.

## Archived

Earlier one-off scripts (live demos, the FastSAM bake-off, the Kalman bug-hunt diagnostic, the autonomous overnight runner, visual hull reconstruction, etc.) live in [trash/](trash/) for reference. They're not maintained and the pipeline doesn't depend on them.
