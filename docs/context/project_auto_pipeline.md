---
name: project_auto_pipeline
description: pipeline.py — gated standalone CLI that auto-runs the full new-object distillation with surfaced decisions
metadata: 
  node_type: memory
  type: project
  originSessionId: bdb7162a-6c96-4391-8f59-078e4c60445f
---

Built (2026-06-03) `pipeline.py` + `pipeline_lib.py`: a **standalone gated CLI** that runs the whole new-object/new-environment tracking pipeline and surfaces every decision instead of requiring manual inspection.

Spine (both entry paths share it): round-1 labels → finetune #1 → dense KF self-label → finetune #2 → eval. The only difference is round-1 label source: **AUTO** (COCO/YOLO-World teacher) vs **SAM** (manual clicks via `sam_label_server.py`).

Gates auto-proceed (PASS), auto-default+log (FLAG_AUTO), or **halt only on true forks** (FLAG_FORK): (1) label source [stage 0 probe], (2) "labels look good — train?" via a rendered H.264 review video before EACH finetune, (3) patch blind cams vs proceed [stage 4, only if > `blind_cam_limit` cams get 0 labels].

Key design points:
- Reuses (not duplicates) `build_dataset` (pseudo_label.py, extracted this session), `train_student` (finetune.py, extracted), `filter_detections` (kalman.py), and `sam_label_server.py` (subprocess).
- `loss_plateau_train`: ultralytics `on_fit_epoch_end` callback stops on **val-loss** plateau (replaces the old bash watcher). `epoch_policy(n_images)` scales epochs to dataset size (small seed trains long ~200; big dense plateaus ~15-20). This matters: tiny seed sets need many epochs or the student detects nothing.
- Honest eval uses a **random-frame** val split (`make_val_split`, which also rewrites data.yaml `path` — needed when datasets are copied). KF precision proxies (recall/P_loose) are the real signal.
- `data/runs/pipeline/<run>/manifest.json` logs every decision; `experiments/<date>_<run>.md` auto-written.

Verified: smoke test (4-clip subset, all gates/forks fire) + real run on `cube_p01_all` (40 clips) → held-out **P02 eval recall 0.784 / P_loose 0.958** (vs 0.874 manual; gap = honest val split shrank seed 30→26 → fewer dense labels). CLI: `python pipeline.py --clips <dir> --object <name> [--label-source auto|sam|probe] [--seed-dataset <ds>] [--eval-clips <dir>] [--resume]`. Builds on [[project_cube_sam_distill]].
