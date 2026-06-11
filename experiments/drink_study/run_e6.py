"""E6 camera-transfer matrix. Train a student on EACH single camera of P01
(reps 1-10, sampled to the standard 3000-frame budget), test each on ALL cameras
(held-out P01 reps 11-15, same person). Produces a 10x10 recall matrix
(train cam x test cam) -> which viewpoints transfer to which.

All training labels are cache-hit (sweep already labeled P01 r10 per camera);
eval runs the student on held-out clips. eval_gate already returns per-camera
recall, which is exactly one matrix row.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import run
from metrics import train_with_metrics
from pipeline_lib import eval_gate, camera_id

TRAIN_REPS = 10            # P01 reps 1-10 (cached on all cams)
TEST_REPS = (10, 15)      # held-out reps 11-15 (0-indexed slice)
CAMS = list(range(1, 11))
RUNS = run.RUNS
E6 = run.STUDY / "e6"


def stage_test_clips(out_dir: Path) -> int:
    """Held-out P01 reps 11-15, ALL cameras, as symlinks for eval_gate."""
    import os
    out_dir.mkdir(parents=True, exist_ok=True)
    stems = run.reps_of("P01", "right")[TEST_REPS[0]:TEST_REPS[1]]
    pdir = Path("/home/imove/Documents/clips/P01")
    n = 0
    for stem in stems:
        for cam in CAMS:
            src = pdir / f"{stem}.{cam}.mp4"
            if src.exists():
                dst = out_dir / f"{stem}.{cam}.mp4"
                if not dst.exists():
                    os.symlink(src.resolve(), dst)
                n += 1
    return n


def train_one_cam(cam: int, teacher, holdout: Path) -> Path:
    cfg_id = f"e6_train_cam{cam}"
    clips = run.clip_files(["P01"], TRAIN_REPS, [cam], "right")
    pairs = []
    for clip in clips:
        pairs += run.label_clip_cached(teacher, clip, use_kf=True)
    if len(pairs) < run.TRAIN_FRAMES:
        print(f"  [cam{cam}] WARNING only {len(pairs)} frames (<{run.TRAIN_FRAMES})", flush=True)
    data_yaml, n_used = run.assemble_dataset(cfg_id, pairs)
    print(f"[cam{cam}] pool={len(pairs)} -> training on {n_used} frames", flush=True)
    # holdout = the staged held-out test clips; train_with_metrics evals it
    # internally (aggregate, ignored here) — the per-cam matrix is built below.
    best, _ = train_with_metrics(
        data_yaml, run.STUDENT, str(RUNS.resolve()), cfg_id, holdout,
        long_run=False, max_epochs=20,
        config={"cfg_id": cfg_id, "train_cam": cam, "reps": TRAIN_REPS,
                "train_frames": n_used})
    return best


def main():
    from ultralytics import YOLO
    E6.mkdir(parents=True, exist_ok=True)
    test_dir = run.STAGE / "e6_test"
    n_test = stage_test_clips(test_dir)
    print(f"E6: staged {n_test} held-out test clips (P01 reps 11-15)", flush=True)

    teacher = None  # all cache-hit; loaded lazily only on a miss
    # Train 10 single-camera students.
    weights = {}
    for cam in CAMS:
        wpath = RUNS / f"e6_train_cam{cam}" / "weights" / "best.pt"
        if wpath.exists():
            print(f"[cam{cam}] already trained, reuse {wpath}", flush=True)
            weights[cam] = wpath
            continue
        if teacher is None:
            # only construct if a clip is actually uncached
            need = any(not (run.LABELCACHE / "kf" / c.stem / ".done").exists()
                       for c in run.clip_files(["P01"], TRAIN_REPS, [cam], "right"))
            if need:
                teacher = YOLO(run.COCO)
        weights[cam] = train_one_cam(cam, teacher, test_dir)

    # Eval each student per test camera -> matrix row.
    matrix = {}
    for cam in CAMS:
        r = eval_gate(str(weights[cam]), test_dir, conf=0.25)
        table = r.metrics.get("per_camera", {})
        row = {f"cam{tc}": table.get(f"cam{tc}", {}).get("recall")
               for tc in CAMS}
        matrix[f"cam{cam}"] = row
        print(f"[train cam{cam}] per-test-cam recall: "
              + " ".join(f"{tc}:{row.get(f'cam{tc}')}" for tc in CAMS), flush=True)

    (E6 / "transfer_matrix.json").write_text(json.dumps(matrix, indent=2))

    # Pretty print the 10x10
    print("\n=== E6 transfer matrix (rows=train cam, cols=test cam) recall ===")
    print("tr\\te " + "".join(f"{tc:>6}" for tc in CAMS))
    for cam in CAMS:
        row = matrix[f"cam{cam}"]
        cells = "".join(
            (f"{row.get(f'cam{tc}'):>6.2f}" if isinstance(row.get(f'cam{tc}'), (int, float))
             else f"{'-':>6}") for tc in CAMS)
        print(f"{cam:>5} {cells}")
    print(f"\nwrote {E6/'transfer_matrix.json'}")


if __name__ == "__main__":
    main()
