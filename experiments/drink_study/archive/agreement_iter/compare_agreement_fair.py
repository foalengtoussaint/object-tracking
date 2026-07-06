"""Fair agreement comparison: reproj error on the SHARED frames where all models
triangulate, so precision isn't confounded by each model covering a different
(easier/harder) frame subset. Incremental log so `tail -f` is informative."""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agreement import (load_calibration, load_rep, detect_rep,
                       triangulate_per_frame, summarize, RES)

CUP = [39, 40, 41, 45, 75]
MODELS = [
    ("teacher (yolo26x COCO)", "data/pretrained/yolo26x-seg.pt", CUP),
    ("student BEFORE (yolo26n base)", "data/pretrained/yolo26n-seg.pt", CUP),
    ("student AFTER (fine-tuned)",
     "experiments/drink_study/runs/baseline/weights/best.pt", None),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("--rep-dir", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()
    from ultralytics import YOLO

    calib = load_calibration(args.calib, target_size=RES)
    rep = load_rep(args.rep_dir, calib)
    print(f"rep cams: {sorted(rep)}\n", flush=True)

    per_frame_by_model = {}
    for k, (name, w, cls) in enumerate(MODELS, 1):
        print(f"[{k}/{len(MODELS)}] {name}", flush=True)
        dets = detect_rep(YOLO(w), rep, args.conf, cls, verbose=True)
        per_frame_by_model[name] = triangulate_per_frame(dets, calib)
        print(f"  own-frames summary: {summarize(per_frame_by_model[name])}\n", flush=True)

    # frames triangulated by ALL models
    n = min(len(v) for v in per_frame_by_model.values())
    shared = [t for t in range(n)
              if all(per_frame_by_model[m][t] is not None for m in per_frame_by_model)]
    print(f"=== FAIR COMPARISON on {len(shared)} shared frames "
          f"(triangulated by all {len(MODELS)} models) ===")
    print(f"{'model':<32}{'median_px':>10}{'mean_px':>9}{'cams':>6}")
    for name in per_frame_by_model:
        errs = [per_frame_by_model[name][t]["median_err"] for t in shared]
        cams = [per_frame_by_model[name][t]["n_cams"] for t in shared]
        print(f"{name:<32}{np.median(errs):>10.2f}{np.mean(errs):>9.2f}"
              f"{np.mean(cams):>6.2f}", flush=True)


if __name__ == "__main__":
    main()
