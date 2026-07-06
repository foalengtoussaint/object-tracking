"""Truly-fair agreement comparison: for each frame, use only the cameras where
ALL models detected the cup, triangulate each model from that identical camera
subset, and compare reprojection error. Same frame + same cameras => the only
variable is each model's box-center localization. Detections are cached per
(rep, model) so refining the analysis doesn't re-run the slow teacher.
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agreement import load_calibration, load_rep, detect_rep, RES
from kalman_3d import triangulate_dlt, project

CUP = [39, 40, 41, 45, 75]
MODELS = [
    ("teacher", "data/pretrained/yolo26x-seg.pt", CUP),
    ("before", "data/pretrained/yolo26n-seg.pt", CUP),
    ("after", "experiments/drink_study/runs/baseline/weights/best.pt", None),
]
CACHE = Path("experiments/drink_study/cache")


def get_dets(name, weights, cls, rep, conf, rep_tag):
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / f"{rep_tag}__{name}__c{conf}.json"
    if cf.exists():
        print(f"  [{name}] cached", flush=True)
        return {c: [tuple(x) if x else None for x in v]
                for c, v in json.loads(cf.read_text()).items()}
    from ultralytics import YOLO
    print(f"  [{name}] detecting...", flush=True)
    dets = detect_rep(YOLO(weights), rep, conf, cls, verbose=True)
    cf.write_text(json.dumps(dets))
    return dets


def reproj_median(dets_model, calib, t, cams):
    pts = [np.array(dets_model[c][t]) for c in cams]
    cobjs = [calib[c] for c in cams]
    X = triangulate_dlt(cobjs, pts)
    errs = []
    for c in cams:
        uv, infront = project(calib[c], X)
        errs.append(float(np.hypot(*(uv - np.array(dets_model[c][t])))) if infront else 1e6)
    return float(np.median(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("--rep-dir", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    calib = load_calibration(args.calib, target_size=RES)
    rep = load_rep(args.rep_dir, calib)
    rep_tag = args.rep_dir.parent.name + "_" + args.rep_dir.name
    print(f"rep cams: {sorted(rep)}\n", flush=True)

    dets = {name: get_dets(name, w, cls, rep, args.conf, rep_tag)
            for name, w, cls in MODELS}
    n = min(len(v) for d in dets.values() for v in d.values())

    def fair_compare(subset: list[str]) -> None:
        rows = {nm: [] for nm in subset}
        cam_counts = []
        for t in range(n):
            seen = {nm: {c for c in dets[nm] if dets[nm][c][t] is not None} for nm in subset}
            shared = set.intersection(*seen.values())
            if len(shared) < 2:
                continue
            cams = sorted(shared)
            cam_counts.append(len(cams))
            for nm in subset:
                rows[nm].append(reproj_median(dets[nm], calib, t, cams))
        if not cam_counts:
            print(f"  no frames where {subset} share >=2 cameras", flush=True)
            return
        print(f"\n=== FAIR ({' vs '.join(subset)}): {len(cam_counts)} frames, "
              f"avg {np.mean(cam_counts):.1f} shared cams/frame "
              f"(same frame + same cameras) ===")
        print(f"{'model':<10}{'median_px':>10}{'mean_px':>9}")
        for nm in subset:
            print(f"{nm:<10}{np.median(rows[nm]):>10.2f}{np.mean(rows[nm]):>9.2f}", flush=True)

    fair_compare(["teacher", "before", "after"])
    fair_compare(["teacher", "after"])


if __name__ == "__main__":
    main()
