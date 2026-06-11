"""Analyze cached per-checkpoint detections (debug_cam10/dets_cache/ep*.json) with
ZERO re-inference. Answers:

  1. Per-epoch rep-median reproj error, ALL cams vs cam_4 dropped
     -> how much of the checkpoint-to-checkpoint variation cam_4 explains.
  2. Per-checkpoint per-camera offender table (each cam's own reproj error)
     -> is cam_4 (or another camera) a CONSISTENT problem, or only at spikes?

Reproj error = distance between a camera's detected box-center and the reprojection
of the all-cam triangulated 3D point (single-object cup).
"""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from kalman_3d import load_calibration, triangulate_dlt, project
from agreement import RES

CACHE = Path("experiments/drink_study/debug_cam10/dets_cache")


def load_ep(ep: int) -> dict:
    d = json.loads((CACHE / f"ep{ep}.json").read_text())
    return {c: [tuple(x) if x else None for x in v] for c, v in d.items()}


def per_frame_errors(dets, calib, drop=None):
    """Returns (rep_median, {cam: [errs]}) using all-cam triangulation."""
    cams_all = [c for c in dets if c != drop]
    n = min(len(dets[c]) for c in cams_all)
    rep_med = []
    percam = {c: [] for c in cams_all}
    for t in range(n):
        obs = {c: dets[c][t] for c in cams_all if dets[c][t] is not None}
        if len(obs) < 2:
            continue
        X = triangulate_dlt([calib[c] for c in obs], [np.array(obs[c]) for c in obs])
        errs = {}
        for c in obs:
            uv, infront = project(calib[c], X)
            errs[c] = float(np.hypot(*(uv - np.array(obs[c])))) if infront else 1e6
        rep_med.append(np.median(list(errs.values())))
        for c, e in errs.items():
            percam[c].append(e)
    return (float(np.median(rep_med)) if rep_med else float("nan")), percam


def main():
    calib = load_calibration("data/calib/P06/calibration.toml", target_size=RES)
    eps = sorted(int(p.stem[2:]) for p in CACHE.glob("ep*.json"))
    if not eps:
        sys.exit(f"no cached detections in {CACHE} yet")
    print(f"cached epochs: {eps}\n")

    # 1) cam_4's share of the variation
    print("=== rep-median reproj (px): all cams vs cam_4 dropped ===")
    print(f"{'ep':>3} {'all':>7} {'no_c4':>7} {'delta':>7}")
    allv, noc4 = [], []
    for ep in eps:
        dets = load_ep(ep)
        a, _ = per_frame_errors(dets, calib)
        b, _ = per_frame_errors(dets, calib, drop="cam_4")
        allv.append(a); noc4.append(b)
        print(f"{ep:>3} {a:>7.1f} {b:>7.1f} {a-b:>7.1f}")
    allv, noc4 = np.array(allv), np.array(noc4)
    print(f"\nstd across epochs: all={np.nanstd(allv):.2f}  no_cam4={np.nanstd(noc4):.2f}")
    print(f"range: all={np.nanmin(allv):.1f}-{np.nanmax(allv):.1f}  "
          f"no_cam4={np.nanmin(noc4):.1f}-{np.nanmax(noc4):.1f}")

    # 2) per-checkpoint per-camera offender table (each cam's median own error)
    cams = sorted((c for c in load_ep(eps[0])), key=lambda k: int(k.split("_")[1]))
    print("\n=== per-camera median reproj (px) per epoch  [worst per row in caps via *] ===")
    print("ep  " + " ".join(f"{c.split('_')[1]:>5}" for c in cams))
    for ep in eps:
        dets = load_ep(ep)
        _, percam = per_frame_errors(dets, calib)
        meds = {c: (np.median(percam[c]) if percam[c] else float("nan")) for c in cams}
        worst = max((c for c in cams if not np.isnan(meds[c])), key=lambda c: meds[c])
        cells = []
        for c in cams:
            v = meds[c]
            s = f"{v:>5.0f}" if not np.isnan(v) else "    -"
            cells.append(s + ("*" if c == worst else " "))
        print(f"{ep:>3} " + "".join(cells))
    print("\n(* = worst camera that epoch; cam numbers are the column headers)")


if __name__ == "__main__":
    main()
