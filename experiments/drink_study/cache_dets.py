"""Cache per-checkpoint cup detections on one P06 rep's 10 cameras, ONCE, so any
downstream agreement/offender analysis runs with zero re-inference.

For each epoch*.pt in --epochs, runs all cameras over all frames and writes the
top-confidence box center per frame to dets_cache/ep{N}.json:
    { "cam_1": [[cx,cy] or null, ...], ... }
Re-runnable: skips epochs already cached.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agreement import detect_rep, RES
from kalman_3d import load_calibration
from ultralytics import YOLO

CLIPS = Path("/home/imove/Documents/clips/P06")
STEM = "P06_drinking_right_20240123_105859"
WDIR = Path("experiments/drink_study/runs/baseline/weights")
OUT = Path("experiments/drink_study/debug_cam10/dets_cache")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, nargs="+",
                    default=[0, 2, 5, 8, 12, 18, 24, 29])
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    calib = load_calibration("data/calib/P06/calibration.toml", target_size=RES)
    rep = {f"cam_{c}": CLIPS / f"{STEM}.{c}.mp4" for c in range(1, 11)
           if f"cam_{c}" in calib and (CLIPS / f"{STEM}.{c}.mp4").exists()}
    print(f"rep cams: {sorted(rep)}", flush=True)

    for ep in args.epochs:
        out = OUT / f"ep{ep}.json"
        if out.exists():
            print(f"ep{ep}: cached", flush=True)
            continue
        w = WDIR / f"epoch{ep}.pt"
        if not w.exists():
            print(f"ep{ep}: no checkpoint {w}", flush=True)
            continue
        print(f"ep{ep}: detecting all cameras ...", flush=True)
        dets = detect_rep(YOLO(str(w)), rep, args.conf, classes=None, verbose=True)
        out.write_text(json.dumps(dets))
        print(f"ep{ep}: wrote {out}", flush=True)


if __name__ == "__main__":
    main()
