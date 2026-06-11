"""Compare inter-camera agreement across several models on one rep.

Prints incremental per-model / per-camera progress so a `tail -f` of the log is
informative. Usage:

    python experiments/drink_study/compare_agreement.py \\
        --calib data/calib/P06/calibration.toml \\
        --rep-dir data/clips/drink_study/baseline/eval
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agreement import load_calibration, load_rep, agreement_for_rep, RES

CUP = [39, 40, 41, 45, 75]   # COCO bottle/wine-glass/cup/bowl/vase

DEFAULT_MODELS = [
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
    print(f"rep cams: {sorted(rep)}  ({len(rep)} cams)\n", flush=True)

    results = []
    for k, (name, weights, cls) in enumerate(DEFAULT_MODELS, 1):
        print(f"[{k}/{len(DEFAULT_MODELS)}] {name}  ({weights})", flush=True)
        model = YOLO(weights)
        s = agreement_for_rep(model, rep, calib, conf=args.conf, classes=cls,
                              verbose=True)
        s["model"] = name
        results.append(s)
        print(f"  -> tri_rate={s['tri_rate']} cams={s['mean_cams_agreeing']} "
              f"median_px={s['median_reproj_px']} inlier={s['mean_inlier_frac']}\n",
              flush=True)

    print("\n=== SUMMARY ===")
    hdr = f"{'model':<32}{'tri_rate':>9}{'cams':>6}{'med_px':>8}{'mean_px':>9}{'inlier':>8}"
    print(hdr)
    for s in results:
        print(f"{s['model']:<32}{s['tri_rate']:>9}{s['mean_cams_agreeing']:>6}"
              f"{s['median_reproj_px']:>8}{s['mean_reproj_px']:>9}{s['mean_inlier_frac']:>8}")


if __name__ == "__main__":
    main()
