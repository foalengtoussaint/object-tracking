"""Re-verify the refill student's held-out detection-rate + 3D precision with the
FIXED (checkpoint-keyed) per_cam_eval. Verbose: prints per-clip progress so the
log is never empty mid-run."""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import cv2
import run
from agreement import agreement_eval
from pipeline_lib import camera_id
from ultralytics import YOLO

CONF = 0.25
W = "experiments/drink_study/runs/pscale_1_clean3d_refill/weights/best_3df1.pt"


def main():
    test_dir = run.STAGE / "percam_eval"
    run.stage_clips(run.TEST, 1, run.ALL_CAMS, "right", test_dir)
    clips = sorted(test_dir.glob("*.mp4"))
    print(f"REFILL re-verify: {len(clips)} held-out clips, fresh inference", flush=True)
    model = YOLO(W)
    by = {}
    for i, clip in enumerate(clips, 1):
        cam = camera_id(clip.stem)
        cap = cv2.VideoCapture(str(clip)); h = t = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            r = model(f, conf=CONF, verbose=False)[0]
            h += 1 if (r.boxes is not None and len(r.boxes)) else 0
            t += 1
        cap.release()
        d = by.setdefault(cam, [0, 0]); d[0] += h; d[1] += t
        print(f"  [{i}/{len(clips)}] {clip.stem}: {h}/{t} det", flush=True)
    rec = {c: d[0] / d[1] for c, d in by.items()}
    cams = sorted(rec, key=lambda k: int(k.replace("cam", "")))
    print("\nrunning gated 3D agreement ...", flush=True)
    a = agreement_eval(W, run.TEST, reps=1, hand="right", gated=True)
    print("\n=== REFILL re-verified (fixed eval) ===", flush=True)
    print(f"  det-rate mean = {np.mean([rec[c] for c in cams]):.3f}", flush=True)
    print(f"  cam10         = {rec.get('cam10', 0):.2f}", flush=True)
    print(f"  tri_rate      = {a.get('tri_rate'):.3f}", flush=True)
    print(f"  median_px     = {a.get('median_reproj_px'):.2f}", flush=True)
    print("  table reported: det-rate 0.803  cam10 0.74  tri_rate 0.917  px 2.99", flush=True)
    print("REVERIFY_DONE", flush=True)


if __name__ == "__main__":
    main()
