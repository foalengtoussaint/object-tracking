"""Ablation: train a P01 student on cams 1-9 ONLY (drop cam_10 = the static
glass), identical otherwise to pscale_1 (P01, reps=2, 3000 frames). Eval
per-camera on held-out P06/P19/P23. If the held-out gain over pscale_1 mostly
comes back WITHOUT cam_10 in training, the glass was the cause; if held-out
cam_10 recall stays ~0, dropping it from training didn't teach the model that view.
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cv2
import numpy as np
import run
from metrics import train_with_metrics
from pipeline_lib import camera_id

CFG = "pscale_1_no10"
CAMS_NO10 = [c for c in run.ALL_CAMS if c != 10]
CACHE = Path("experiments/drink_study/cache/percam")
CONF = 0.25


def main():
    from ultralytics import YOLO
    # --- train on cams 1-9 ---
    clips = run.clip_files(["P01"], 2, CAMS_NO10, "right")
    pairs = []
    for clip in clips:
        pairs += run.label_clip_cached(None, clip, use_kf=True)  # all cached
    data_yaml, n = run.assemble_dataset(CFG, pairs)
    print(f"[{CFG}] pool={len(pairs)} -> training on {n} frames (cams 1-9)", flush=True)
    test_dir = run.STAGE / "percam_eval"   # same held-out clips as percam_recall
    run.stage_clips(run.TEST, 1, run.ALL_CAMS, "right", test_dir)
    best, _ = train_with_metrics(
        data_yaml, run.STUDENT, str(run.RUNS.resolve()), CFG, test_dir,
        long_run=False, max_epochs=20,
        config={"cfg_id": CFG, "train_participants": ["P01"], "reps": 2,
                "train_cameras": CAMS_NO10, "train_frames": n})

    # --- per-camera held-out recall (cached) ---
    CACHE.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(best))
    by_cam = {}
    for clip in sorted(test_dir.glob("*.mp4")):
        cam = camera_id(clip.stem)
        cf = CACHE / f"{CFG}__{clip.stem}.json"
        if cf.exists():
            pres = json.loads(cf.read_text())
        else:
            cap = cv2.VideoCapture(str(clip)); pres = []
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                r = model(f, conf=CONF, verbose=False)[0]
                pres.append(1 if (r.boxes is not None and len(r.boxes) > 0) else 0)
            cap.release(); cf.write_text(json.dumps(pres))
        d = by_cam.setdefault(cam, [0, 0]); d[0] += sum(pres); d[1] += len(pres)
    rec = {c: (d[0]/d[1] if d[1] else 0) for c, d in by_cam.items()}

    # --- compare to pscale_1 (with cam_10 in training) ---
    base = json.load(open(CACHE / "percam_recall.json"))["pscale_1"]
    cams = sorted(rec, key=lambda k: int(k.replace("cam", "")))
    print("\n=== per-camera held-out recall ===", flush=True)
    print("cfg              " + " ".join(f"{c:>6}" for c in cams), flush=True)
    print("pscale_1 (w/cam10)" + " ".join(f"{base.get(c,0):>6.2f}" for c in cams), flush=True)
    print("no_cam10 (1-9)   " + " ".join(f"{rec.get(c,0):>6.2f}" for c in cams), flush=True)
    print("delta            " + " ".join(f"{rec.get(c,0)-base.get(c,0):>+6.2f}" for c in cams), flush=True)
    mall = np.mean([rec[c] for c in cams]); ball = np.mean([base.get(c,0) for c in cams])
    print(f"\nmean held-out recall: pscale_1={ball:.3f}  no_cam10={mall:.3f}  delta={mall-ball:+.3f}", flush=True)
    print(f"held-out cam10 recall: pscale_1={base.get('cam10',0):.2f} -> no_cam10={rec.get('cam10',0):.2f}", flush=True)
    (CACHE / f"{CFG}_recall.json").write_text(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
