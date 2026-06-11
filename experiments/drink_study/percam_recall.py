"""Per-camera held-out recall for the pscale configs — to separate REAL
participant-generalization from the cam_10 glass-dilution confound.

Caches each (config, clip) -> per-frame detection-presence so re-runs / new
questions are instant. Recall per camera = fraction of frames with >=1 detection
(matches eval_gate's recall), computed on the held-out TEST participants.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cv2
import numpy as np
import run
from pipeline_lib import camera_id

CACHE = Path("experiments/drink_study/cache/percam")
CONF = 0.25
CONFIGS = ["pscale_1", "pscale_2", "pscale_4"]


def clip_presence(model, clip: Path) -> list[int]:
    cap = cv2.VideoCapture(str(clip))
    pres = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        r = model(f, conf=CONF, verbose=False)[0]
        pres.append(1 if (r.boxes is not None and len(r.boxes) > 0) else 0)
    cap.release()
    return pres


def cached_presence(cfg: str, clip: Path, model_getter) -> list[int]:
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / f"{cfg}__{clip.stem}.json"
    if cf.exists():
        return json.loads(cf.read_text())
    pres = clip_presence(model_getter(), clip)
    cf.write_text(json.dumps(pres))
    return pres


def main():
    from ultralytics import YOLO
    test_dir = run.STAGE / "percam_eval"
    run.stage_clips(run.TEST, 1, run.ALL_CAMS, "right", test_dir)
    clips = sorted(test_dir.glob("*.mp4"))
    print(f"eval clips: {len(clips)} (held-out P06/P19/P23)", flush=True)

    res = {}
    for cfg in CONFIGS:
        _m = {}
        def getter(cfg=cfg, _m=_m):
            if "m" not in _m:
                _m["m"] = YOLO(str(run.RUNS / cfg / "weights" / "best.pt"))
            return _m["m"]
        by_cam = {}
        for clip in clips:
            cam = camera_id(clip.stem)
            pres = cached_presence(cfg, clip, getter)
            d = by_cam.setdefault(cam, [0, 0])
            d[0] += sum(pres); d[1] += len(pres)
        res[cfg] = {c: (d[0] / d[1] if d[1] else 0) for c, d in by_cam.items()}
        print(f"{cfg}: cached/evaluated {len(clips)} clips", flush=True)

    cams = sorted({c for d in res.values() for c in d}, key=lambda k: int(k.replace("cam", "")))
    print("\n=== per-camera held-out recall ===", flush=True)
    print("cfg        " + " ".join(f"{c:>6}" for c in cams), flush=True)
    for cfg in CONFIGS:
        print(f"{cfg:<11}" + " ".join(f"{res[cfg].get(c,0):>6.2f}" for c in cams), flush=True)
    print("\nrecall gain pscale_1 -> pscale_4 per camera:", flush=True)
    gains = {c: res["pscale_4"].get(c, 0) - res["pscale_1"].get(c, 0) for c in cams}
    for c in cams:
        print(f"  {c}: {gains[c]:+.2f}", flush=True)
    g = np.array(list(gains.values()))
    print(f"\ngain spread: mean {g.mean():+.2f}, std {g.std():.2f}, "
          f"cam10 gain {gains.get('cam10',0):+.2f}, "
          f"cams>+0.05: {sum(v>0.05 for v in gains.values())}/{len(gains)}", flush=True)
    (CACHE / "percam_recall.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
