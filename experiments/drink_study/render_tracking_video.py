"""Render an annotated tracking video for one or more drink_study reps.

For each rep, picks the camera whose cup detections move the most (so we never
land on a static-FP camera), then overlays per frame:
  * RED dot   = raw 2D student detection on that camera (cache/student_dets)
  * GREEN dot = the 3D RTS cup track reprojected into that camera (cache/track3d)
  * a phase banner (cup-only segmentation) + time + #cams kept by triangulation

Where green and red diverge, or green vanishes / #kept drops, you can SEE the
tracking break that made the rep an outlier.

    python experiments/drink_study/render_tracking_video.py P10_P10_drinking_right_20240202_153258
    python experiments/drink_study/render_tracking_video.py --nodrink     # all no-drink reps
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import segment_cup_only as sc
import kf_accuracy as ka
from kalman_3d import load_calibration, project
import gpu_decode

CLIPS = Path(os.environ.get("OT_CLIPS_ROOT", "/home/imove/Documents/clips"))
DET = Path("experiments/drink_study/cache/student_dets")
TRACK = Path("experiments/drink_study/cache/track3d")
OUT = Path("experiments/drink_study/cache/track_videos")
PHASE_BGR = {"rest_pre": (200, 200, 200), "forward_transport": (232, 155, 76),
             "drinking": (76, 85, 232), "back_transport": (59, 162, 240),
             "rest_post": (153, 153, 153)}


def best_camera(p, stem, raw):
    scores = {}
    for c, v in raw.items():
        pts = np.array([x for x in v if x is not None], float)
        if len(pts) < 0.3 * len(v):
            continue
        scores[c] = float(np.hypot(*(pts.max(0) - pts.min(0))))
    return max(scores, key=scores.get)


def render(track_stem: str):
    p = track_stem.split("_")[0]
    clipstem = track_stem.replace("__pscale_4", "")[len(p) + 1:]   # P10_drinking_right_...
    d = json.loads((TRACK / f"{track_stem}.json").read_text())
    frames = d["frames"]
    rts = [f["rts"] for f in frames]
    kept = [len(f["kept"]) for f in frames]
    _, xyz = sc.load_track(track_stem)
    seg = sc.segment_cup_only(xyz)
    phase = seg["phase"]

    raw = json.loads((DET / f"{p}_{clipstem}__pscale_4__c{ka.CONF}.json").read_text())
    cam = best_camera(p, clipstem, raw)
    camnum = cam.split("_")[1]
    dets = raw[cam]
    calib = load_calibration(f"data/calib/{p}/calibration.toml", target_size=ka.RES)
    cc = calib.get(cam)

    video = CLIPS / p / f"{clipstem}.{camnum}.mp4"
    W, H, nv, _ = gpu_decode.dims(video)
    n = min(len(frames), nv, len(dets))
    OUT.mkdir(parents=True, exist_ok=True)
    sf = 0.5                                   # downscale for size
    ow, oh = int(W * sf), int(H * sf)
    outp = OUT / f"{track_stem}__cam{camnum}.mp4"
    vw = cv2.VideoWriter(str(outp), cv2.VideoWriter_fourcc(*"mp4v"), 30, (ow, oh))
    decoder = "GPU/NVDEC" if gpu_decode.gpu_available() else "CPU"

    for fr, img in enumerate(gpu_decode.frames(video)):
        if fr >= n:
            break
        # reprojected 3D RTS (green)
        if rts[fr] is not None and cc is not None:
            uv, infront = project(cc, np.array(rts[fr], float))
            if infront:
                cv2.circle(img, (int(uv[0]), int(uv[1])), 14, (0, 230, 0), 3)
        # raw 2D detection (red)
        if dets[fr] is not None:
            cv2.circle(img, (int(dets[fr][0]), int(dets[fr][1])), 9, (0, 0, 255), -1)
        # phase banner
        nm = sc.PHASE_NAMES[int(phase[fr])] if fr < len(phase) else "?"
        cv2.rectangle(img, (0, 0), (W, 70), PHASE_BGR[nm], -1)
        cv2.putText(img, f"{nm}", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        cv2.putText(img, f"t={fr/sc.FPS:5.1f}s  f{fr}  cams_kept={kept[fr]}  "
                    f"speed={seg['speed'][fr]:4.0f}mm/s  disp={seg['disp'][fr]:4.0f}mm",
                    (12, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, "red=2D det  green=3D track reproj", (W - 620, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        vw.write(cv2.resize(img, (ow, oh)))
    vw.release()
    print(f"wrote {outp}  (cam {camnum}, {n} frames, decode={decoder})", flush=True)
    return outp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reps", nargs="*")
    ap.add_argument("--nodrink", action="store_true", help="render all no-drink reps")
    args = ap.parse_args()
    if args.nodrink:
        allr = json.loads(Path("experiments/drink_study/cache/cup_only_all.json").read_text())
        reps = [r["rep"] for r in allr if not r["drink"]]
    else:
        reps = args.reps
    print(f"rendering {len(reps)} reps", flush=True)
    for i, r in enumerate(reps, 1):
        print(f"[{i}/{len(reps)}] {r}", flush=True)
        render(r)


if __name__ == "__main__":
    main()
