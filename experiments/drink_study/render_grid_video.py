"""Render a synchronized ALL-CAMERAS grid video for a rep.

Tiles every camera of a rep side by side, each overlaid with:
  RED dot   = raw 2D student detection on that camera
  GREEN dot = the 3D RTS track reprojected into that camera (calibrated cams only)
plus a global phase banner + time + #cams kept. Lets you see, e.g., what all 9
P24 cameras are simultaneously locking onto when the track diverges.

Cameras decoded via NVDEC (gpu_decode); one ffmpeg pipe per camera, read
frame-synchronized so memory stays bounded.

    python experiments/drink_study/render_grid_video.py P24_..._105712
    python experiments/drink_study/render_grid_video.py --flagged
"""
from __future__ import annotations
import argparse, json, math, os, sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import segment_cup_only as sc
import kf_accuracy as ka
import gpu_decode
from kalman_3d import load_calibration, project

CLIPS = Path(os.environ.get("OT_CLIPS_ROOT", "/home/imove/Documents/clips"))
DET = Path("experiments/drink_study/cache/student_dets")
TRACK = Path("experiments/drink_study/cache/track3d")
OUT = Path("experiments/drink_study/cache/grid_videos")
PHASE_BGR = {"rest_pre": (200, 200, 200), "forward_transport": (232, 155, 76),
             "drinking": (76, 85, 232), "back_transport": (59, 162, 240),
             "rest_post": (153, 153, 153)}
TW, TH = 480, 270            # per-camera tile size


def render(track_stem: str):
    p = track_stem.split("_")[0]
    clipstem = track_stem.replace("__pscale_4", "")[len(p) + 1:]
    d = json.loads((TRACK / f"{track_stem}.json").read_text())
    fr = d["frames"]
    rts = [f["rts"] for f in fr]
    kept = [len(f["kept"]) for f in fr]
    _, sxyz = sc.load_track(track_stem)
    seg = sc.segment_cup_only(sxyz)
    phase = seg["phase"]

    raw = json.loads((DET / f"{p}_{clipstem}__pscale_4__c{ka.CONF}.json").read_text())
    calib = load_calibration(f"data/calib/{p}/calibration.toml", target_size=ka.RES)

    # tile EVERY camera that has a clip (cams 1..10), not just the cached ones —
    # so under-cached participants (P10/P03: only 5-cam detections) still show all
    # their views; red/green overlays just appear only where dets/calib exist.
    gens, dims = {}, {}
    nframes = len(fr)
    for cn in range(1, 11):
        c = f"cam_{cn}"
        vp = CLIPS / p / f"{clipstem}.{cn}.mp4"
        if not vp.exists():
            continue
        w, h, n, _ = gpu_decode.dims(vp)
        dims[c] = (w, h)
        gens[c] = gpu_decode.frames(vp)
        nframes = min(nframes, n)
    cams = sorted(gens, key=lambda c: int(c.split("_")[1]))

    cols = math.ceil(math.sqrt(len(cams)))
    rows = math.ceil(len(cams) / cols)
    GW, GH = cols * TW, rows * TH + 40        # +40 banner
    OUT.mkdir(parents=True, exist_ok=True)
    outp = OUT / f"{track_stem}__grid.mp4"
    vw = cv2.VideoWriter(str(outp), cv2.VideoWriter_fourcc(*"mp4v"), 30, (GW, GH))
    print(f"{track_stem}: {len(cams)} cams, {nframes} frames -> {GW}x{GH}", flush=True)

    for fi in range(nframes):
        tiles = {}
        for c in cams:
            img = next(gens[c], None)
            if img is None:
                tiles[c] = np.zeros((TH, TW, 3), np.uint8); continue
            w, h = dims[c]
            sx, sy = TW / w, TH / h
            small = cv2.resize(img, (TW, TH))
            # reprojected 3D (green) — calibrated cams only
            if rts[fi] is not None and c in calib:
                uv, infront = project(calib[c], np.array(rts[fi], float))
                if infront:
                    cv2.circle(small, (int(uv[0] * sx), int(uv[1] * sy)), 8, (0, 230, 0), 2)
            # raw 2D detection (red) — only for cameras that were cached
            cam_dets = raw.get(c)
            dd = cam_dets[fi] if cam_dets and fi < len(cam_dets) else None
            if dd is not None:
                cv2.circle(small, (int(dd[0] * sx), int(dd[1] * sy)), 6, (0, 0, 255), -1)
            cv2.putText(small, c, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2)
            tiles[c] = small

        grid = np.zeros((GH, GW, 3), np.uint8)
        nm = sc.PHASE_NAMES[int(phase[fi])] if fi < len(phase) else "?"
        cv2.rectangle(grid, (0, 0), (GW, 40), PHASE_BGR[nm], -1)
        cv2.putText(grid, f"{nm}   t={fi/sc.FPS:5.1f}s  f{fi}  cams_kept={kept[fi]}   "
                    f"red=2D det  green=3D reproj",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        for k, c in enumerate(cams):
            r, cc = divmod(k, cols)
            grid[40 + r * TH:40 + (r + 1) * TH, cc * TW:(cc + 1) * TW] = tiles[c]
        vw.write(grid)
    vw.release()
    print(f"  wrote {outp}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reps", nargs="*")
    ap.add_argument("--flagged", action="store_true")
    args = ap.parse_args()
    if args.flagged:
        rows = json.loads(Path("experiments/drink_study/cache/flagged_trials.json").read_text())
        reps = [r["trial"] + "__pscale_4" for r in rows
                if r["updown"] or r["lateral"] or r["drink_fail"]]
    else:
        reps = [r if r.endswith("__pscale_4") else r + "__pscale_4" for r in args.reps]
    print(f"rendering {len(reps)} grid videos", flush=True)
    for r in reps:
        render(r)


if __name__ == "__main__":
    main()
