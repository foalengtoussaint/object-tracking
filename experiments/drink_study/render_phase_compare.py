"""Side-by-side phase-segmentation video: CUP-ONLY vs BIOMECH (pose) on one trial.

Renders the footage (best cup-view camera) with two phase timelines stacked on
top — cup-only (segment_cup_only on our 3D track) and biomech (the pose-pipeline
phase_intervals saved in the biomech npz) — plus a moving cursor and the current
phase of each method, with a marker when they disagree. Lets you watch where the
two segmentations differ (mainly the hand-only reaching / returning phases).

    python experiments/drink_study/render_phase_compare.py P06_P06_drinking_right_20240123_110054
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent))
import segment_cup_only as sc
import gpu_decode

CLIPS = Path(__import__("os").environ.get("OT_CLIPS_ROOT", "/home/imove/Documents/clips"))
DET = Path("experiments/drink_study/cache/student_dets_clean3d_refill")
TRACK = Path("experiments/drink_study/cache/track3d_clean3d_refill")
CACHE = Path("experiments/drink_study/cache")
OUT = Path("experiments/drink_study/cache/phase_compare")
FPS = 60.0
# unified phase colours (BGR)
PC = {"rest_pre": (190, 190, 190), "reaching": (200, 200, 70), "forward_transport": (232, 155, 76),
      "drinking": (76, 85, 232), "back_transport": (59, 162, 240), "returning": (190, 90, 190),
      "rest_post": (140, 140, 140)}


def best_cam(p, stem):
    raw = json.loads((DET / f"{p}_{stem}__clean3d_refill__c0.25.json").read_text())
    scores = {}
    for c, v in raw.items():
        pts = np.array([x for x in v if x is not None], float)
        if len(pts) >= 0.3 * len(v):
            scores[c] = float(np.hypot(*(pts.max(0) - pts.min(0))))
    return max(scores, key=scores.get)


def cup_phases(track_stem):
    d = json.loads((TRACK / f"{track_stem}__clean3d_refill.json").read_text())
    xyz = np.array([f["rts"] if f["rts"] else [np.nan] * 3 for f in d["frames"]], float)
    v = np.isfinite(xyz).all(1); idx = np.flatnonzero(v)
    for a in range(3):
        xyz[:, a] = np.interp(np.arange(len(xyz)), idx, xyz[idx, a])
    return sc.segment_cup_only(xyz)["intervals"], len(xyz)


def phase_at(intervals, fi):
    for n, s, e in intervals:
        if s <= fi < e:
            return n
    return "?"


def draw_strip(canvas, y, label, intervals, T, W, fi):
    h = 26
    cv2.putText(canvas, label, (4, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    x0 = 110
    bw = W - x0 - 10
    for n, s, e in intervals:
        a = x0 + int(s / T * bw); b = x0 + int(e / T * bw)
        cv2.rectangle(canvas, (a, y), (b, y + h), PC.get(n, (100, 100, 100)), -1)
    cx = x0 + int(fi / T * bw)
    cv2.line(canvas, (cx, y - 2), (cx, y + h + 2), (255, 255, 255), 2)


def render(trial):                       # trial = "P06_P06_drinking_right_..."
    p = trial.split("_")[0]
    stem = trial[len(p) + 1:]            # P06_drinking_right_...
    cup_iv, Tc = cup_phases(trial)
    b = np.load(CACHE / f"biomech_{trial}.npz", allow_pickle=True)
    bio_iv = [(n, int(s), int(e)) for n, s, e in b["phase_intervals"]]
    Tb = bio_iv[-1][2]
    T = min(Tc, Tb)
    cam = best_cam(p, stem); cn = cam.split("_")[1]
    video = CLIPS / p / f"{stem}.{cn}.mp4"
    W, H, nv, _ = gpu_decode.dims(video)
    n = min(T, nv)
    sf = 0.5; vw, vh = int(W * sf), int(H * sf)
    BAN = 96
    OUT.mkdir(parents=True, exist_ok=True)
    outp = OUT / f"{trial}__phasecmp.mp4"
    writer = cv2.VideoWriter(str(outp), cv2.VideoWriter_fourcc(*"mp4v"), 30, (vw, vh + BAN))
    GW = vw
    for fi, img in enumerate(gpu_decode.frames(video)):
        if fi >= n:
            break
        frame = cv2.resize(img, (vw, vh))
        canvas = np.zeros((vh + BAN, vw, 3), np.uint8)
        canvas[BAN:] = frame
        # banner with two strips
        cup_now = phase_at(cup_iv, fi); bio_now = phase_at(bio_iv, fi)
        draw_strip(canvas, 8, "CUP-ONLY", cup_iv, T, GW, fi)
        draw_strip(canvas, 44, "BIOMECH", bio_iv, T, GW, fi)
        # disagreement: map hand-only phases for the "real" comparison
        m = {"reaching": "rest_pre", "returning": "rest_post"}
        disagree = m.get(bio_now, bio_now) != cup_now
        tag = "DIFF" if disagree else "="
        col = (0, 0, 255) if disagree else (0, 220, 0)
        cv2.putText(canvas, f"t={fi/FPS:4.1f}s  cup:{cup_now}  bio:{bio_now}  [{tag}]",
                    (4, BAN - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        writer.write(canvas)
    writer.release()
    print(f"wrote {outp}  (cam {cn}, {n} frames)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trials", nargs="+")
    args = ap.parse_args()
    for t in args.trials:
        render(t)


if __name__ == "__main__":
    main()
