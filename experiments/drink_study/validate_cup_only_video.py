"""Video-grounded sanity check for cup-only phase segmentation (no pose).

For one rep: run segment_cup_only on the cached 3D track, pick the camera with
the most cup detections, then pull one representative frame from the MIDDLE of
each phase, draw the cup 2D detection on it, and assemble a contact sheet with
the speed/disp + phase timeline on top. Lets us eyeball whether 'drinking' is
really cup-at-mouth, 'forward_transport' is the lift, etc.

    python experiments/drink_study/validate_cup_only_video.py P01 P01_drinking_right_20231220_141546
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import segment_cup_only as sc

CLIPS = Path(__import__("os").environ.get("OT_CLIPS_ROOT", "/home/imove/Documents/clips"))
DETCACHE = Path("experiments/drink_study/cache/student_dets")
CONF = "0.25"


def best_camera(detstem: str, participant: str) -> tuple[str, dict]:
    """cam with most non-null cup detections for this rep."""
    cf = DETCACHE / f"{participant}_{detstem}__pscale_4__c{CONF}.json"
    raw = json.loads(cf.read_text())
    counts = {c: sum(x is not None for x in v) for c, v in raw.items()}
    cam = max(counts, key=counts.get)
    return cam, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("participant")
    ap.add_argument("stem", help="rep stem, e.g. P01_drinking_right_20231220_141546")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    p, stem = args.participant, args.stem

    track_stem = f"{p}_{stem}__pscale_4"
    _, xyz = sc.load_track(track_stem)
    r = sc.segment_cup_only(xyz)
    T = len(xyz)

    cam, raw = best_camera(stem, p)
    camnum = cam.split("_")[1]
    dets = raw[cam]
    video = CLIPS / p / f"{stem}.{camnum}.mp4"
    print(f"rep {p}/{stem}  T={T}  best cam={cam} ({sum(x is not None for x in dets)} dets)  video={video.name}", flush=True)
    print("phases:", [(nm, round((e-s)/sc.FPS, 1)) for nm, s, e in r["intervals"]], flush=True)

    cap = cv2.VideoCapture(str(video))
    vW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # detections were computed on the same frames; scale det coords if video res
    # differs from the resolution dets were stored in (assume dets in 1280x720).
    DET_W, DET_H = 1280, 720
    sx, sy = vW / DET_W, vH / DET_H

    intervals = r["intervals"]
    fig, axes = plt.subplots(2, len(intervals), figsize=(3.0 * len(intervals), 6),
                             gridspec_kw={"height_ratios": [1, 2.2]})
    # top row: timeline spanning all columns -> use a single axis
    for ax in axes[0]:
        ax.remove()
    tl = fig.add_subplot(2, 1, 1)
    t = np.arange(T) / sc.FPS
    tl.plot(t, r["speed"], color="#333", lw=1.0)
    tl2 = tl.twinx(); tl2.plot(t, r["disp"], color="#7a4ce8", ls="--", lw=1.0)
    for nm, s, e in intervals:
        tl.axvspan(s / sc.FPS, e / sc.FPS, color=sc.PHASE_COLORS[nm], alpha=0.30, lw=0)
    tl.set_xlim(0, t[-1]); tl.set_ylabel("speed mm/s"); tl2.set_ylabel("disp mm")
    tl.set_title(f"{p}/{stem}  (cam {camnum})  — cup-only phases", fontsize=9)

    for k, (nm, s, e) in enumerate(intervals):
        fr = min((s + e) // 2, nv - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, img = cap.read()
        ax = axes[1][k]
        if ok:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            d = dets[fr] if fr < len(dets) else None
            if d is not None:
                cv2.circle(img, (int(d[0] * sx), int(d[1] * sy)), 12, (255, 0, 0), 3)
            ax.imshow(img)
            tl.axvline(fr / sc.FPS, color="k", lw=0.8, ls=":")
        ax.set_title(f"{nm}\nframe {fr} ({fr/sc.FPS:.1f}s)", fontsize=8,
                     color=sc.PHASE_COLORS[nm] if nm != "rest_pre" else "#555")
        ax.axis("off")
    cap.release()
    out = Path(args.out or f"experiments/drink_study/cache/cup_only_validate_{p}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
