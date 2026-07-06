"""Consensus-status video: refill student on all cams of a held-out rep, with a
per-frame banner showing whether the >=3-camera 3D consensus is reached.

Per frame: detect cup in every camera (box centre = the 2D obs), run the gated
consensus (iteratively drop the worst-reprojecting cam until <=30px, require >=3).
  - green box  = camera IS in the consensus inlier set
  - orange box = camera detected but was EJECTED (disagrees)
  - red label  = camera fired nothing
Top banner: "CONSENSUS: N cams" (green) or "NO CONSENSUS (<3)" (red) -- the red
frames are exactly the ones the pipeline emits NO 3D position for.

    python experiments/drink_study/viz_consensus.py
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cv2
import numpy as np
import run
from _paths import CLIPS_ROOT
from kalman_3d import load_calibration, triangulate_dlt, project
from agreement import RES

REFILL = "experiments/drink_study/runs/pscale_1_clean3d_refill/weights/best_3df1.pt"
CONF = 0.25
THR = 30.0
MINC = 3
TW, TH = 480, 270


def gated(obs, calib):
    """Return (X, kept_set). Iteratively eject worst-reprojecting cam to <=THR, >=MINC."""
    cur = dict(obs)
    while len(cur) >= 2:
        X = triangulate_dlt([calib[c] for c in cur], [np.array(cur[c]) for c in cur])
        e = {c: float(np.hypot(*(project(calib[c], X)[0] - np.array(cur[c])))) for c in cur}
        w = max(e, key=e.get)
        if e[w] <= THR:
            break
        del cur[w]
    if len(cur) < MINC:
        return None, set()
    return triangulate_dlt([calib[c] for c in cur], [np.array(cur[c]) for c in cur]), set(cur)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", default="P06")
    ap.add_argument("--out", default="experiments/drink_study/cache/consensus_status.mp4")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(REFILL)
    calib = load_calibration(f"data/calib/{args.participant}/calibration.toml", target_size=RES)
    stem = run.reps_of(args.participant, "right")[0]
    cams = [c for c in range(1, 11) if f"cam_{c}" in calib
            and (CLIPS_ROOT / args.participant / f"{stem}.{c}.mp4").exists()]
    caps = {c: cv2.VideoCapture(str(CLIPS_ROOT / args.participant / f"{stem}.{c}.mp4")) for c in cams}
    n = min(int(caps[c].get(cv2.CAP_PROP_FRAME_COUNT)) for c in cams)

    cols = min(5, len(cams)); rows = (len(cams) + cols - 1) // cols
    BANNER = 50
    gw, gh = cols * TW, rows * TH + BANNER
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (gw, gh))
    print(f"rendering {n} frames, {len(cams)} cams -> {args.out}", flush=True)

    n_no = 0
    for fr in range(n):
        frames, obs = {}, {}
        for c in cams:
            ok, img = caps[c].read()
            frames[c] = img if ok else np.zeros((RES[1], RES[0], 3), np.uint8)
            r = model(frames[c], conf=CONF, verbose=False)[0]
            if r.boxes is not None and len(r.boxes):
                b = max(r.boxes, key=lambda b: float(b.conf[0]))   # top detection = the obs
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                obs[f"cam_{c}"] = ((x1 + x2) / 2, (y1 + y2) / 2)
                frames[c] = (frames[c], (int(x1), int(y1), int(x2), int(y2)))
            else:
                frames[c] = (frames[c], None)
        X, kept = gated({k: v for k, v in obs.items()}, calib)
        ncon = len(kept)
        if ncon < MINC:
            n_no += 1

        canvas = np.zeros((gh, gw, 3), np.uint8)
        # banner
        if ncon >= MINC:
            cv2.rectangle(canvas, (0, 0), (gw, BANNER), (0, 120, 0), -1)
            txt = f"CONSENSUS: {ncon} cams agree   (frame {fr})"
        else:
            cv2.rectangle(canvas, (0, 0), (gw, BANNER), (0, 0, 160), -1)
            txt = f"NO CONSENSUS (<3)  -- pipeline emits NO position   (frame {fr})"
        cv2.putText(canvas, txt, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        for i, c in enumerate(cams):
            img, box = frames[c]
            ck = f"cam_{c}"
            if box is not None:
                color = (0, 255, 0) if ck in kept else (0, 165, 255)   # green inlier / orange ejected
                cv2.rectangle(img, box[:2], box[2:], color, 5)
            tile = cv2.resize(img, (TW, TH))
            if box is None:
                lab, col = f"cam_{c}: none", (0, 0, 255)
            elif ck in kept:
                lab, col = f"cam_{c}: IN", (0, 255, 0)
            else:
                lab, col = f"cam_{c}: ejected", (0, 165, 255)
            cv2.putText(tile, lab, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            rr, cc = divmod(i, cols)
            canvas[BANNER + rr * TH:BANNER + (rr + 1) * TH, cc * TW:(cc + 1) * TW] = tile
        vw.write(canvas)
        if (fr + 1) % 100 == 0:
            print(f"  {fr+1}/{n}", flush=True)
    vw.release()
    for c in caps.values():
        c.release()
    print(f"frames with NO consensus: {n_no}/{n} ({100*n_no/n:.0f}%)", flush=True)
    print(f"DONE wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
