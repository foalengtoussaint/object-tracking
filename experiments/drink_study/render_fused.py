"""Render the confidence-weighted cup+pose fusion onto the footage.

Top: best cup-view camera. Bottom: a live panel with the two distance-to-mouth
signals, the two per-frame confidences, the fused drink-evidence E (with the
on/off thresholds), and the reconciled phase bar -- all with a moving cursor, so
you can watch the handoff: cup confidence collapses at the occluded dwell while
pose carries it, and E rises above either source where they agree.

    python experiments/drink_study/render_fused.py P23_P23_drinking_right_20240716_151359
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import fuse_phases as F
import gpu_decode
import kf_accuracy as ka
from kalman_3d import load_calibration, project

CLIPS = Path(os.environ.get("OT_CLIPS_ROOT", "/home/imove/Documents/clips"))
DET = Path("experiments/drink_study/cache/student_dets_clean3d_refill")
OUT = Path("experiments/drink_study/cache/fused_video")
FPS = 60.0
PANEL_H = 360
PHASE_COL = {"rest_pre": (190, 190, 190), "forward_transport": (232, 155, 76),
             "drinking": (76, 85, 232), "back_transport": (59, 162, 240),
             "rest_post": (140, 140, 140)}


def best_cam(trial):
    ts = F.track_stem(trial)
    p = ts.split("_")[0]
    raw = json.loads((DET / f"{ts}__clean3d_refill__c0.25.json").read_text())
    scores = {}
    for c, v in raw.items():
        pts = np.array([x for x in v if x is not None], float)
        if len(pts) >= 0.3 * len(v):
            scores[c] = float(np.hypot(*(pts.max(0) - pts.min(0))))
    return max(scores, key=scores.get)


def line(panel, ys, x0, y0, w, h, color, thick=1):
    pts = [(x0 + int(i / (len(ys) - 1) * w), y0 + h - int(np.clip(v, 0, 1) * h))
           for i, v in enumerate(ys)]
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(panel, a, b, color, thick)


def render(trial):
    res = F.fuse(trial)
    T = res["T"]
    ts = F.track_stem(trial); p = ts.split("_")[0]; stem = ts[len(p) + 1:]
    cam = best_cam(trial); cn = cam.split("_")[1]
    video = CLIPS / p / f"{stem}.{cn}.mp4"
    W, H, nv, _ = gpu_decode.dims(video)
    n = min(T, nv)
    sf = 0.5; vw, vh = int(W * sf), int(H * sf)

    # calibration for reprojecting the tracked 3D points into this camera
    calib = load_calibration(f"data/calib/{p}/calibration.toml", target_size=ka.RES)
    camcal = calib[cam]
    cup3d, wrist3d, head3d = res["cup_xyz"], res["wrist_xyz"], res["head_xyz"]
    rx, ry = vw / ka.RES[0], vh / ka.RES[1]

    def reproj(X):
        if X is None or not np.isfinite(X).all():
            return None
        (u, v), ok = project(camcal, np.asarray(X, float))
        return (int(u * rx), int(v * ry)) if ok else None

    # normalise distances to [0,1] for the panel (cap 800mm)
    dcap = 800.0
    cupd = np.clip(res["cup_mouth"][:n] / dcap, 0, 1)
    wrid = np.clip(res["wrist_mouth"][:n] / dcap, 0, 1)
    wc = res["w_cup"][:n]; wp = res["w_pose"][:n]
    ec = res["e_cup"][:n]; ep = res["e_pose"][:n]; E = res["E"][:n]
    phases = res["reconciled"]

    OUT.mkdir(parents=True, exist_ok=True)
    outp = OUT / f"{trial}__fused.mp4"
    writer = cv2.VideoWriter(str(outp), cv2.VideoWriter_fourcc(*"mp4v"), 30, (vw, vh + PANEL_H))
    print(f"[{trial}] cam {cn}, {n} frames -> {outp}", flush=True)
    print(f"  FUSED drink {('%.2f-%.2fs'%(res['drink'][0]/FPS,res['drink'][1]/FPS)) if res['drink'] else 'NONE'}",
          flush=True)

    # static panel rows
    x0, w = 80, vw - 100
    rows = [("dist→mouth", vh + 16, 70), ("confidence", vh + 110, 70),
            ("fused E", vh + 204, 70)]
    phase_y = vh + 292

    for fi, img in enumerate(gpu_decode.frames(video)):
        if fi >= n:
            break
        frame = cv2.resize(img, (vw, vh))
        canvas = np.zeros((vh + PANEL_H, vw, 3), np.uint8)
        canvas[:vh] = frame
        cur_ph = next((nm for nm, s, e in phases if s <= fi < e), "?")

        # --- reprojected tracked points on the footage ---
        pc = reproj(cup3d[fi]) if fi < len(cup3d) else None
        pw = reproj(wrist3d[fi]) if fi < len(wrist3d) else None
        ph = reproj(head3d[fi]) if fi < len(head3d) else None
        # connecting lines (the two distance-to-mouth signals, drawn)
        if pc and ph:
            cv2.line(canvas, pc, ph, (255, 200, 120), 1)
        if pw and ph:
            cv2.line(canvas, pw, ph, (120, 120, 255), 1)
        if ph:   # head joint, used as mouth proxy = green
            cv2.circle(canvas, ph, 7, (0, 220, 0), 2)
            cv2.putText(canvas, "head (mouth proxy)", (ph[0] + 8, ph[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 0), 1)
        if pw:   # wrist = red
            cv2.circle(canvas, pw, 7, (60, 60, 230), 2)
            cv2.putText(canvas, "wrist", (pw[0] + 8, pw[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 230), 1)
        if pc:   # cup = blue, filled (it's the object)
            cv2.circle(canvas, pc, 8, (255, 150, 60), -1)
            cv2.putText(canvas, "cup", (pc[0] + 8, pc[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 60), 1)

        # header on the video
        cv2.putText(canvas, f"{trial}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(canvas, f"t={fi/FPS:4.2f}s  phase: {cur_ph}", (8, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, PHASE_COL.get(cur_ph, (255, 255, 255)), 2)

        # --- panel ---
        for lbl, y, h in rows:
            cv2.rectangle(canvas, (x0, y), (x0 + w, y + h), (40, 40, 40), 1)
            cv2.putText(canvas, lbl, (4, y + h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        # distances (inverted so 'near mouth' = high) — show raw close=high
        line(canvas, 1 - cupd, x0, rows[0][1], w, rows[0][2], (255, 150, 60), 2)   # cup blue
        line(canvas, 1 - wrid, x0, rows[0][1], w, rows[0][2], (60, 60, 230), 2)    # wrist red
        # confidences
        line(canvas, wc, x0, rows[1][1], w, rows[1][2], (255, 150, 60), 2)
        line(canvas, wp, x0, rows[1][1], w, rows[1][2], (60, 60, 230), 2)
        # evidence + fused E + thresholds
        line(canvas, ec, x0, rows[2][1], w, rows[2][2], (255, 150, 60), 1)
        line(canvas, ep, x0, rows[2][1], w, rows[2][2], (60, 60, 230), 1)
        line(canvas, E, x0, rows[2][1], w, rows[2][2], (255, 255, 255), 2)
        for thr in (F.E_ON, F.E_OFF):
            yy = rows[2][1] + rows[2][2] - int(thr * rows[2][2])
            cv2.line(canvas, (x0, yy), (x0 + w, yy), (90, 90, 90), 1)
        # phase bar
        for nm, s, e in phases:
            a = x0 + int(s / n * w); b = x0 + int(min(e, n) / n * w)
            cv2.rectangle(canvas, (a, phase_y), (b, phase_y + 24), PHASE_COL.get(nm, (100, 100, 100)), -1)
        # cursor across all panel rows
        cx = x0 + int(fi / n * w)
        cv2.line(canvas, (cx, vh + 10), (cx, phase_y + 26), (255, 255, 255), 1)
        # legend
        cv2.putText(canvas, "cup", (x0 + w - 120, vh + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 60), 1)
        cv2.putText(canvas, "wrist/pose", (x0 + w - 80, vh + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 230), 1)
        writer.write(canvas)
    writer.release()
    print(f"  wrote {outp}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trials", nargs="+")
    args = ap.parse_args()
    for t in args.trials:
        render(t)


if __name__ == "__main__":
    main()
