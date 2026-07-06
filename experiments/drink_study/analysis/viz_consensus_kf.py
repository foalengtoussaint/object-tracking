"""P23 consensus + INTERPOLATION video: shows that frames which DON'T triangulate
(<3-cam consensus) are filled by the 3D KF -> RTS estimate.

Per frame:
  - run gated consensus (>=3 cams). Frames with no consensus = the gaps.
  - run causal KalmanFilter3D fed every cam's detection (its gate rejects outliers)
    then RTS-smooth -> a position for EVERY frame (the interpolation).
  - reproject the RTS estimate into each camera as a filled circle:
      GREEN  = this frame HAS consensus (real triangulation)
      YELLOW = NO consensus this frame -> position is KF-INTERPOLATED (coasted)
  - top banner: CONSENSUS (n) green  /  NO CONSENSUS -> INTERPOLATED yellow.

So the yellow frames are exactly the 24% that don't triangulate, with the KF's
filled-in cup position drawn so you can see the track never drops.

    python experiments/drink_study/viz_consensus_kf.py --participant P23
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
from kalman_3d import load_calibration, project, KalmanFilter3D
import kf_accuracy as ka

REFILL = "experiments/drink_study/runs/pscale_1_clean3d_refill/weights/best_3df1.pt"
CONF = 0.25
TW, TH = 480, 270


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", default="P23")
    ap.add_argument("--weights", default=REFILL)
    ap.add_argument("--out", default="experiments/drink_study/cache/consensus_kf_P23.mp4")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(args.weights)
    from agreement import RES
    calib = load_calibration(f"data/calib/{args.participant}/calibration.toml", target_size=RES)
    stem = run.reps_of(args.participant, "right")[0]
    cams = [c for c in range(1, 11) if f"cam_{c}" in calib
            and (CLIPS_ROOT / args.participant / f"{stem}.{c}.mp4").exists()]
    caps = {c: cv2.VideoCapture(str(CLIPS_ROOT / args.participant / f"{stem}.{c}.mp4")) for c in cams}
    n = min(int(caps[c].get(cv2.CAP_PROP_FRAME_COUNT)) for c in cams)

    # --- pass 1: detect all cams all frames (cache boxes + centroids in memory) ---
    print(f"pass 1: detecting {n} frames x {len(cams)} cams ...", flush=True)
    boxes = {c: [None] * n for c in cams}     # (cx,cy) centroid or None
    imgs_cache = None                          # we re-read frames in pass 2
    for c in cams:
        cap = caps[c]
        for fr in range(n):
            ok, img = cap.read()
            if not ok:
                break
            r = model(img, conf=CONF, verbose=False)[0]
            if r.boxes is not None and len(r.boxes):
                b = max(r.boxes, key=lambda b: float(b.conf[0])).xyxy[0].tolist()
                boxes[c][fr] = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        print(f"  cam_{c} done", flush=True)

    dets = {f"cam_{c}": boxes[c] for c in cams}
    ck = {f"cam_{c}": c for c in cams}
    cam_keys = sorted(dets, key=lambda k: int(k.split("_")[1]))

    # --- KF -> RTS over the detections (same driver as viz_replay) ---
    ka.calib, ka.DETS, ka.CAMS, ka.N = calib, dets, cam_keys, n
    consensus = [ka.gated_consensus({c: dets[c][fr] for c in cam_keys if dets[c][fr] is not None})
                 for fr in range(n)]
    kf = KalmanFilter3D()
    xs_pred, Ps_pred, xs_upd, Ps_upd, Fs, has = [], [], [], [], [], []
    for fr in range(n):
        t = fr / ka.FPS
        obs = {c: np.array(dets[c][fr], float) for c in cam_keys if dets[c][fr] is not None}
        if not kf.initialized:
            if consensus[fr] is not None:
                kf.init(consensus[fr], t)
            for L in (xs_pred, Ps_pred, xs_upd, Ps_upd):
                L.append(kf.x.copy() if kf.initialized else None)
            Fs.append(np.eye(6)); has.append(kf.initialized); continue
        dt = t - kf.t_last
        kf.predict(t); F = np.eye(6); F[:3, 3:] = dt * np.eye(3)
        xs_pred.append(kf.x.copy()); Ps_pred.append(kf.P.copy()); Fs.append(F)
        for c in obs:
            kf.update(calib[c], obs[c], t)
        xs_upd.append(kf.x.copy()); Ps_upd.append(kf.P.copy()); has.append(True)
    xs_s = ka._rts_backward(xs_pred, Ps_pred, xs_upd, Ps_upd, Fs, has)
    rts = [x[:3].copy() if x is not None else None for x in xs_s]

    # --- pass 2: render, reproject RTS estimate into each cam ---
    cols = min(5, len(cams)); rows = (len(cams) + cols - 1) // cols
    BANNER = 50; gw, gh = cols * TW, rows * TH + BANNER
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (gw, gh))
    n_interp = 0
    print("pass 2: rendering ...", flush=True)
    for fr in range(n):
        has_con = consensus[fr] is not None
        if not has_con:
            n_interp += 1
        canvas = np.zeros((gh, gw, 3), np.uint8)
        col_banner = (0, 120, 0) if has_con else (0, 140, 200)   # green / amber
        txt = (f"CONSENSUS (real triangulation)   frame {fr}" if has_con
               else f"NO CONSENSUS -> position is KF-INTERPOLATED   frame {fr}")
        cv2.rectangle(canvas, (0, 0), (gw, BANNER), col_banner, -1)
        cv2.putText(canvas, txt, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        X = rts[fr]
        for i, c in enumerate(cams):
            ok, img = caps[c].read()
            if not ok:
                img = np.zeros((RES[1], RES[0], 3), np.uint8)
            # draw the detection box center (small) + the reprojected RTS estimate (big)
            d = dets[f"cam_{c}"][fr]
            if d is not None:
                cv2.circle(img, (int(d[0]), int(d[1])), 10, (255, 255, 255), 2)
            if X is not None:
                uv, infront = project(calib[f"cam_{c}"], X)
                if infront:
                    col = (0, 255, 0) if has_con else (0, 255, 255)   # green / yellow
                    cv2.circle(img, (int(uv[0]), int(uv[1])), 16, col, -1)
                    cv2.circle(img, (int(uv[0]), int(uv[1])), 16, (0, 0, 0), 2)
            tile = cv2.resize(img, (TW, TH))
            cv2.putText(tile, f"cam_{c}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            rr, cc = divmod(i, cols)
            canvas[BANNER + rr*TH:BANNER+(rr+1)*TH, cc*TW:(cc+1)*TW] = tile
        # legend
        cv2.putText(canvas, "white circle = detection  |  filled = KF/RTS cup (green=consensus, yellow=interpolated)",
                    (12, gh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        vw.write(canvas)
        if (fr + 1) % 100 == 0:
            print(f"  {fr+1}/{n}", flush=True)
    vw.release()
    for c in caps.values():
        c.release()
    print(f"interpolated (no-consensus) frames: {n_interp}/{n} ({100*n_interp/n:.0f}%)", flush=True)
    print(f"DONE wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
