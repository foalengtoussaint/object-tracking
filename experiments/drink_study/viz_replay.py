"""Offline Rerun replay of the full cup-tracking pipeline on a cached trial.

Takes ONE recorded drinking rep with good detections, replays the exact
gating -> KF -> RTS pipeline we validated in `kf_accuracy.py`, and logs
everything to Rerun so you can scrub the 3D track against all 10 camera views.

No GPU, no re-inference: student detections come straight from the cache
(`cache/student_dets/<P>_<stem>__pscale_4__c0.25.json`). Calibration from
`data/calib/<P>/calibration.toml`. Camera frames are read on demand from
`/home/imove/Documents/clips/<P>/<stem>.<i>.mp4`.

Pipeline per frame (identical to the analysis script)
-----------------------------------------------------
  1. GATE   : `gated_consensus` -- iteratively drop the worst-reprojecting
              camera until <=30px, require >=3 agreeing cams -> the clean
              geometric "truth" point (logged white).
  2. KF     : causal `KalmanFilter3D` fed every camera's raw 2D detection;
              its EKF Mahalanobis gate rejects the glass/bracelet. The
              per-camera accept/reject flag colours the 2D overlay
              (green = used, red = gated out). -> ONLINE trail (red).
  3. RTS    : Rauch-Tung-Striebel backward pass over the stored forward
              states -> OFFLINE smoothed trail (green), future-aware.

Entities logged (timelines: "frame" index + "time" seconds)
------------------------------------------------------------
  world/cam_<i>                      camera pose (static)
  world/cam_<i>/image                downscaled JPEG frame
  world/cam_<i>/image/det            student detection (green=used / red=gated)
  world/consensus                    per-frame gated-consensus point (white)
  world/cup_kf / world/cup_kf/trail  causal KF estimate + trail (red)
  world/cup_rts / world/cup_rts/trail  RTS-smoothed estimate + trail (green)

Run
---
    python experiments/drink_study/viz_replay.py                 # default P01 rep, spawn viewer
    python experiments/drink_study/viz_replay.py --participant P06
    python experiments/drink_study/viz_replay.py --no-frames     # skip video (3D + dets only, fast)
    python experiments/drink_study/viz_replay.py --save out.rrd  # write a recording instead of spawning
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rerun as rr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))   # for kf_accuracy helpers
sys.path.insert(0, str(ROOT))                               # for kalman_3d / viz_rerun

from kalman_3d import KalmanFilter3D, load_calibration, project
import kf_accuracy as ka           # gated_consensus, _rts_backward, FPS, THR, CONF
import viz_rerun as vr             # camera/frame/3D logging helpers (reused)

# Default trial: P01's cleanest drinking rep (100% reference coverage in the sweep).
DEFAULT_P = "P01"
DEFAULT_STEM = "P01_drinking_right_20231220_141546"
CLIPS = Path("/home/imove/Documents/clips")
RES = (1920, 1080)


def load_dets(p: str, stem: str) -> dict:
    """Cached pscale_4 student detections for the rep (no GPU)."""
    cf = (Path(__file__).resolve().parent / "cache" / "student_dets"
          / f"{p}_{stem}__pscale_4__c{ka.CONF}.json")
    if not cf.exists():
        raise SystemExit(f"no cached student dets at {cf}\n"
                         f"run kf_accuracy.py first to populate the cache.")
    d = json.loads(cf.read_text())
    return {c: [tuple(x) if x else None for x in v] for c, v in d.items()}


def run_pipeline(calib, dets, cams, n):
    """Replay gating -> causal KF (with per-cam accept flags) -> RTS.

    Returns (consensus, kf_est, kf_accept, rts_est) where
      consensus[fr]  : np.ndarray|None  -- gated >=3-cam point
      kf_est[fr]     : np.ndarray|None  -- causal KF position (mm)
      kf_accept[fr]  : {cam: bool}      -- did the KF gate accept this cam's det
      rts_est[fr]    : np.ndarray|None  -- RTS-smoothed position (mm)
    """
    # point ka's module globals at this rep so gated_consensus/_rts_backward work
    ka.calib, ka.DETS, ka.CAMS, ka.N = calib, dets, cams, n

    consensus = [ka.gated_consensus({c: dets[c][fr] for c in cams
                                     if dets[c][fr] is not None})
                 for fr in range(n)]

    kf = KalmanFilter3D()
    kf_est, kf_accept = [], []
    xs_pred, Ps_pred, xs_upd, Ps_upd, Fs, has = [], [], [], [], [], []
    for fr in range(n):
        t = fr / ka.FPS
        obs = {c: np.array(dets[c][fr], float) for c in cams
               if dets[c][fr] is not None}
        if not kf.initialized:
            if consensus[fr] is not None:
                kf.init(consensus[fr], t)
            kf_est.append(kf.x[:3].copy() if kf.initialized else None)
            kf_accept.append({})
            xs_pred.append(kf.x.copy() if kf.initialized else None)
            Ps_pred.append(kf.P.copy() if kf.initialized else None)
            xs_upd.append(kf.x.copy() if kf.initialized else None)
            Ps_upd.append(kf.P.copy() if kf.initialized else None)
            Fs.append(np.eye(6)); has.append(kf.initialized)
            continue
        dt = t - kf.t_last
        kf.predict(t)                       # frame-level predict (sets t_last=t)
        F = np.eye(6); F[:3, 3:] = dt * np.eye(3)
        xs_pred.append(kf.x.copy()); Ps_pred.append(kf.P.copy()); Fs.append(F)
        acc = {}
        for c in obs:                       # each update's internal predict is a no-op (dt=0)
            ok, _ = kf.update(calib[c], obs[c], t)
            acc[c] = ok
        xs_upd.append(kf.x.copy()); Ps_upd.append(kf.P.copy()); has.append(True)
        kf_est.append(kf.x[:3].copy())
        kf_accept.append(acc)

    xs_s = ka._rts_backward(xs_pred, Ps_pred, xs_upd, Ps_upd, Fs, has)
    rts_est = [x[:3].copy() if x is not None else None for x in xs_s]
    return consensus, kf_est, kf_accept, rts_est


def log_point(entity, X_mm, color, radius, trail, trail_color):
    X_m = X_mm / vr.MM_PER_M
    rr.log(entity, rr.Points3D(X_m.reshape(1, 3), colors=[color], radii=[radius]))
    trail.append(X_m.copy())
    if len(trail) > vr.TRAIL_MAX:
        del trail[: len(trail) - vr.TRAIL_MAX]
    if len(trail) >= 2:
        rr.log(f"{entity}/trail",
               rr.LineStrips3D([np.asarray(trail)], colors=[trail_color], radii=[0.003]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participant", default=DEFAULT_P)
    ap.add_argument("--stem", default=None,
                    help="rep stem; default = participant's drinking_right rep")
    ap.add_argument("--no-frames", action="store_true",
                    help="skip logging camera video (faster, 3D + dets only)")
    ap.add_argument("--save", default=None, help="write .rrd instead of serving a viewer")
    ap.add_argument("--spawn", action="store_true",
                    help="spawn the native Rerun viewer (needs the `rerun` binary on PATH); "
                         "default serves a browser viewer like viz_rerun.py")
    ap.add_argument("--web-port", type=int, default=9090)
    ap.add_argument("--grpc-port", type=int, default=9876)
    args = ap.parse_args()

    p = args.participant
    stem = args.stem or (DEFAULT_STEM if p == DEFAULT_P
                         else f"{p}_drinking_right")
    if args.stem is None and p != DEFAULT_P:
        # resolve the participant's drinking_right rep from the clips dir
        hits = sorted({f.name.rsplit(".", 2)[0]
                       for f in (CLIPS / p).glob(f"{p}_drinking_right_*.mp4")})
        if not hits:
            raise SystemExit(f"no drinking_right rep found for {p} in {CLIPS/p}")
        stem = hits[0]

    print(f"[{p}] rep {stem}", flush=True)
    calib = load_calibration(f"data/calib/{p}/calibration.toml", target_size=RES)
    raw = load_dets(p, stem)
    dets = {c: v for c, v in raw.items() if c in calib}
    cams = sorted(dets, key=lambda k: int(k.split("_")[1]))
    n = min(len(v) for v in dets.values())
    print(f"  {len(cams)} calibrated cams, {n} frames -> running pipeline ...", flush=True)

    consensus, kf_est, kf_accept, rts_est = run_pipeline(calib, dets, cams, n)
    kf_cov = sum(e is not None for e in kf_est)
    cons_cov = sum(c is not None for c in consensus)
    print(f"  consensus cov {cons_cov}/{n}  KF cov {kf_cov}/{n}", flush=True)

    # --- Rerun sink: save / native spawn / browser web viewer (default) ---
    rr.init("drink_study_replay", recording_id=f"{p}_{stem}")
    if args.save:
        rr.save(args.save)
    elif args.spawn:
        rr.spawn()
    else:
        import urllib.parse
        server_uri = rr.serve_grpc(grpc_port=args.grpc_port)
        rr.serve_web_viewer(web_port=args.web_port, open_browser=False,
                            connect_to=server_uri)
        encoded = urllib.parse.quote(
            f"rerun+http://127.0.0.1:{args.grpc_port}/proxy", safe="")
        print(f"  open http://127.0.0.1:{args.web_port}/?url={encoded}", flush=True)
    vr.log_static_cameras(calib)

    # lazy-open the camera videos only if we're logging frames
    caps = {}
    if not args.no_frames:
        import cv2
        for c in cams:
            i = c.split("_")[1]
            mp4 = CLIPS / p / f"{stem}.{i}.mp4"
            if mp4.exists():
                caps[c] = cv2.VideoCapture(str(mp4))

    kf_trail, rts_trail = [], []
    for fr in range(n):
        rr.set_time("frame", sequence=fr)
        rr.set_time("time", duration=fr / ka.FPS)

        # camera frames + 2D detection overlays (colour by KF gate accept/reject)
        for c in cams:
            scale = vr.viz_scale_for(calib[c])
            if c in caps:
                ok, frame = caps[c].read()
                if ok:
                    vr.log_frame_downscaled(c, frame, scale)
            z = dets[c][fr]
            if z is not None:
                acc = kf_accept[fr].get(c, True)
                vr.log_cup_2d(c, z[0], z[1], acc, scale)
            else:
                rr.log(f"world/{c}/image/det", rr.Clear(recursive=False))

        # gated-consensus geometric truth (white)
        if consensus[fr] is not None:
            rr.log("world/consensus",
                   rr.Points3D((consensus[fr] / vr.MM_PER_M).reshape(1, 3),
                               colors=[[255, 255, 255, 230]], radii=[0.012]))
        else:
            rr.log("world/consensus", rr.Clear(recursive=False))

        # causal KF (red) and RTS-smoothed (green) tracks
        if kf_est[fr] is not None:
            log_point("world/cup_kf", kf_est[fr], [255, 80, 80, 255], 0.018,
                      kf_trail, [255, 80, 80, 180])
        if rts_est[fr] is not None:
            log_point("world/cup_rts", rts_est[fr], [80, 230, 120, 255], 0.018,
                      rts_trail, [80, 230, 120, 180])

    for cap in caps.values():
        cap.release()

    if args.save:
        print(f"  wrote {args.save}", flush=True)
    elif args.spawn:
        print("  viewer spawned (entities streamed). Scrub the 'frame' timeline.", flush=True)
    else:
        print("  web viewer serving; Ctrl-C to stop. Scrub the 'frame' timeline.", flush=True)
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
