"""Rerun viewer to CONFIRM the double-minimum question by eye: does the cup stay at the
mouth through the swallow-SHELF between the two minima (entry lip-contact + exit tip-up),
i.e. is it ONE continuous drink — and are the models wrong when they split it?

On one scrubbable timeline:
  - synced best-camera video (watch the cup at the lips through the shelf),
  - cup->mouth DISTANCE (mm) as a scalar (the two minima + the shelf are visible),
  - dwell as 0/1 signals for TRUTH (bridged), proxy21, base17 — see who splits vs bridges,
  - a status line flagging when the models DISAGREE (one calls drink, the other doesn't).

GPU-free: reuses the cached LOPO predictions in cache/worst_preds_bridged.json (no training)
and the bridged mouth truth. Video decode uses gpu_decode if available.

    python experiments/drink_study/viz_doublemin.py P10_..._153141 --save out.rrd
    python experiments/drink_study/viz_doublemin.py P13_..._161825            # web viewer
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import argparse, json, sys
from pathlib import Path
import numpy as np
import rerun as rr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import learn_seg as LS
import mouth_features as MF
import render_phase_compare as RP
import gpu_decode

HZ = 60.0
MM_PER_M = 1000.0
from _paths import CACHE as _C
PRED = _C / "worst_preds_bridged.json"
LF = _C / "lopo_fused"
DEFAULT = "P10_P10_drinking_right_20240202_153141__clean3d_refill"


def _span(prob, thr):
    return LS.span_from_prob(np.asarray(prob), thr)


def _load(video_sub):
    preds = json.load(open(PRED))
    hits = [k for k in preds if video_sub in k or k in video_sub]
    if not hits:
        raise SystemExit(f"no cached prediction for '{video_sub}' in {PRED.name}")
    v = hits[0]
    import glob
    fh = [f for f in glob.glob(str(LF / "*.npz")) if v in str(np.load(f, allow_pickle=True)["video"])]
    d = np.load(fh[0], allow_pickle=True)
    fused = np.asarray(d["fused"], float); T = len(fused)
    dist = MF.mouth_channels(v, T)[:, 0]                       # cup->mouth mm
    tsp = MF.mouth_span(v, T)                                  # bridged truth span
    p = preds[v]
    sp_prox = _span(p["proxy21"]["prob"], p["proxy21"]["thr"])
    sp_base = _span(p["base17"]["prob"], p["base17"]["thr"])
    pr_prox = np.asarray(p["proxy21"]["prob"]); pr_base = np.asarray(p["base17"]["prob"])
    return v, fused, dist, tsp, sp_prox, sp_base, pr_prox, pr_base, T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rep", nargs="?", default=DEFAULT)
    ap.add_argument("--no-frames", action="store_true")
    ap.add_argument("--save", default=None)
    ap.add_argument("--spawn", action="store_true")
    ap.add_argument("--web-port", type=int, default=9092)
    ap.add_argument("--grpc-port", type=int, default=9878)
    args = ap.parse_args()

    v, fused, dist, tsp, sp_prox, sp_base, pr_prox, pr_base, T = _load(args.rep)
    print(f"rep {v}  T={T}", flush=True)
    print(f"  bridged TRUTH dwell={tsp}  proxy21={sp_prox}  base17={sp_base}", flush=True)

    rr.init("doublemin", recording_id=v)
    if args.save:
        rr.save(args.save)
    elif args.spawn:
        rr.spawn()
    else:
        import urllib.parse
        uri = rr.serve_grpc(grpc_port=args.grpc_port)
        rr.serve_web_viewer(web_port=args.web_port, open_browser=False, connect_to=uri)
        enc = urllib.parse.quote(f"rerun+http://127.0.0.1:{args.grpc_port}/proxy", safe="")
        print(f"  open http://127.0.0.1:{args.web_port}/?url={enc}", flush=True)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    m = np.isfinite(fused).all(1)
    if m.any():
        rr.log("world/track", rr.LineStrips3D([fused[m] / MM_PER_M], colors=[[120, 120, 160]],
                                              radii=[0.003]), static=True)

    def sig(span):
        a = np.zeros(T)
        if span:
            a[span[0]:span[1]] = 1.0
        return a
    s_true, s_prox, s_base = sig(tsp), sig(sp_prox), sig(sp_base)

    caps = None
    if not args.no_frames:
        try:
            tj = json.loads((RP.TRACK / f"{v}.json").read_text())
            cs = tj["stem"]; p = cs.split("_")[0]
            cam = RP.best_cam(p, cs); cn = cam.split("_")[1] if cam else "3"
            clip = RP.CLIPS / p / f"{cs}.{cn}.mp4"
            if clip.exists():
                caps = gpu_decode.frames(clip)
                print(f"  camera: {clip.name} (cam{cn})", flush=True)
        except Exception as e:
            print(f"  (camera skipped: {e})", flush=True)

    frame_iter = enumerate(caps) if caps is not None else ((fr, None) for fr in range(T))
    for fr, img in frame_iter:
        if fr >= T:
            break
        rr.set_time("frame", sequence=fr)
        rr.set_time("time", duration=fr / HZ)
        if np.isfinite(fused[fr]).all():
            near = np.isfinite(dist[fr]) and dist[fr] < 80
            rr.log("world/cup", rr.Points3D((fused[fr] / MM_PER_M).reshape(1, 3),
                                            colors=[[220, 60, 60] if near else [90, 160, 90]],
                                            radii=[0.016]))
        rr.log("cup_to_mouth_mm", rr.Scalars(float(dist[fr]) if np.isfinite(dist[fr]) else 0.0))
        rr.log("dwell/TRUTH_bridged", rr.Scalars(s_true[fr]))
        rr.log("dwell/proxy21", rr.Scalars(s_prox[fr]))
        rr.log("dwell/base17", rr.Scalars(s_base[fr]))
        rr.log("prob/proxy21", rr.Scalars(float(pr_prox[fr])))
        rr.log("prob/base17", rr.Scalars(float(pr_base[fr])))
        # disagreement flag
        drk = [nm for nm, s in [("TRUTH", s_true), ("proxy21", s_prox), ("base17", s_base)]
               if s[fr] > 0.5]
        disagree = len(drk) not in (0, 3)
        tag = f"cup→mouth {dist[fr]:5.0f}mm   drink: {'/'.join(drk) if drk else '—'}"
        if disagree:
            tag += "   << MODELS DISAGREE"
        rr.log("status", rr.TextLog(tag, level="WARN" if disagree else "INFO"))
        if img is not None:
            im = img[..., ::-1]
            h, w = im.shape[:2]
            if w > 960:
                import cv2
                im = cv2.resize(im, (960, int(h * 960 / w)))
            rr.log("camera/best", rr.Image(im).compress(jpeg_quality=75))

    tail = ("wrote " + args.save) if args.save else "viewer serving — scrub 'frame'"
    print(f"  done. {tail}", flush=True)
    if not args.save and not args.spawn:
        import time
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
