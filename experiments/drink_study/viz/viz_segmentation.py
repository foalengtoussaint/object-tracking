"""Rerun viewer for the drink-PHASE SEGMENTATION results — for showing a supervisor.

Two things at once, on one scrubbable timeline:
  (1) the fused 3D cup track coloured BY PHASE (rest / forward-transport / DRINKING /
      back-transport / rest), with a moving cup marker that turns red during the dwell,
      and the sub-mm mocap track drawn faint for reference; and
  (2) the drink-DWELL comparison — TRUE (mocap gate) vs TUNED gate vs HYBRID learned
      segmenter — as three 0/1 timeline signals (the slide-9 finding, interactive), plus
      a synced best-camera video panel so you can SEE whether the cup is at the mouth in
      the frames the methods disagree on.

Everything comes from the SAME caches the deck numbers come from (cache/lopo_fused +
the learned-segmenter LOPO), so what you see matches the reported results exactly. The
HYBRID span is produced by training that rep's held-out LOPO fold (reuses
render_dwell_compare). No GPU needed for the 3D view; the hybrid fold + camera video use
GPU / video decode if available.

    python experiments/drink_study/viz_segmentation.py                       # default rep, spawn viewer
    python experiments/drink_study/viz_segmentation.py P23_..._151359         # a specific rep (video stem or substring)
    python experiments/drink_study/viz_segmentation.py --no-frames            # skip the camera video (3D + bands only, fast)
    python experiments/drink_study/viz_segmentation.py --no-hybrid            # skip training the fold (TRUE + TUNED only)
    python experiments/drink_study/viz_segmentation.py --save seg.rrd         # write a recording instead of spawning
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
import rerun as rr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import segment_cup_only as S
import learn_seg as LS                     # geo_span, TUN, HZ
import render_phase_compare as RP          # best_cam, TRACK, CLIPS
import gpu_decode

from _paths import CACHE as _C
LF = _C / "lopo_fused"
HZ = 60.0
MM_PER_M = 1000.0
# hex -> [r,g,b] for rerun
def _rgb(hexs):
    h = hexs.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]
PHASE_RGB = {k: _rgb(v) for k, v in S.PHASE_COLORS.items()}
DEFAULT = "P23_P23_drinking_right_20240716_151359__clean3d_refill"


def load_rep(key):
    """Find the lopo_fused npz whose video contains `key`."""
    files = sorted(glob.glob(str(LF / "*.npz")))
    hits = [f for f in files if key in Path(f).stem or key in str(np.load(f, allow_pickle=True)["video"])]
    if not hits:
        raise SystemExit(f"no lopo_fused rep matching '{key}'")
    d = np.load(hits[0], allow_pickle=True)
    return d, str(d["video"])


def phase_at(intervals, fr):
    for nm, s, e in intervals:
        if s <= fr < e:
            return nm
    return intervals[-1][0]


def span_signal(span, T):
    """0/1 array over T frames for a (s,e) dwell span."""
    a = np.zeros(T)
    if span:
        a[span[0]:span[1]] = 1.0
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rep", nargs="?", default=DEFAULT, help="video stem or substring")
    ap.add_argument("--no-frames", action="store_true", help="skip the camera video panel")
    ap.add_argument("--no-hybrid", action="store_true", help="skip training the hybrid fold")
    ap.add_argument("--save", default=None, help="write .rrd instead of spawning viewer")
    ap.add_argument("--spawn", action="store_true", help="spawn native viewer (default: web viewer)")
    ap.add_argument("--web-port", type=int, default=9090)
    ap.add_argument("--grpc-port", type=int, default=9876)
    args = ap.parse_args()

    d, video = load_rep(args.rep)
    fused = np.asarray(d["fused"], float)
    true = np.asarray(d["true"], float)
    T = len(fused)
    print(f"rep {video}  ({T} frames @ {HZ:.0f}Hz)", flush=True)

    # phases from the fused track (what the pipeline segments); dwell spans for the 3 methods
    seg = S.segment_cup_only(fused, fps=HZ, **LS.TUN)
    intervals = seg["intervals"]
    span_true = LS.geo_span(true)                       # mocap-gate "truth"
    span_tuned = LS.geo_span(fused, **LS.TUN)           # production gate on the video track
    span_hybrid = None
    if not args.no_hybrid:
        try:
            import render_dwell_compare as RD
            print("  training hybrid LOPO fold for this rep ...", flush=True)
            reps = RD.build(); byv = {r["video"]: r for r in reps}
            cin = reps[0]["fx"].shape[1]
            full = [k for k in byv if video in k or k in video]
            if full:
                span_hybrid = RD.hybrid_span(reps, byv, cin, full[0])
        except Exception as e:
            print(f"  (hybrid skipped: {e})", flush=True)
    print(f"  dwell spans  TRUE={span_true}  TUNED={span_tuned}  HYBRID={span_hybrid}", flush=True)

    # --- sink ---
    rr.init("drink_segmentation", recording_id=video)
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
    # static context: faint full mocap-true path + phase-coloured fused segments
    gmask = np.isfinite(true).all(1)
    if gmask.any():
        rr.log("world/mocap_true_path",
               rr.LineStrips3D([true[gmask] / MM_PER_M], colors=[120, 120, 160], radii=[0.0015]),
               static=True)
    for nm, s, e in intervals:
        seg_pts = fused[s:e + 1] / MM_PER_M
        if len(seg_pts) >= 2:
            rr.log(f"world/track_phases/{nm}",
                   rr.LineStrips3D([seg_pts], colors=[PHASE_RGB[nm]], radii=[0.004]), static=True)

    # camera video (best cam), synced -- 2D panel, no calib needed
    caps = None
    if not args.no_frames:
        try:
            tj = json.loads((RP.TRACK / f"{video}.json").read_text())
            clipstem = tj["stem"]; p = clipstem.split("_")[0]
            cam = RP.best_cam(p, clipstem); cn = cam.split("_")[1] if cam else "3"
            clip = RP.CLIPS / p / f"{clipstem}.{cn}.mp4"
            if clip.exists():
                caps = gpu_decode.frames(clip)
                print(f"  camera panel: {clip.name} (cam{cn})", flush=True)
            else:
                print(f"  (no clip {clip}; skipping camera panel)", flush=True)
        except Exception as e:
            print(f"  (camera panel skipped: {e})", flush=True)

    sig_true = span_signal(span_true, T)
    sig_tuned = span_signal(span_tuned, T)
    sig_hyb = span_signal(span_hybrid, T) if span_hybrid else None

    frame_iter = enumerate(caps) if caps is not None else ((fr, None) for fr in range(T))
    for fr, img in frame_iter:
        if fr >= T:
            break
        rr.set_time("frame", sequence=fr)
        rr.set_time("time", duration=fr / HZ)
        ph = phase_at(intervals, fr)
        col = PHASE_RGB[ph]

        # moving cup marker, coloured by current phase
        rr.log("world/cup", rr.Points3D((fused[fr] / MM_PER_M).reshape(1, 3),
                                        colors=[col], radii=[0.014]))
        # dwell-band signals as scalar time series (0/1)
        rr.log("dwell/TRUE_mocap", rr.Scalars(sig_true[fr]))
        rr.log("dwell/TUNED_gate", rr.Scalars(sig_tuned[fr]))
        if sig_hyb is not None:
            rr.log("dwell/HYBRID", rr.Scalars(sig_hyb[fr]))
        # a text label: current phase + which methods call this frame DRINK
        drk = [nm for nm, s in [("TRUE", sig_true), ("TUNED", sig_tuned),
                                ("HYBRID", sig_hyb if sig_hyb is not None else np.zeros(T))]
               if s[fr] > 0.5]
        disagree = ("HYBRID" in drk) and ("TRUE" not in drk)
        tag = f"phase={ph}   DRINK: {'/'.join(drk) if drk else '—'}"
        if disagree:
            tag += "   << HYBRID-only (cup at mouth, truth misses?)"
        rr.log("status", rr.TextLog(tag, level="WARN" if disagree else "INFO"))

        if img is not None:
            im = img[..., ::-1]                                # BGR->RGB
            h, w = im.shape[:2]
            if w > 960:                                        # downscale wide frames
                import cv2
                im = cv2.resize(im, (960, int(h * 960 / w)))
            rr.log("camera/best", rr.Image(im).compress(jpeg_quality=75))

    tail = ("wrote " + args.save) if args.save else "viewer serving — scrub the 'frame' timeline"
    print(f"  done. {tail}", flush=True)
    if not args.save and not args.spawn:
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
