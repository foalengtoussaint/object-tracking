"""Multi-camera video of the refill-model detections with the inlier-gate verdict.

For one trial, lays out all calibrated cameras in a grid. On every frame:
  - every refill-model raw detection is drawn as a dot, coloured GREEN if it
    survived the >=3-cam inlier gate (kept), RED if it was rejected;
  - the 3D track point is reprojected into each camera as a hollow circle, CYAN
    when the frame has a real consensus, MAGENTA when the value is interpolated
    (RTS through a gap -> no consensus this frame).
A header strip shows t/frame, #raw cams, #kept cams, and CONSENSUS vs INTERPOLATED.

    python experiments/drink_study/render_refill_gate.py P23_P23_drinking_right_20240312_115737
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gpu_decode
import kf_accuracy as ka
import segment_cup_only as sc
from kalman_3d import load_calibration, triangulate_dlt, project
from cache_track3d import load_dets, kept_and_px

CLIPS = Path(os.environ.get("OT_CLIPS_ROOT", "/home/imove/Documents/clips"))
DET = Path("experiments/drink_study/cache/student_dets_clean3d_refill")
TRACK = Path("experiments/drink_study/cache/track3d_clean3d_refill")
OUT = Path("experiments/drink_study/cache/refill_gate")
FPS = 60.0
TILE_W, TILE_H = 480, 270          # per-camera tile (16:9 half-ish)
HEAD = 110                         # room for header text + cup-phase strip
# cup-only phase colours (BGR)
PC = {"rest_pre": (190, 190, 190), "forward_transport": (232, 155, 76),
      "drinking": (76, 85, 232), "back_transport": (59, 162, 240),
      "rest_post": (140, 140, 140)}


def cup_phases(rts, T):
    """segment_cup_only on a track (list of [x,y,z]|None); returns intervals."""
    xyz = np.array([p if p else [np.nan] * 3 for p in rts], float)
    v = np.isfinite(xyz).all(1); idx = np.flatnonzero(v)
    if len(idx) < 10:
        return []
    for a in range(3):
        xyz[:, a] = np.interp(np.arange(len(xyz)), idx, xyz[idx, a])
    return sc.segment_cup_only(xyz)["intervals"]


def phase_at(intervals, fi):
    for n, s, e in intervals:
        if s <= fi < e:
            return n
    return "?"


def draw_phase_strip(canvas, y, intervals, T, x0, W, fi):
    h = 22
    bw = W - x0 - 10
    for n, s, e in intervals:
        a = x0 + int(s / T * bw); b = x0 + int(e / T * bw)
        cv2.rectangle(canvas, (a, y), (b, y + h), PC.get(n, (100, 100, 100)), -1)
    cx = x0 + int(fi / T * bw)
    cv2.line(canvas, (cx, y - 2), (cx, y + h + 2), (255, 255, 255), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trial")
    ap.add_argument("--start", type=float, default=0.0, help="start time (s)")
    ap.add_argument("--end", type=float, default=1e9, help="end time (s)")
    ap.add_argument("--rescue", action="store_true",
                    help="overlay 2-cam rescues (rescue2cam_<trial>.json): yellow dots/circle")
    ap.add_argument("--full", action="store_true",
                    help="use the FULL-weight rescue file (rescue2cam_<trial>_full.json)")
    args = ap.parse_args()
    trial = args.trial
    p = trial.split("_")[0]
    stem = trial[len(p) + 1:]

    calib = load_calibration(f"data/calib/{p}/calibration.toml", target_size=ka.RES)
    detf = DET / f"{p}_{stem}__clean3d_refill__c0.25.json"
    dets = load_dets(detf, calib)
    cams = sorted(dets, key=lambda k: int(k.split("_")[1]))
    n = min(len(v) for v in dets.values())

    # 3D track (consensus + rts) from the cached consensus-anchored track
    tr = json.loads((TRACK / f"{trial}__clean3d_refill.json").read_text())["frames"]
    cons3d = [f["consensus"] for f in tr]
    rts3d = [f["rts"] for f in tr]

    # precompute per-frame kept-camera set
    kept_per = []
    for fr in range(n):
        obs = {c: dets[c][fr] for c in cams if dets[c][fr] is not None}
        kept, _ = kept_and_px(obs, calib) if len(obs) >= 2 else ([], None)
        kept_per.append(set(kept))

    # optional 2-cam rescue overlay: {fr: set(pair)} + rescued rts_new for the circle
    rescue_pair = {}; rts_new = None
    if args.rescue:
        rsfx = "_full" if args.full else ""
        rf = Path(f"experiments/drink_study/cache/rescue2cam_{trial}{rsfx}.json")
        rd = json.loads(rf.read_text())
        rescue_pair = {int(fr): set(pair) for fr, pair, *_ in rd["rescued"]}
        rts_new = rd["rts_new"]
        rts3d = rts_new                                 # draw the rescued track
        print(f"[rescue] {len(rescue_pair)} rescued frames overlaid", flush=True)

    # cup-only phases on the track being drawn (rescued if --rescue, else hard-gate)
    phase_iv = cup_phases(rts3d, n)
    drink_n = sum(e - s for nm, s, e in phase_iv if nm == "drinking")
    print(f"[phases] {len(phase_iv)} intervals; drinking frames={drink_n} "
          f"({'present' if drink_n else 'NONE'})", flush=True)

    # grid layout
    ncam = len(cams)
    cols = 4 if ncam > 6 else 3
    rows = (ncam + cols - 1) // cols
    GW, GH = cols * TILE_W, rows * TILE_H

    # open all camera decoders lazily -> read frame-synced by index
    vids = {}
    for c in cams:
        cn = c.split("_")[1]
        vp = CLIPS / p / f"{stem}.{cn}.mp4"
        if not vp.exists():
            vp = Path(f"data/calib/{p}/cam-{cn}.mp4")    # fallback (won't match stem)
        vids[c] = vp

    f0 = int(args.start * FPS)
    f1 = min(n, int(args.end * FPS)) if args.end < 1e9 else n
    OUT.mkdir(parents=True, exist_ok=True)
    rtag = ("_rescue_full" if args.full else "_rescue") if args.rescue else ""
    outp = OUT / f"{trial}__refillgate{rtag}.mp4"
    writer = cv2.VideoWriter(str(outp), cv2.VideoWriter_fourcc(*"mp4v"), 30, (GW, GH + HEAD))
    print(f"[{trial}] {ncam} cams, frames {f0}-{f1} -> {outp}", flush=True)

    # open generators
    gens = {c: gpu_decode.frames(vids[c]) for c in cams}
    cur = {c: None for c in cams}

    SCX, SCY = TILE_W / ka.RES[0], TILE_H / ka.RES[1]
    for fr in range(f1):
        for c in cams:
            try:
                cur[c] = next(gens[c])
            except StopIteration:
                pass
        if fr < f0:
            continue
        canvas = np.zeros((GH + HEAD, GW, 3), np.uint8)
        kept = kept_per[fr]
        rpair = rescue_pair.get(fr, set())
        X = np.array(rts3d[fr], float) if rts3d[fr] else None
        has_cons = cons3d[fr] is not None
        is_rescue = bool(rpair)                          # this frame filled by a 2-cam rescue
        for i, c in enumerate(cams):
            ry, rx = divmod(i, cols)
            ox, oy = rx * TILE_W, ry * TILE_H + HEAD
            img = cur[c]
            tile = cv2.resize(img, (TILE_W, TILE_H)) if img is not None else np.zeros((TILE_H, TILE_W, 3), np.uint8)
            # reprojected 3D track point: cyan=consensus, yellow=rescued, magenta=interp
            if X is not None:
                (u, v), ok = project(calib[c], X)
                if ok:
                    col = (255, 200, 0) if has_cons else (0, 230, 230) if is_rescue else (200, 0, 200)
                    cv2.circle(tile, (int(u * SCX), int(v * SCY)), 9, col, 2)
            # raw detection: green=kept, yellow=used in rescue pair, red=rejected
            d = dets[c][fr]
            if d is not None:
                u, v = d
                col = (0, 220, 0) if c in kept else (0, 230, 230) if c in rpair else (0, 0, 255)
                cv2.circle(tile, (int(u * SCX), int(v * SCY)), 5, col, -1)
            cv2.putText(tile, c, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            canvas[oy:oy + TILE_H, ox:ox + TILE_W] = tile
        # header
        nraw = sum(dets[c][fr] is not None for c in cams)
        state = "CONSENSUS" if has_cons else "RESCUED (2-cam)" if is_rescue else "INTERPOLATED"
        scol = (0, 220, 0) if has_cons else (0, 230, 230) if is_rescue else (200, 0, 200)
        cv2.putText(canvas, f"{trial}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(canvas, f"t={fr/FPS:5.2f}s  f={fr}   raw={nraw}cam  kept={len(kept)}cam",
                    (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(canvas, state, (GW - 380, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, scol, 2)
        cv2.putText(canvas, "green=kept yellow=rescued(2cam) red=rejected  O cyan=cons/yellow=rescue/magenta=interp",
                    (GW - 760, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
        # cup-only phase strip + current phase label
        cur_ph = phase_at(phase_iv, fr)
        cv2.putText(canvas, "CUP PHASE:", (8, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(canvas, cur_ph, (118, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    PC.get(cur_ph, (255, 255, 255)), 2)
        draw_phase_strip(canvas, 96, phase_iv, n, 300, GW, fr)
        writer.write(canvas)
    writer.release()
    print(f"wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
