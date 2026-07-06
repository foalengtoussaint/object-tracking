"""Render a video comparing the video cup track vs QTM mocap ground truth.

Overlays per frame, reprojected into one real camera view:
  * GREEN circle = video 3D cup track (RTS, cache/track3d_clean3d_refill)
  * CYAN  circle = QTM optical mocap cup centroid, transformed QTM-lab -> W0 by
    the per-rep Kabsch (qtm_align), then reprojected
  * a live error readout = 3D distance (mm) between the two

Where green drifts off the real cup while cyan stays on it, you SEE the
confident-wrong tracking failure the mocap exposes (e.g. P14 left vs right).

    python render_qtm_compare.py P14 left 4        # participant side rep_index
    python render_qtm_compare.py P14 right 0
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
import kf_accuracy as ka
from kalman_3d import load_calibration, project
import gpu_decode
from qtm_video_map import build_map, monotonic_align
from qtm_align import (video_reps, video_track, _resample, _speed, _xcorr_lag,
                       kabsch, VIDEO_FPS, COMMON_HZ)
from qtm_c3d import load_trial

CLIPS = Path(os.environ.get("OT_CLIPS_ROOT", "/home/imove/Documents/clips"))
from _paths import CACHE as _C
OUT = _C / "qtm_compare_videos"


def _pair(pid, side, rep_idx):
    """Return (video_rep_path, c3d_stem, R, t, lag) for one rep, R/t fit on THIS rep."""
    m = build_map()[pid]
    reps = video_reps(pid, side)
    c3d = [(c, r) for c, r in m.pairs if (r.side or "").lower() == side]
    vdur = [video_track(r).shape[0] / VIDEO_FPS for r in reps]
    cdur = [load_trial(c).duration_s for c, _ in c3d]
    idx = dict(monotonic_align(vdur, cdur)[0])
    if rep_idx not in idx:
        raise SystemExit(f"rep {rep_idx} not in aligned pairs for {pid} {side}")
    vpath = reps[rep_idx]; c3d_stem = c3d[idx[rep_idx]][0]
    # sync + fit transform on this rep
    tr = load_trial(c3d_stem)
    vr = _resample(video_track(vpath), VIDEO_FPS); mr = _resample(tr.centroid(), tr.rate)
    lag, corr = _xcorr_lag(_speed(vr, COMMON_HZ), _speed(mr, COMMON_HZ))
    if lag >= 0:
        v = vr[lag:]; mo = mr[: len(v)]
    else:
        mo = mr[-lag:]; v = vr[: len(mo)]
    L = min(len(v), len(mo)); v, mo = v[:L], mo[:L]
    ok = ~(np.isnan(v).any(1) | np.isnan(mo).any(1))
    R, t, rms = kabsch(mo[ok], v[ok])
    print(f"{pid} {side} rep{rep_idx}: {vpath.stem} <-> {c3d_stem}  lag={lag}  "
          f"sync_corr={corr:.2f}  rep_rms={rms:.1f}mm")
    return vpath, c3d_stem, R, t, lag


def render(pid, side, rep_idx):
    vpath, c3d_stem, R, t, lag = _pair(pid, side, rep_idx)
    d = json.loads(vpath.read_text())
    frames = d["frames"]
    rts = [f["rts"] for f in frames]
    kept = [len(f["kept"] or []) for f in frames]

    # mocap centroid in W0, resampled to video fps, lag-shifted to align frame indices
    tr = load_trial(c3d_stem)
    mr = _resample(tr.centroid(), tr.rate)            # (Tm,3) lab @ 60Hz
    moW = mr @ R.T + t                                 # -> W0
    # video frame fr corresponds to mocap index (fr - lag)
    def mocap_at(fr):
        j = fr - lag
        return moW[j] if 0 <= j < len(moW) else None

    clipstem = vpath.stem
    for suf in ("__clean3d_refill", "__pscale_4"):
        clipstem = clipstem.replace(suf, "")
    clipstem = clipstem[len(pid) + 1:]                 # drinking_left_2024...
    calib = load_calibration(f"data/calib/{pid}/calibration.toml", target_size=ka.RES)

    # pick the camera where the two tracks are most separated on average (most telling)
    best_cam, best_sep = None, -1
    for cam, cc in calib.items():
        seps = []
        for fr in range(min(len(rts), len(moW))):
            mp = mocap_at(fr)
            if rts[fr] is None or mp is None:
                continue
            a, fa = project(cc, np.array(rts[fr], float))
            b, fb = project(cc, np.array(mp, float))
            if fa and fb:
                seps.append(np.hypot(*(a - b)))
        if seps and np.median(seps) > best_sep:
            best_sep, best_cam = float(np.median(seps)), cam
    cam = best_cam; camnum = cam.split("_")[1]; cc = calib[cam]

    video = CLIPS / pid / f"{clipstem}.{camnum}.mp4"
    if not video.exists():
        raise SystemExit(f"clip not found: {video}")
    W, H, nv, _ = gpu_decode.dims(video)
    n = min(len(frames), nv)
    OUT.mkdir(parents=True, exist_ok=True)
    sf = 0.6; ow, oh = int(W * sf), int(H * sf)
    tag = f"{pid}_{side}_rep{rep_idx}_cam{camnum}"
    outp = OUT / f"{tag}.mp4"
    vw = cv2.VideoWriter(str(outp), cv2.VideoWriter_fourcc(*"mp4v"), 30, (ow, oh))

    for fr, img in enumerate(gpu_decode.frames(video)):
        if fr >= n:
            break
        mp = mocap_at(fr)
        err = None
        if rts[fr] is not None and mp is not None:
            err = float(np.linalg.norm(np.array(rts[fr]) - mp))
        # GT mocap (cyan)
        if mp is not None:
            uv, infront = project(cc, np.array(mp, float))
            if infront:
                cv2.circle(img, (int(uv[0]), int(uv[1])), 16, (255, 255, 0), 3)
        # video track (green)
        if rts[fr] is not None:
            uv, infront = project(cc, np.array(rts[fr], float))
            if infront:
                cv2.circle(img, (int(uv[0]), int(uv[1])), 12, (0, 230, 0), 3)
        cv2.rectangle(img, (0, 0), (W, 60), (40, 40, 40), -1)
        cv2.putText(img, f"{pid} {side} rep{rep_idx}  t={fr/VIDEO_FPS:4.1f}s  cams={kept[fr]}"
                    + (f"  ERR={err:5.0f}mm" if err is not None else "  ERR=--"),
                    (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0, 0, 255) if (err or 0) > 100 else (0, 255, 0), 2)
        cv2.putText(img, "green=video track   cyan=QTM ground truth", (12, H - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        vw.write(cv2.resize(img, (ow, oh)))
    vw.release()
    print(f"wrote {outp}  (cam {camnum}, {n} frames, median 2D sep {best_sep:.0f}px)", flush=True)
    return outp


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("participant")
    ap.add_argument("side", choices=["left", "right"])
    ap.add_argument("rep", type=int)
    args = ap.parse_args()
    render(args.participant, args.side, args.rep)


if __name__ == "__main__":
    main()
