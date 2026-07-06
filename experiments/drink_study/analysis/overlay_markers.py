"""Draw the mocap MARKERS into the video to debug the cup->mouth distance.

The distance's two minima don't look like cup-at-mouth in the raw video — so either the
mouth PROXY is mis-placed or the alignment is off. This projects the actual mocap points
into the camera so you can SEE where the proxy sits:
  - cup markers (4)          GREEN dots
  - head markers (5)         BLUE dots
  - mouth proxy              RED X          <- is it on the mouth?
  - cup centroid             YELLOW dot
and prints the live cup->mouth distance + a bar that fills as the distance shrinks.

Alignment: the mocap is in the QTM lab frame; the video pipeline's cup TRACK is in the
camera-calib W0 frame. We recover mocap->W0 by robust Kabsch on the synced cup-centroid vs
the video track (same recipe as qtm_align), then project with kalman_3d.

    python experiments/drink_study/overlay_markers.py P13_..._161825 [--cam 2] [--out x.mp4]
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import argparse, json, sys
from pathlib import Path
import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                          # repo root holds kalman_3d.py
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import qtm_align as Q
import kalman_3d as ka
from qtm_c3d import load_trial
from _paths import QTM_C3D_HEAD, CLIPS_ROOT
import render_phase_compare as RP

RES = (1920, 1080)     # calib native size == clip size for these BRIO recordings


def mocap_to_w0(cup_track_w0, mocap_centroid, rate):
    """Robust Kabsch mapping mocap (lab) -> W0, from synced cup-centroid vs video track.
    Returns (R, t, lag, corr) with mocap_w0 = mocap @ R.T + t."""
    vr = Q._resample(cup_track_w0, Q.VIDEO_FPS)
    mr = Q._resample(mocap_centroid, rate)
    lag, corr = Q._xcorr_lag(Q._speed(vr, Q.COMMON_HZ), Q._speed(mr, Q.COMMON_HZ))
    if lag >= 0:
        v = vr[lag:]; mo = mr[:len(v)]; voff = lag
    else:
        mo = mr[-lag:]; v = vr[:len(mo)]; voff = 0
    L = min(len(v), len(mo)); v, mo = v[:L], mo[:L]
    ok = ~(np.isnan(v).any(1) | np.isnan(mo).any(1))
    R, t, rms = Q.kabsch(mo[ok], v[ok], robust=True)
    return R, t, lag, corr, voff, rms


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rep")
    ap.add_argument("--cam", default=None, help="camera number (default: best cam)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # resolve the video rep -> track json + c3d pairing
    from _paths import CACHE as _C
    al = json.load(open(_C / "qtm_align.json")); V = {}
    for p in al.values():
        if isinstance(p, dict) and p.get("ok"):
            for r in p["reps"]:
                V[r["video"]] = r
    vk = [k for k in V if args.rep in k or k in args.rep]
    if not vk:
        raise SystemExit(f"no aligned rep matching '{args.rep}'")
    video = vk[0]; rec = V[video]; c3d = rec["c3d"]
    tj = json.loads((RP.TRACK / f"{video}.json").read_text())
    stem = tj["stem"]; p = stem.split("_")[0]
    cn = args.cam or (RP.best_cam(p, stem) or "cam_2").split("_")[1]
    clip = CLIPS_ROOT / p / f"{stem}.{cn}.mp4"
    if not clip.exists():
        raise SystemExit(f"no clip {clip}")
    calib = ka.load_calibration(f"data/calib/{p}/calibration.toml", target_size=RES)
    cam = calib[f"cam_{cn}"]
    print(f"{video}  c3d={c3d}  cam_{cn}", flush=True)

    # cup track in W0 (the pipeline's own 3D track): prefer rts, else consensus/kf
    def _xyz(fr):
        for k in ("rts", "consensus", "kf"):
            if fr.get(k) is not None:
                return fr[k]
        return [np.nan, np.nan, np.nan]
    cup_w0 = np.array([_xyz(fr) for fr in tj["frames"]], float)
    # mocap trial: markers + mouth proxy (lab frame)
    tr = load_trial(c3d, root=QTM_C3D_HEAD)
    cup_c = tr.centroid()
    R, t, lag, corr, voff, rms = mocap_to_w0(cup_w0, cup_c, tr.rate)
    print(f"  mocap->W0 Kabsch rms={rms:.1f}mm  sync corr={corr:.2f} lag={lag}", flush=True)

    def to_w0(X):                     # (N,3) mocap -> W0
        return X @ R.T + t
    from qtm_c3d import CUP_MARKERS, HEAD_MARKERS
    idx = {l: i for i, l in enumerate(tr.labels)}
    cupm = tr.markers[:, [idx[m] for m in CUP_MARKERS if m in idx]]        # (T,4,3)
    headm = tr.markers[:, [idx[m] for m in HEAD_MARKERS if m in idx]]      # (T,5,3)
    mouth = tr.head_centroid()                                            # head-centroid point
    dist = tr.cup_to_head()                                               # the NEW dwell signal

    cap = cv2.VideoCapture(str(clip))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); Hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sx, sy = W / RES[0], Hh / RES[1]
    out = args.out or str(_C / "rrd" / f"OVERLAY_{p}_{cn}.mp4")
    wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, Hh))

    def draw(img, X_w0, color, r=5, x=False):
        if not np.isfinite(X_w0).all():
            return
        (u, v), ok = ka.project(cam, X_w0)
        if ok and np.isfinite(u) and np.isfinite(v):
            u, v = int(u * sx), int(v * sy)
            if x:
                cv2.drawMarker(img, (u, v), color, cv2.MARKER_TILTED_CROSS, 18, 3)
            else:
                cv2.circle(img, (u, v), r, color, -1)

    # video frame f corresponds to mocap frame: mo_i where video index = voff + i*(VFPS/rate)
    # simpler: resample mocap arrays to video length via the lag mapping used for distance
    Tv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # map each video frame -> mocap frame (account for lag; rate ratio)
    ratio = tr.rate / Q.VIDEO_FPS
    fr = 0
    dmin, dmax = np.nanmin(dist), np.nanpercentile(dist, 95)
    while True:
        ret, img = cap.read()
        if not ret:
            break
        mi = int(round((fr - lag) * ratio)) if lag >= 0 else int(round(fr * ratio)) - (-lag)
        if 0 <= mi < tr.n_frames:
            for k in range(cupm.shape[1]):
                draw(img, to_w0(cupm[mi, k]), (80, 220, 80))
            for k in range(headm.shape[1]):
                draw(img, to_w0(headm[mi, k]), (230, 140, 60))
            draw(img, to_w0(mouth[mi]), (60, 60, 235), x=True)                 # mouth proxy = red X
            draw(img, to_w0(cup_c[mi]), (60, 220, 235), r=6)                   # cup centroid = yellow
            d = dist[mi]
            if np.isfinite(d):
                frac = float(np.clip(1 - (d - dmin) / (dmax - dmin + 1e-6), 0, 1))
                cv2.rectangle(img, (20, 20), (20 + int(300 * frac), 45), (60, 60, 235), -1)
                cv2.rectangle(img, (20, 20), (320, 45), (255, 255, 255), 1)
                cv2.putText(img, f"cup->mouth {d:5.0f} mm", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        wr.write(img)
        fr += 1
        if fr % 60 == 0:
            print(f"  frame {fr}/{Tv}", flush=True)
    cap.release(); wr.release()
    print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
