"""Marker overlay WITH dwell bars — see WHERE truth / proxy21 / base17 disagree, on the video.

Projects the mocap cup (green) + head (blue) markers + cup centroid (yellow) into the video,
and adds a DWELL-TIMELINE strip along the bottom with a moving playhead:
    truth   (orange)  = mocap cup→head van-Andel dwell
    proxy21 (red)     = model prediction (+head distance)
    base17  (grey)    = model prediction (video only)
So you can watch the drink and tell if the model's dwell disagreement is the TRUTH being
wrong (dwell bar doesn't match the actual sip) or the MODEL being wrong.

    python experiments/drink_dwell/overlay.py <rep-substring> [--cam N]
Writes renders/OVERLAY_<rep>.mp4. Needs local footage + a trained worst_preds.json
(run plot.py first so proxy21/base17 spans exist for this rep).
"""
from __future__ import annotations
import sys as _s, pathlib as _p
_s.path.insert(0, str(_p.Path(__file__).resolve().parent))
import argparse, json
import numpy as np
import cv2

HERE = _p.Path(__file__).resolve().parent
_DS = HERE.parents[0] / "drink_study"
# reuse drink_study's calib + projection + track dir (shared infra, not duplicated)
_s.path.insert(0, str(_DS / "lib"))
_s.path.insert(0, str(_DS.parents[1]))
import kf_accuracy as ka
from mocap import load_trial, VIDEO_FPS, CUP_MARKERS, HEAD_MARKERS
from truth import dwell_truth
import features as F

RES = (1920, 1080)
CLIPS = _p.Path(__import__("os").environ.get("OT_CLIPS_ROOT", str(HERE.parents[1] / "clips")))
TRACK = _DS / "cache" / "track3d_clean3d_refill"
ALIGN = _DS / "cache" / "qtm_align.json"
PREDS = HERE / "cache" / "worst_preds.json"
OUTDIR = HERE / "renders"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rep"); ap.add_argument("--cam", default=None); ap.add_argument("--out", default=None)
    a = ap.parse_args()

    al = json.load(open(ALIGN)); V = {}
    for p in al.values():
        if isinstance(p, dict) and p.get("ok"):
            for r in p["reps"]:
                V[r["video"]] = r
    vk = [k for k in V if a.rep in k or k in a.rep]
    if not vk:
        raise SystemExit(f"no aligned rep matching '{a.rep}'")
    video = vk[0]; rec = V[video]; c3d = rec["c3d"]
    tj = json.loads((TRACK / f"{video}.json").read_text())
    stem = tj["stem"]; pid = stem.split("_")[0]
    cn = a.cam or "2"
    clip = CLIPS / pid / f"{stem}.{cn}.mp4"
    if not clip.exists():
        raise SystemExit(f"no clip {clip}")
    calib = ka.load_calibration(f"data/calib/{pid}/calibration.toml", target_size=RES)
    cam = calib[f"cam_{cn}"]

    def _xyz(fr):
        for k in ("rts", "consensus", "kf"):
            if fr.get(k) is not None:
                return fr[k]
        return [np.nan, np.nan, np.nan]
    cup_w0 = np.array([_xyz(fr) for fr in tj["frames"]], float)
    tr = load_trial(c3d)
    lag = rec["lag"]                                    # KNOWN-GOOD sync from qtm_align.json
    # SAME alignment the features use — one function, so overlay and model never disagree.
    R, t, rms = F.mocap_to_w0(cup_w0, tr.centroid(), tr.rate, lag)
    print(f"{video} c3d={c3d} cam_{cn}  Kabsch rms={rms:.1f}mm lag={lag} (sync={rec.get('sync_corr')})",
          flush=True)

    def to_w0(X):
        return X @ R.T + t
    idx = {l: i for i, l in enumerate(tr.labels)}
    cupm = tr.markers[:, [idx[m] for m in CUP_MARKERS if m in idx]]
    headm = tr.markers[:, [idx[m] for m in HEAD_MARKERS if m in idx]]
    cup_c = tr.centroid(); head_c = tr.head_centroid()
    dist = tr.cup_to_head()
    # MMC = the video/tracked cup (cup_w0) — ALREADY in W0, no Kabsch needed. Drawn MAGENTA so
    # you can compare it to the mocap cup (yellow): where the mocap cup gap-fills/teleports, the
    # video track stays continuous (or vice-versa). cup_w0 is on the 60Hz track grid.
    track_ratio = tr.rate / 60.0        # mocap-frame -> track-frame (track is 60Hz)

    # --- dwell spans, all mapped to VIDEO frames ---
    dw = dwell_truth(tr)
    ratio = tr.rate / VIDEO_FPS
    def mocap_span_to_video(sp):
        if not sp:
            return None
        # inverse of the per-frame map below: video f -> mocap mi; solve for f
        s = int(sp[0] / ratio) + (lag if lag >= 0 else 0)
        e = int(sp[1] / ratio) + (lag if lag >= 0 else 0)
        return (s, e)
    truth_v = mocap_span_to_video(dw.span)
    # model spans (track frames == video-track grid) from worst_preds.json
    prox_v = base_v = None
    if PREDS.exists():
        pc = json.load(open(PREDS))
        rp = pc.get(video)
        if rp:
            from model import span_from_prob
            for key, dst in [("proxy21", "prox"), ("base17", "base")]:
                if key in rp:
                    sp = span_from_prob(np.array(rp[key]["prob"]), rp[key]["thr"])
                    if key == "proxy21": prox_v = sp
                    else: base_v = sp
    # track frames are 60Hz; video is 30fps -> scale by 0.5 for the playhead strip
    def track_to_video(sp):
        return (int(sp[0] * VIDEO_FPS / 60.0), int(sp[1] * VIDEO_FPS / 60.0)) if sp else None
    prox_v = track_to_video(prox_v); base_v = track_to_video(base_v)

    cap = cv2.VideoCapture(str(clip))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); Hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Tv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sx, sy = W / RES[0], Hh / RES[1]
    OUTDIR.mkdir(exist_ok=True)
    out = a.out or str(OUTDIR / f"OVERLAY_{stem}.mp4")
    wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, Hh))

    def draw(img, X, color, r=5, x=False):
        if not np.isfinite(X).all():
            return
        (u, v), ok = ka.project(cam, X)
        if ok and np.isfinite(u) and np.isfinite(v):
            u, v = int(u * sx), int(v * sy)
            cv2.drawMarker(img, (u, v), color, cv2.MARKER_TILTED_CROSS, 18, 3) if x \
                else cv2.circle(img, (u, v), r, color, -1)

    # dwell strip geometry (bottom band)
    x0, x1 = 40, W - 40; ROWS = [("truth", truth_v, (60, 150, 235)),
                                 ("proxy21", prox_v, (60, 60, 220)),
                                 ("base17", base_v, (150, 150, 150))]
    def fx(f):
        return int(x0 + (x1 - x0) * f / max(Tv - 1, 1))

    fr = 0
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
            draw(img, to_w0(head_c[mi]), (60, 60, 235), x=True)
            draw(img, to_w0(cup_c[mi]), (60, 220, 235), r=6)                    # OMC cup (yellow)
            d = dist[mi]
            if np.isfinite(d):
                cv2.putText(img, f"cup->head {d:5.0f} mm", (40, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # MMC = tracked/video cup (already in W0). Track is per-video-frame, so index by fr.
        if 0 <= fr < len(cup_w0):
            draw(img, cup_w0[fr], (235, 60, 235), r=7, x=True)                  # MMC cup (magenta X)
        # legend
        cv2.putText(img, "OMC cup (mocap)", (40, Hh - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (60, 220, 235), 2)
        cv2.putText(img, "MMC cup (video track)", (230, Hh - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (235, 60, 235), 2)
        # --- dwell strip ---
        yb = Hh - 90
        for ri, (nm, sp, col) in enumerate(ROWS):
            y = yb + ri * 24
            cv2.line(img, (x0, y), (x1, y), (90, 90, 90), 1)
            cv2.putText(img, nm, (x0 - 34, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
            if sp:
                cv2.rectangle(img, (fx(sp[0]), y - 8), (fx(sp[1]), y + 8), col, -1)
        # playhead
        cv2.line(img, (fx(fr), yb - 12), (fx(fr), yb + len(ROWS) * 24), (255, 255, 255), 2)
        wr.write(img); fr += 1
        if fr % 60 == 0:
            print(f"  frame {fr}/{Tv}", flush=True)
    cap.release(); wr.release()
    print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
