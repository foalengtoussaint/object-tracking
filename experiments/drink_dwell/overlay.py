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
import argparse, json, glob
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


def _velocity_fit(cup_track, tr, lag0, search=8, hz=60.0, speed=80.0, good=20.0):
    """VELOCITY-VECTOR rotation: R = argmin |v_video - R v_mocap| on moving drink frames, best lag.
    Returns (R, drink_good_frac, lag) or (None,0,lag0). Rotation only (translation from centroids)."""
    from mocap import resample as _rs, VIDEO_FPS as _VF
    from velfit import _procrustes_rot
    vd = np.diff(_rs(cup_track, _VF), axis=0) * hz
    omc60 = _rs(tr.centroid(), tr.rate)
    n = len(vd) + 1
    dw = dwell_truth(tr); drink = np.zeros(n - 1, bool)
    sp = dw.span_at(n) if dw.span else None
    if sp:
        drink[sp[0]:min(sp[1], n - 1)] = True
    vs = np.linalg.norm(vd, axis=1)
    best = (None, 0.0, lag0)
    for lag in range(lag0 - search, lag0 + search + 1):
        idxo = np.arange(n) - lag
        ok = (idxo >= 1) & (idxo < len(omc60))
        od = np.full((n, 3), np.nan)
        od[ok] = (omc60[idxo[ok]] - omc60[idxo[ok] - 1]) * hz
        od = od[:-1]
        os = np.linalg.norm(od, axis=1)
        mv = (vs > speed) & (os > speed) & np.isfinite(vs) & np.isfinite(os)
        ff = mv & drink
        if ff.sum() < 6:
            ff = mv
        if ff.sum() < 6:
            continue
        R = _procrustes_rot(vd[ff], od[ff])
        rv = od @ R.T; rs = np.linalg.norm(rv, axis=1)
        cos = np.sum(vd * rv, 1) / (vs * rs + 1e-9)
        ev = mv & drink
        if ev.sum() < 4:
            ev = mv
        g = float(np.mean(np.degrees(np.arccos(np.clip(cos[ev], -1, 1))) < good))
        if g > best[1]:
            best = (R, g, lag)
    return best
CLIPS = _p.Path(__import__("os").environ.get("OT_CLIPS_ROOT", str(HERE.parents[1] / "clips")))
TRACK = _DS / "cache" / "track3d_clean3d_refill"
ALIGN = _DS / "cache" / "qtm_align.json"
PREDS = HERE / "cache" / "worst_preds.json"
OUTDIR = HERE / "renders"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rep"); ap.add_argument("--cam", default=None); ap.add_argument("--out", default=None)
    ap.add_argument("--compare-fits", action="store_true",
                    help="also draw a comparison fit (cyan) next to the main fit (yellow)")
    ap.add_argument("--session", action="store_true",
                    help="rotate the mocap cup by the robust SESSION rotation (constant per rig) "
                         "with per-trial translation — overrides this trial's degenerate per-trial "
                         "rotation; --compare-fits shows per-trial(cyan) vs session-R(yellow)")
    ap.add_argument("--scale", action="store_true",
                    help="session-R held fixed + fit scale s + t on top; prints s")
    ap.add_argument("--pscale", action="store_true",
                    help="apply the PER-PARTICIPANT scale s_p (cache/per_participant_scale.json): "
                         "un-compress the W0/video side by 1/s_p about its centroid before the fit. "
                         "P24/P16/P23/P02 are ~3-6% compressed; visible in the HEAD markers.")
    ap.add_argument("--simrot", action="store_true",
                    help="ONE global (R, s, t) fit JOINTLY over the whole session (rotation AND "
                         "scale together) via Umeyama")
    ap.add_argument("--velfit", action="store_true",
                    help="rotate the mocap cup by the VELOCITY-VECTOR fit (Procrustes on 3D "
                         "velocity) instead of the position Kabsch — fixes the degenerate-"
                         "rotation reps (P16); --compare-fits then shows position(cyan) vs velocity(yellow)")
    ap.add_argument("--fused", action="store_true",
                    help="fit the alignment on the KF-smoothed fused track (default is RAW consensus, "
                         "which doesn't overshoot at the drink); with --compare-fits, cyan=fused vs yellow=raw")
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
    cup_w0 = np.array([_xyz(fr) for fr in tj["frames"]], float)     # smoothed (rts/kf) track for display
    # RAW consensus track (unfiltered, NaN at occlusion). Used for the FIT by default.
    cup_raw = np.array([fr.get("consensus") if fr.get("consensus") is not None else [np.nan]*3
                        for fr in tj["frames"]], float)
    # FUSED = TCN velocity-fill track (fills the occluded gaps) — from the npz, not the JSON.
    cup_fused = None
    _npz = glob.glob(str(F.FUSED_DIR / f"*{video}*.npz"))
    if _npz:
        cup_fused = np.asarray(np.load(_npz[0], allow_pickle=True)["fused"], float)
    # BIOMECH head joint 67 (markerless, ALREADY in W0 mm) — the head this session's head+cup fit uses.
    # One point per video frame; drawn WHITE so you can compare it to the OMC head cluster (blue X).
    bio_head = None
    import os as _os, re as _re
    _base = video.replace("__clean3d_refill", ""); _single = _re.sub(r'^(P\d+)_\1_', r'\1_', _base)
    _bm = next((str(F.CACHE / f"biomech_{c}.npz") if hasattr(F, "CACHE") else
                str(HERE.parents[1] / "experiments" / "drink_study" / "cache" / f"biomech_{c}.npz")
                for c in (_base, _single)
                if _os.path.exists(str(HERE.parents[1] / "experiments" / "drink_study" / "cache" / f"biomech_{c}.npz"))), None)
    if _bm:
        _kp = np.load(_bm, allow_pickle=True)["keypoints3d"]
        bio_head = _kp[:, 67, :3].astype(float)
        bio_head[_kp[:, 67, 3] < 0.1] = np.nan          # low-conf head -> NaN (won't draw)
    tr = load_trial(c3d)
    lag = rec["lag"]                                    # KNOWN-GOOD sync from qtm_align.json
    # DEFAULT = fit on the RAW consensus track (matches features.build_rep): the KF-smoothed
    # track overshoots at the drink and rotates the fit up to 17deg. --fused to compare.
    use_raw = not a.fused                      # default = fit on RAW consensus
    fit_track = cup_w0 if a.fused else cup_raw
    DRAW = None
    if a.pscale:
        # DUMB PLAYER: the analysis (emit_draw_points.py) precomputed, PER VIDEO FRAME, the exact
        # 3D points to draw — OMC head/cup already R,t-transformed, biomech head + tracked cup
        # already scaled — all in one scaled-W0 space with ONE frame correspondence. Here we only
        # load + project + draw. The render CANNOT disagree with the numbers: they are these arrays.
        _dp = HERE.parents[1] / "experiments" / "drink_study" / "cache" / "draw_points" / f"{video}.npz"
        if not _dp.exists():
            raise SystemExit(f"--pscale: no draw_points for {video}; run emit_draw_points.py first")
        DRAW = dict(np.load(_dp))
        R = DRAW["R"]; t = DRAW["t"]; rms = float(DRAW["head_gap_med"])
        print(f"{video} c3d={c3d} cam_{cn}  DUMB-PLAYER draw_points  "
              f"head={float(DRAW['head_gap_med']):.1f} cup={float(DRAW['cup_gap_med']):.1f} "
              f"good={float(DRAW['good_frac']):.0%} s_p={float(DRAW['s_p']):.3f}", flush=True)
    if not a.pscale:
        # SAME alignment the features use — one function, so overlay and model never disagree.
        R, t, rms = F.mocap_to_w0(fit_track, tr.centroid(), tr.rate, lag)
        print(f"{video} c3d={c3d} cam_{cn}  fit-on={'RAW cons' if use_raw else 'fused'}  "
              f"Kabsch rms={rms:.1f}mm lag={lag} (sync={rec.get('sync_corr')})", flush=True)
    # comparison fit (cyan X) = the OTHER track's hard-exclude fit, so --compare-fits shows
    # raw-fit (yellow) vs fused-fit (cyan) side by side.
    other = cup_w0 if use_raw else cup_raw
    Rold, told, rms_old = F.mocap_to_w0(other, tr.centroid(), tr.rate, lag, exclude=True)
    cmp_label = "fused-fit" if use_raw else "raw-fit"
    # --session / --velfit: swap in a different rotation. CRITICAL: use the SAME alignment_for()
    # that the metric scoring uses, so the number and the render can NEVER disagree (the old
    # overlay reimplemented the translation with a length-mismatched fallback that skipped the
    # sync -> yellow point off the cup while the metric said 4mm). One code path now.
    scale_s = 1.0
    if a.session or a.velfit or a.scale or a.simrot:
        from session_align import alignment_for
        mode = ("session" if a.session else "velocity" if a.velfit
                else "simrot" if a.simrot else "scale")
        res = alignment_for(video, mode, npz=np.load(_npz[0], allow_pickle=True) if _npz else None, tr=tr)
        if res is not None:
            Rold, told, rms_old = R, t, rms          # per-trial position fit -> the comparison
            cmp_label = "position-fit"
            R, t, info = res
            scale_s = info.get("s", 1.0)
            print(f"  --{mode}: {info}", flush=True)
    if a.compare_fits:
        print(f"  compare: main({'raw' if use_raw else 'fused'}) rms={rms:.1f}mm  vs  "
              f"{cmp_label} rms={rms_old:.1f}mm", flush=True)

    def to_w0(X):
        return scale_s * (X @ R.T) + t
    def to_w0_old(X):
        return X @ Rold.T + told
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

    # --- PER-VIDEO-FRAME HUD numbers (a "big combination of numbers", live in the render) ---
    # Everything on the video-frame grid (len Tv-ish). Computed from the SAME (R,t,scale_s) fit.
    from mocap import resample as _rs
    _mocap_w0 = scale_s * (_rs(tr.centroid(), tr.rate) @ R.T) + t     # OMC cup in W0, 60Hz
    _vid = _rs(cup_fused if cup_fused is not None else cup_w0, VIDEO_FPS)   # MMC cup, 60Hz
    def _grab(arr, i):
        return arr[i] if 0 <= i < len(arr) else np.array([np.nan]*3)
    def hud_at(fr):
        """dict of live numbers for video frame fr."""
        j = fr - lag                                  # mocap-index aligned to this video frame
        vc = _grab(_vid, fr); oc = _grab(_mocap_w0, j)
        dist = float(np.linalg.norm(vc - oc)) if np.isfinite(vc).all() and np.isfinite(oc).all() else np.nan
        # velocities (frame-to-frame), angle between them
        vv = _grab(_vid, fr) - _grab(_vid, fr - 1)
        ov = _grab(_mocap_w0, j) - _grab(_mocap_w0, j - 1)
        sv = float(np.linalg.norm(vv) * VIDEO_FPS); so = float(np.linalg.norm(ov) * VIDEO_FPS)
        ang = np.nan
        if np.isfinite(vv).all() and np.isfinite(ov).all() and sv > 1e-6 and so > 1e-6:
            ang = float(np.degrees(np.arccos(np.clip(np.dot(vv, ov)/(np.linalg.norm(vv)*np.linalg.norm(ov)+1e-9), -1, 1))))
        return dict(dist=dist, ang=ang, sv=sv, so=so)

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
        if DRAW is not None:
            # DUMB PLAYER: everything indexed by video frame fr, no to_w0/mi. These arrays ARE
            # the reported numbers (draw_points/<video>.npz). blue X=OMC head, yellow=OMC cup,
            # white=biomech head, magenta X=tracked cup — all in the same scaled-W0 space.
            def _d(arr, color, **kw):
                if 0 <= fr < len(arr) and np.isfinite(arr[fr]).all():
                    draw(img, arr[fr], color, **kw)
            _d(DRAW["omc_cup"], (60, 220, 235), r=6)                           # OMC cup (yellow)
            _d(DRAW["omc_head"], (60, 60, 235), x=True)                        # OMC head (blue X)
            _d(DRAW["cup_track"], (235, 60, 235), r=7, x=True)                 # tracked cup (magenta X)
            _d(DRAW["bio_head"], (255, 255, 255), r=6)                         # biomech head (white)
        else:
            mi = int(round((fr - lag) * ratio)) if lag >= 0 else int(round(fr * ratio)) - (-lag)
            if 0 <= mi < tr.n_frames:
                for k in range(cupm.shape[1]):
                    draw(img, to_w0(cupm[mi, k]), (80, 220, 80))
                for k in range(headm.shape[1]):
                    draw(img, to_w0(headm[mi, k]), (230, 140, 60))
                draw(img, to_w0(head_c[mi]), (60, 60, 235), x=True)
                draw(img, to_w0(cup_c[mi]), (60, 220, 235), r=6)                # NEW-fit cup (yellow)
                if a.compare_fits:
                    draw(img, to_w0_old(cup_c[mi]), (230, 230, 60), r=6, x=True)  # OLD-fit cup (cyan X)
                d = dist[mi]
                if np.isfinite(d):
                    cv2.putText(img, f"cup->head {d:5.0f} mm", (40, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            # MMC tracked/video cups (already in W0), per-video-frame -> index by fr.
            if 0 <= fr < len(cup_raw):
                draw(img, cup_raw[fr], (235, 60, 235), r=7, x=True)            # MMC RAW (magenta X)
            if cup_fused is not None and 0 <= fr < len(cup_fused):
                draw(img, cup_fused[fr], (60, 170, 255), r=6)                  # MMC FUSED/TCN (orange)
            if bio_head is not None and 0 <= fr < len(bio_head):
                draw(img, bio_head[fr], (255, 255, 255), r=6)                  # MMC head67 (white)
        # legend
        cv2.putText(img, "OMC cup (mocap)", (40, Hh - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (60, 220, 235), 2)
        cv2.putText(img, "MMC raw (X)", (230, Hh - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (235, 60, 235), 2)
        cv2.putText(img, "MMC fused/TCN", (355, Hh - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (60, 170, 255), 2)
        cv2.putText(img, "OMC head (X) / MMC head67 (o)", (40, Hh - 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        if a.compare_fits:
            cv2.putText(img, f"OMC {cmp_label}(X)", (500, Hh - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (230, 230, 60), 2)
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
        # --- LIVE NUMBER HUD (top-right): the "big combination of numbers" for this frame ---
        h = hud_at(fr)
        in_drink = bool(truth_v and truth_v[0] <= fr <= truth_v[1])
        phase = "DRINK" if in_drink else ("move" if (np.isfinite(h["sv"]) and h["sv"] > 80) else "still")
        ang_col = (80, 220, 80) if (np.isfinite(h["ang"]) and h["ang"] < 20) else \
                  ((60, 180, 235) if (np.isfinite(h["ang"]) and h["ang"] < 45) else (60, 60, 235))
        lines = [
            (f"fit: {cmp_label if False else ('session' if a.session else 'sim' if a.simrot else 'scale' if a.scale else 'position')}  s={scale_s:.3f}", (230,230,230)),
            (f"phase: {phase}", (235,200,60) if in_drink else (200,200,200)),
            (f"vel angle: {h['ang']:5.0f} deg" if np.isfinite(h['ang']) else "vel angle:   -- ", ang_col),
            (f"cup-cup:  {h['dist']:5.0f} mm" if np.isfinite(h['dist']) else "cup-cup:   -- ", (200,200,200)),
            (f"vid speed:{h['sv']:5.0f} mm/s", (200,200,200)),
            (f"omc speed:{h['so']:5.0f} mm/s", (200,200,200)),
        ]
        bx = W - 340
        cv2.rectangle(img, (bx - 12, 24), (W - 20, 24 + 26*len(lines) + 10), (30,30,30), -1)
        for li, (txt, col) in enumerate(lines):
            cv2.putText(img, txt, (bx, 50 + li*26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2)
        wr.write(img); fr += 1
        if fr % 60 == 0:
            print(f"  frame {fr}/{Tv}", flush=True)
    cap.release(); wr.release()
    print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
