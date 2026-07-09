"""Precompute, PER VIDEO FRAME, the exact 3D points overlay.py should draw — all in one
scaled-W0 space, using ONE frame correspondence. The render becomes a dumb player: load this
npz, project each point, draw. It cannot disagree with the numbers because the numbers are
computed from these same arrays.

Output per rep: cache/draw_points/<video>.npz with (T_video, 3) arrays:
    omc_head, omc_cup            (centroids, R,t-transformed -> blue X / yellow)
    bio_head                     (biomech head67, scaled -> white)
    cup_track                    (tracked cup rts, scaled -> magenta)
  + head_gap_med / cup_gap_med / good_frac  (the drawn-point residuals; MUST match report)

Correspondence: for video frame fr, the OMC sample is the resample+lag SYNC index (same as the
fit). We build OMC arrays ON THE VIDEO-FRAME GRID by resampling omc->VIDEO_FPS then applying lag,
so omc[fr] pairs with bio/cup[fr] EXACTLY as the fit paired them. head_gap here == fit head_med.
"""
import sys, os, json, numpy as np, re
sys.path.insert(0, os.path.expanduser("~/Documents/object_tracking/experiments/drink_dwell"))
from mocap import load_trial, resample as rs, VIDEO_FPS
DS = os.path.expanduser("~/Documents/object_tracking/experiments/drink_study")
CACHE = f"{DS}/cache"; TRACK = f"{CACHE}/track3d_clean3d_refill"; J = 67; GOOD_MM = 15.0
OUT = f"{CACHE}/draw_points"; os.makedirs(OUT, exist_ok=True)


def bcup(v):
    fr = json.load(open(f"{TRACK}/{v}.json"))["frames"]; c = np.full((len(fr), 3), np.nan)
    for i, f in enumerate(fr):
        x = f.get("rts") or f.get("kf")
        if x is not None: c[i] = x
    return c


def on_video_grid(omc_arr, rate, lag, T):
    """Resample OMC (rate Hz) -> VIDEO_FPS grid and apply lag so index fr pairs with video fr.
    Returns (T,3) padded with NaN. Mirrors syncp: vr=resample(video), mr=resample(omc); lag aligns."""
    mr = rs(omc_arr, rate)                       # omc on the common (VIDEO_FPS) grid
    out = np.full((T, omc_arr.shape[1]) if omc_arr.ndim == 2 else (T, 3), np.nan)
    for fr in range(T):
        j = fr - lag if lag >= 0 else fr + (-lag)   # syncp: video[lag:] vs mr[:] (lag>=0)
        # syncp lag>=0: v=vr[lag:], mo=mr[:len]; so video frame fr(>=lag) <-> mr[fr-lag]
        # lag<0:        mo=mr[-lag:], v=vr[:len]; so video fr <-> mr[fr-lag] = mr[fr+|lag|]
        if 0 <= j < len(mr):
            out[fr] = mr[j]
    return out


def main():
    fits = json.load(open(f"{CACHE}/align_head_cup_fits.json"))["reps"]
    n = 0
    for video, f in fits.items():
        if not os.path.exists(f"{TRACK}/{video}.json"): continue
        R = np.array(f["R"]); t = np.array(f["t"]); sp = f["s_p"]
        cc = np.array(f["scale_center"]) if f.get("scale_center") is not None else None
        lag = f["lag"]
        try: tr = load_trial(f["c3d"])
        except Exception: continue
        base = video.replace("__clean3d_refill", ""); sing = re.sub(r'^(P\d+)_\1_', r'\1_', base)
        bm = next((f"{CACHE}/biomech_{c}.npz" for c in (base, sing)
                   if os.path.exists(f"{CACHE}/biomech_{c}.npz")), None)
        if not bm: continue
        kp = np.load(bm, allow_pickle=True)["keypoints3d"]
        bio = kp[:, J, :3].astype(float); bio[kp[:, J, 3] < 0.1] = np.nan
        cup = bcup(video)
        T = min(len(bio), len(cup))
        bio, cup = bio[:T], cup[:T]
        # scale the W0/video side about the SAVED centroid (bit-identical to the fit)
        if cc is not None:
            bio = cc + (bio - cc) / sp
            cup = cc + (cup - cc) / sp
        # OMC on the video-frame grid, then R,t -> scaled-W0
        omc_head_g = on_video_grid(tr.head_centroid(), tr.rate, lag, T)
        omc_cup_g = on_video_grid(tr.centroid(), tr.rate, lag, T)
        omc_head_w0 = omc_head_g @ R.T + t
        omc_cup_w0 = omc_cup_g @ R.T + t

        # residuals from the DRAWN arrays (this is what the video shows)
        hg = np.linalg.norm(bio - omc_head_w0, axis=1); hg = hg[np.isfinite(hg)]
        cg = np.linalg.norm(cup - omc_cup_w0, axis=1); cg = cg[np.isfinite(cg)]
        allr = np.concatenate([hg, cg]) if hg.size and cg.size else np.array([])
        np.savez(f"{OUT}/{video}.npz",
                 omc_head=omc_head_w0, omc_cup=omc_cup_w0, bio_head=bio, cup_track=cup,
                 head_gap_med=np.median(hg) if hg.size else np.nan,
                 cup_gap_med=np.median(cg) if cg.size else np.nan,
                 good_frac=(allr < GOOD_MM).mean() if allr.size else np.nan,
                 R=R, t=t, s_p=sp)
        n += 1
        if any(k in video for k in ("P16_drinking_left_20240306_105920",
                                    "P24_drinking_left_20240724_110041")):
            print(f"  {video[:42]:<42} DRAWN head={np.median(hg):.1f} cup={np.median(cg):.1f} "
                  f"good={(allr<GOOD_MM).mean():.0%}  (fit head_med={f['head_med']:.1f})", flush=True)
    print(f"wrote {n} -> {OUT}/")


if __name__ == "__main__":
    main()
