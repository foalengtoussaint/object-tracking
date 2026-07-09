"""Compute the head+cup joint Kabsch (no exclude) per rep and SAVE R,t + the residuals it
produces -> cache/align_head_cup_fits.json. The render (overlay.py --pscale) LOADS this exact
R,t and projects with it, so the video shows the SAME fit these numbers describe. One source
of truth: the R,t written here is the R,t drawn on screen.

Fit space: OMC (mocap frame) -> W0. We fit M(=OMC head+cup) -> V(=biomech head + tracked cup),
so R,t maps OMC into W0 exactly like overlay's to_w0(X) = X@R.T + t. This is what places the
blue OMC-head and yellow OMC-cup markers. The white biomech head is drawn in raw W0 (unchanged);
after this fit the blue OMC head should sit on/near the white biomech head = the reported head_med.
"""
import sys, os, json, numpy as np, re
sys.path.insert(0, os.path.expanduser("~/Documents/object_tracking/experiments/drink_dwell"))
from mocap import load_trial, kabsch, resample as rs, VIDEO_FPS
DS = os.path.expanduser("~/Documents/object_tracking/experiments/drink_study")
CACHE = f"{DS}/cache"; TRACK = f"{CACHE}/track3d_clean3d_refill"; J = 67; GOOD_MM = 15.0


def bcup(v):
    fr = json.load(open(f"{TRACK}/{v}.json"))["frames"]; c = np.full((len(fr), 3), np.nan)
    for i, f in enumerate(fr):
        x = f.get("rts") or f.get("kf")
        if x is not None: c[i] = x
    return c


def syncp(a, m, rate, lag):
    vr = rs(a, VIDEO_FPS); mr = rs(m, rate)
    if lag >= 0: vv = vr[lag:]; mo = mr[:len(vv)]
    else: mo = mr[-lag:]; vv = vr[:len(mo)]
    L = min(len(vv), len(mo)); return vv[:L], mo[:L]


def main():
    sp = json.load(open(f"{CACHE}/per_participant_scale.json"))["scale_per_participant"]
    qa = json.load(open(f"{CACHE}/qtm_align.json"))
    ws = {w["c3d"] for w in json.load(open(f"{CACHE}/wrist_swap_sweep.json"))["wrist_swaps"]}
    out = {}
    n = 0
    for pid, ent in qa.items():
        for r in ent.get("reps", []):
            v = r["video"]
            if not os.path.exists(f"{TRACK}/{v}.json"): continue
            base = v.replace("__clean3d_refill", ""); sing = re.sub(r'^(P\d+)_\1_', r'\1_', base)
            bp = next((f"{CACHE}/biomech_{c}.npz" for c in (base, sing)
                       if os.path.exists(f"{CACHE}/biomech_{c}.npz")), None)
            if not bp: continue
            try: t3 = load_trial(r["c3d"])
            except Exception: continue
            kp = np.load(bp, allow_pickle=True)["keypoints3d"]
            bh = kp[:, J, :3].astype(float); bh[kp[:, J, 3] < 0.1] = np.nan
            vh, mh = syncp(bh, t3.head_centroid(), t3.rate, r["lag"])
            vc, mc = syncp(bcup(v), t3.centroid(), t3.rate, r["lag"])
            L = min(len(vh), len(vc)); vh, mh, vc, mc = vh[:L], mh[:L], vc[:L], mc[:L]
            V = np.concatenate([vh, vc]); M = np.concatenate([mh, mc])
            tag = np.concatenate([np.zeros(L, int), np.ones(L, int)])
            # APPLY per-participant scale to the W0/video side (this is the fix; head+cup fit
            # without it is ~20mm bad). Un-compress V by 1/s_p about its centroid, THEN fit.
            s = float(sp.get(pid, 1.0))
            scale_c = None
            if abs(s - 1) > 0.03:
                fin = np.isfinite(V).all(1)
                scale_c = V[fin].mean(0)
                V = scale_c + (V - scale_c) / s
            ok = np.isfinite(V).all(1) & np.isfinite(M).all(1)
            if ok.sum() < 10: continue
            R, t, _ = kabsch(M[ok], V[ok], robust=True)          # OMC -> W0(scaled), no exclude
            res = np.linalg.norm(V[ok] - (M[ok] @ R.T + t), axis=1)
            tg = tag[ok]
            out[v] = {
                "R": R.tolist(), "t": t.tolist(),
                "rms_mm": float(np.sqrt((res ** 2).mean())),
                "head_med": float(np.median(res[tg == 0])) if (tg == 0).any() else None,
                "cup_med": float(np.median(res[tg == 1])) if (tg == 1).any() else None,
                "good_frac": float((res < GOOD_MM).mean()),
                "s_p": s,
                "scale_center": scale_c.tolist() if scale_c is not None else None,
                "c3d": r["c3d"], "lag": r["lag"], "wrist_swap": r["c3d"] in ws,
            }
            n += 1
    json.dump({"note": "head+cup joint Kabsch (OMC->W0), no exclude. R,t are what overlay.py "
                       "--pscale loads and draws. head_med/cup_med/good_frac are FROM this R,t.",
               "good_mm": GOOD_MM, "reps": out},
              open(f"{CACHE}/align_head_cup_fits.json", "w"), indent=1)
    print(f"wrote {CACHE}/align_head_cup_fits.json  ({n} reps)")
    # sanity: print the reps we're about to render
    for v in out:
        if any(k in v for k in ("P24_drinking_left_20240724_110041", "P16_drinking_left_20240306_105920")):
            f = out[v]
            print(f"  {v[:40]:<40} head={f['head_med']:.1f} cup={f['cup_med']:.1f} "
                  f"good={f['good_frac']:.0%} s_p={f['s_p']:.3f}")


if __name__ == "__main__":
    main()
