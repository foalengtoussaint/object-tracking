"""Is the cup alignment residual concentrated in the DRINKING phase?

For each aligned rep: recompute the PER-FRAME cup residual (biomech cup vs OMC cup,
OMC mapped into W0 via the cached per-rep R,t, synced by lag), segment the biomech cup
track into van-Andel phases, and compare cup residual DRINKING vs NON-DRINKING.
"""
import sys, os, json
import numpy as np

DD = os.path.expanduser("~/Documents/object_tracking/experiments/drink_dwell")
sys.path.insert(0, DD)
from mocap import load_trial, resample as resample3d, VIDEO_FPS  # noqa
DS = os.path.expanduser("~/Documents/object_tracking/experiments/drink_study")
sys.path.insert(0, f"{DS}/lib")
import segment_cup_only as sc  # noqa
CACHE = f"{DS}/cache"; TRACK = f"{CACHE}/track3d_clean3d_refill"
CUP_FIELD = "rts"


def biomech_cup(vstem):
    p = f"{TRACK}/{vstem}.json"
    if not os.path.exists(p):
        return None
    fr = json.load(open(p))["frames"]
    cup = np.full((len(fr), 3), np.nan)
    for i, f in enumerate(fr):
        xyz = f.get(CUP_FIELD) or f.get("consensus") or f.get("kf")
        if xyz is not None:
            cup[i] = xyz
    return cup


def sync_pair(v_arr, m_arr, rate, lag):
    vr = resample3d(v_arr, VIDEO_FPS); mr = resample3d(m_arr, rate)
    if lag >= 0:
        v = vr[lag:]; mo = mr[:len(v)]
    else:
        mo = mr[-lag:]; v = vr[:len(mo)]
    L = min(len(v), len(mo))
    return v[:L], mo[:L]


def main():
    al = json.load(open(f"{CACHE}/align_head_cup.json"))
    reps = al["reps"]
    drink_res, other_res = [], []      # pooled per-frame residuals
    per_rep = []                       # (drink_med, other_med, ratio)
    for vstem, f in reps.items():
        cup = biomech_cup(vstem)
        if cup is None:
            continue
        try:
            t3 = load_trial(f["c3d"])
        except Exception:
            continue
        vc, mc = sync_pair(cup, t3.centroid(), t3.rate, f["lag"])
        R = np.array(f["R"]); t = np.array(f["t"])
        mc_w0 = mc @ R.T + t                       # OMC cup in W0
        resid = np.linalg.norm(vc - mc_w0, axis=1)  # (L,) per-frame cup residual

        # segment the biomech cup (in W0) into phases at the SAME synced sampling
        seg = sc.segment_cup_only(vc)
        phase = seg["phase"]                        # (L,) int, 2 = drinking
        drink = phase == sc.P_DRINK if hasattr(sc, "P_DRINK") else \
            np.array([sc.PHASE_NAMES[int(p)] == "drinking" for p in phase])

        ok = np.isfinite(resid)
        d = resid[ok & drink]; o = resid[ok & ~drink]
        if d.size < 3 or o.size < 3:
            continue
        drink_res.append(d); other_res.append(o)
        per_rep.append((vstem.split("_")[0], np.median(d), np.median(o)))

    D = np.concatenate(drink_res); O = np.concatenate(other_res)
    print(f"reps with both phases: {len(per_rep)}  frames: drink={D.size} other={O.size}")
    print(f"\nPOOLED per-frame cup residual (mm):")
    print(f"  DRINKING   : med {np.median(D):.1f} | mean {D.mean():.1f} | p90 {np.percentile(D,90):.1f}")
    print(f"  NON-drink  : med {np.median(O):.1f} | mean {O.mean():.1f} | p90 {np.percentile(O,90):.1f}")
    print(f"  ratio (drink med / non med): {np.median(D)/np.median(O):.2f}x")

    # per-rep: how consistently is drinking worse?
    ratios = np.array([dm / om for _, dm, om in per_rep if om > 0])
    worse = np.mean([dm > om for _, dm, om in per_rep])
    print(f"\nPER-REP: drinking-worse in {worse*100:.0f}% of reps; "
          f"median per-rep ratio {np.median(ratios):.2f}x")
    # a few worst offenders
    per_rep.sort(key=lambda x: -(x[1] - x[2]))
    print("\nreps where drinking most exceeds rest (pid, drink_med, other_med):")
    for pid, dm, om in per_rep[:8]:
        print(f"  {pid:<5} drink {dm:5.1f}  other {om:5.1f}  (+{dm-om:.1f})")


if __name__ == "__main__":
    main()
