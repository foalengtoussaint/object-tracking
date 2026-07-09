"""Fit the mocap->W0 transform on REST-phase frames ONLY, then measure how it predicts
the whole rep (rest = in-sample, moving+drink = out-of-sample). Compare vs all-frames fit.

Both fits are plain robust Kabsch (NO exclude). Rest = rest_pre|rest_post from the cup
segmentation. We report head & cup residual split by rest / moving / drink and in/out-of-sample.
"""
import sys, os, json, numpy as np, re
sys.path.insert(0, os.path.expanduser("~/Documents/object_tracking/experiments/drink_dwell"))
from mocap import load_trial, resample as resample3d, VIDEO_FPS, kabsch
DS = os.path.expanduser("~/Documents/object_tracking/experiments/drink_study")
sys.path.insert(0, f"{DS}/lib"); import segment_cup_only as sc
CACHE = f"{DS}/cache"; TRACK = f"{CACHE}/track3d_clean3d_refill"; J_HEAD = 67
STATIC = {"rest_pre", "rest_post"}


def bcup(v):
    fr = json.load(open(f"{TRACK}/{v}.json"))["frames"]; c = np.full((len(fr), 3), np.nan)
    for i, f in enumerate(fr):
        x = f.get("rts") or f.get("kf")
        if x is not None: c[i] = x
    return c


def sync(v_, m_, rate, lag):
    vr = resample3d(v_, VIDEO_FPS); mr = resample3d(m_, rate)
    if lag >= 0: v = vr[lag:]; mo = mr[:len(v)]
    else: mo = mr[-lag:]; v = vr[:len(mo)]
    L = min(len(v), len(mo)); return v[:L], mo[:L]


def fit(V, M, mask):
    ok = mask & np.isfinite(V).all(1) & np.isfinite(M).all(1)
    if ok.sum() < 10: return None
    R, t, _ = kabsch(M[ok], V[ok], robust=True)
    return R, t


def med(a):
    a = a[np.isfinite(a)]; return np.median(a) if a.size else np.nan


def main():
    al = json.load(open(f"{CACHE}/align_head_cup.json"))["reps"]
    # collectors: [restfit|allfit][channel][phasegroup]
    rows = {ff: {ch: {ph: [] for ph in ("rest", "move", "drink")}
                 for ch in ("head", "cup")} for ff in ("restfit", "allfit")}
    nrep = 0
    for v, f in al.items():
        if not os.path.exists(f"{TRACK}/{v}.json"): continue
        base = v.replace("__clean3d_refill", ""); single = re.sub(r'^(P\d+)_\1_', r'\1_', base)
        bp = next((f"{CACHE}/biomech_{c}.npz" for c in (base, single)
                   if os.path.exists(f"{CACHE}/biomech_{c}.npz")), None)
        if bp is None: continue
        try: t3 = load_trial(f["c3d"])
        except Exception: continue
        kp = np.load(bp, allow_pickle=True)["keypoints3d"]
        bh = kp[:, J_HEAD, :3].astype(float); bh[kp[:, J_HEAD, 3] < 0.1] = np.nan
        vh, mh = sync(bh, t3.head_centroid(), t3.rate, f["lag"])
        vc, mc = sync(bcup(v), t3.centroid(), t3.rate, f["lag"])
        L = min(len(vh), len(vc)); vh, mh, vc, mc = vh[:L], mh[:L], vc[:L], mc[:L]

        seg = sc.segment_cup_only(vc); names = np.array([sc.PHASE_NAMES[int(p)] for p in seg["phase"]])
        rest = np.isin(names, list(STATIC)); drink = names == "drinking"; move = ~rest & ~drink
        if rest.sum() < 10: continue

        # stack head+cup with a phase mask duplicated for both channels
        V = np.concatenate([vh, vc]); M = np.concatenate([mh, mc])
        rest2 = np.concatenate([rest, rest])
        for ff, mask in (("restfit", rest2), ("allfit", np.ones(len(V), bool))):
            r = fit(V, M, mask)
            if r is None: continue
            R, t = r
            resid = np.linalg.norm(V - (M @ R.T + t), axis=1)
            hn = len(vh)
            rh, rc = resid[:hn], resid[hn:]
            for ch, rr in (("head", rh), ("cup", rc)):
                rows[ff][ch]["rest"].append(med(rr[rest]))
                rows[ff][ch]["move"].append(med(rr[move]) if move.sum() else np.nan)
                rows[ff][ch]["drink"].append(med(rr[drink]) if drink.sum() else np.nan)
        nrep += 1

    print(f"reps: {nrep}\n")
    print(f"{'fit':<9}{'chan':<6}{'rest(in)':>10}{'move(out)':>11}{'drink(out)':>12}")
    for ff in ("restfit", "allfit"):
        for ch in ("head", "cup"):
            r = rows[ff][ch]
            print(f"{ff:<9}{ch:<6}"
                  f"{med(np.array(r['rest'])):>10.1f}"
                  f"{med(np.array(r['move'])):>11.1f}"
                  f"{med(np.array(r['drink'])):>12.1f}")
    print("\n(median over reps of each rep's per-phase median residual, mm)")
    print("rest = frames the rest-only fit trained on; move/drink = OUT-of-sample for restfit")


if __name__ == "__main__":
    main()
