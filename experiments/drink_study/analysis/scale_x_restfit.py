"""Do per-participant SCALE and REST-ONLY fit STACK? 4-way per-participant comparison,
scored by GOOD-FRAME FRACTION (frames with head+cup residual < GOOD_MM), never median.

  fit on:   ALL frames        vs   REST frames only (rest_pre|rest_post)
  scale:    none              vs   per-participant s_p (divide W0 by s_p about centroid)

Residual measured over ALL frames (rest-only fit is judged on how it predicts the whole rep).
head+cup stacked so scale is NOT absorbable by a rigid fit.
"""
import sys, os, json, numpy as np, re
sys.path.insert(0, os.path.expanduser("~/Documents/object_tracking/experiments/drink_dwell"))
from mocap import load_trial, kabsch, resample as rs, VIDEO_FPS
DS = os.path.expanduser("~/Documents/object_tracking/experiments/drink_study")
CACHE = f"{DS}/cache"; TRACK = f"{CACHE}/track3d_clean3d_refill"
sys.path.insert(0, f"{DS}/lib"); import segment_cup_only as sc
J = 67; STATIC = {"rest_pre", "rest_post"}; GOOD_MM = 15.0


def bcup(v):
    fr = json.load(open(f"{TRACK}/{v}.json"))["frames"]; c = np.full((len(fr), 3), np.nan)
    for i, f in enumerate(fr):
        x = f.get("rts") or f.get("kf")
        if x is not None: c[i] = x
    return c


def syncp(v, m, rate, lag):
    vr = rs(v, VIDEO_FPS); mr = rs(m, rate)
    if lag >= 0: vv = vr[lag:]; mo = mr[:len(vv)]
    else: mo = mr[-lag:]; vv = vr[:len(mo)]
    L = min(len(vv), len(mo)); return vv[:L], mo[:L]


def main():
    sp = json.load(open(f"{CACHE}/per_participant_scale.json"))["scale_per_participant"]
    qa = json.load(open(f"{CACHE}/qtm_align.json"))
    ws = {w["c3d"] for w in json.load(open(f"{CACHE}/wrist_swap_sweep.json"))["wrist_swaps"]}
    # collect per-rep good-frac for each of the 4 conditions
    conds = ["all_noscale", "all_scale", "rest_noscale", "rest_scale"]
    byp = {}
    for pid, ent in qa.items():
        s = sp.get(pid, 1.0)
        for r in ent.get("reps", []):
            if r["c3d"] in ws: continue
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
            seg = sc.segment_cup_only(vc); nm = np.array([sc.PHASE_NAMES[int(p)] for p in seg["phase"]])
            rest = np.isin(nm, list(STATIC))
            if rest.sum() < 10: continue
            for scaled in (False, True):
                V = np.concatenate([vh, vc]).copy(); M = np.concatenate([mh, mc])
                if scaled and abs(s - 1) > 0.03:
                    c0 = np.nanmean(V, 0); V = c0 + (V - c0) / s
                rest2 = np.concatenate([rest, rest])
                for restonly in (False, True):
                    mask = rest2 if restonly else np.ones(len(V), bool)
                    ok = mask & np.isfinite(V).all(1) & np.isfinite(M).all(1)
                    if ok.sum() < 10: continue
                    R, t, _ = kabsch(M[ok], V[ok], robust=True)
                    okall = np.isfinite(V).all(1) & np.isfinite(M).all(1)
                    res = np.linalg.norm(V[okall] - (M[okall] @ R.T + t), axis=1)
                    gf = float((res < GOOD_MM).mean())
                    key = f"{'rest' if restonly else 'all'}_{'scale' if scaled else 'noscale'}"
                    byp.setdefault(pid, {c: [] for c in conds})[key].append(gf)

    print(f"good-frame % (head+cup resid < {GOOD_MM}mm), mean over reps; wrist-swaps excluded\n")
    print(f"{'pid':<5}{'s_p':>6}  {'all/no':>8}{'all/sc':>8}{'rest/no':>9}{'rest/sc':>9}   best")
    focus = []
    for pid in sorted(byp):
        s = sp.get(pid, 1.0); d = byp[pid]
        m = {c: (np.mean(d[c]) if d[c] else np.nan) for c in conds}
        best = max((c for c in conds if not np.isnan(m[c])), key=lambda c: m[c])
        star = " *" if abs(s - 1) > 0.03 else ""
        print(f"{pid:<5}{s:>6.3f}  {m['all_noscale']:>7.0%}{m['all_scale']:>8.0%}"
              f"{m['rest_noscale']:>9.0%}{m['rest_scale']:>9.0%}   {best}{star}")
        if abs(s - 1) > 0.03: focus.append(pid)
    print("\n* = scale-compressed participant (|s_p-1|>3%)")
    # aggregate over the scale-compressed group
    if focus:
        agg = {c: [] for c in conds}
        for pid in focus:
            for c in conds: agg[c] += byp[pid][c]
        print(f"\nSCALE-COMPRESSED GROUP {focus}:")
        for c in conds:
            print(f"  {c:<14}{np.mean(agg[c]):>6.1%}")


if __name__ == "__main__":
    main()
