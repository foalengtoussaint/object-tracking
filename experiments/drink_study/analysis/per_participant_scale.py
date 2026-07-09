"""Per-participant SCALE correction for the biomech->OMC alignment.

Diagnosis (2026-07-09): the 'hard 14' reps are dominated by a per-participant ~5%
scale compression (P24 0.952, P16 0.949) in the markerless W0 frame — NOT rotation,
NOT tracking. A per-rep global Umeyama scale HURT because it scaled the good reps too.

Fix here: fit ONE scale factor s_p per participant (pooled over all that participant's
head+cup frames), apply s_p to the biomech W0 points, then re-run the SAME plain robust
Kabsch (no exclude) and re-measure. Compare no-scale vs per-participant-scale.

s_p is estimated the scale-only, rotation/translation-invariant way: pool every
head+cup residual pair (V_i - V_j) vs (RM_i - RM_j) chord lengths across the
participant's reps and take the ratio that minimizes least-squares =
 s_p = <||dM||^2>^-1-weighted ... we use the closed form from the pooled cross term.
Actually simplest robust estimator: s_p = median over all within-rep frame-pairs of
||OMC pair|| / ||MMC pair||  (both invariant to R,t). Chord ratios can't be faked by fit.
"""
import sys, os, json, glob, re
import numpy as np

DD = os.path.expanduser("~/Documents/object_tracking/experiments/drink_dwell")
sys.path.insert(0, DD)
from mocap import load_trial, kabsch, resample as resample3d, VIDEO_FPS  # noqa

DS = os.path.expanduser("~/Documents/object_tracking/experiments/drink_study")
CACHE = f"{DS}/cache"
TRACK = f"{CACHE}/track3d_clean3d_refill"
J_HEAD = 67
CUP_FIELD = "rts"
GOOD_MM = 15.0     # good-frame threshold (per feedback: report FRACTION below threshold)


def single_p(stem):
    return re.sub(r'^(P\d+)_\1_', r'\1_', stem)


def biomech_path(vstem):
    base = vstem.replace("__clean3d_refill", "")
    for c in (base, single_p(base)):
        p = f"{CACHE}/biomech_{c}.npz"
        if os.path.exists(p):
            return p
    return None


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


def sync_pair(v, m, rate, lag):
    vr = resample3d(v, VIDEO_FPS); mr = resample3d(m, rate)
    if lag >= 0:
        vv = vr[lag:]; mo = mr[:len(vv)]
    else:
        mo = mr[-lag:]; vv = vr[:len(mo)]
    L = min(len(vv), len(mo))
    return vv[:L], mo[:L]


def load_rep(vstem, c3d, lag):
    """Return synced, cleaned (V,M,tag) stacked head+cup for one rep, or None."""
    bmp = biomech_path(vstem)
    cup = biomech_cup(vstem)
    if bmp is None or cup is None:
        return None
    b = np.load(bmp, allow_pickle=True)["keypoints3d"]
    head = b[:, J_HEAD, :3].astype(float)
    head[b[:, J_HEAD, 3] < 0.1] = np.nan
    try:
        t3 = load_trial(c3d)
    except Exception:
        return None
    oc = t3.centroid()
    oh = t3.head_centroid() if t3.has_head() else None
    head_valid = np.isfinite(oh).all(1).mean() if oh is not None else 0.0
    use_head = head_valid >= 0.90
    if not use_head:
        head = np.full_like(head, np.nan)
        oh = oc
    vh, mh = sync_pair(head, oh if oh is not None else oc, t3.rate, lag)
    vc, mc = sync_pair(cup, oc, t3.rate, lag)
    L = min(len(vh), len(vc))
    vh, mh, vc, mc = vh[:L], mh[:L], vc[:L], mc[:L]
    V = np.concatenate([vh, vc]); M = np.concatenate([mh, mc])
    tag = np.concatenate([np.zeros(L, int), np.ones(L, int)])
    ok = ~(np.isnan(V).any(1) | np.isnan(M).any(1))
    if ok.sum() < 10:
        return None
    return V[ok], M[ok], tag[ok], bool(use_head)


def chord_scale(V, M, npair=4000):
    """Fit-independent scale = median ||V pair|| / ||M pair|| over random within-rep frame pairs.
    Invariant to R,t (only depends on inter-point distances). >1 means MMC longer than OMC."""
    n = len(V)
    if n < 3:
        return None
    rng = np.random.default_rng(0)
    i = rng.integers(0, n, npair); j = rng.integers(0, n, npair)
    ok = i != j
    dv = np.linalg.norm(V[i[ok]] - V[j[ok]], axis=1)
    dm = np.linalg.norm(M[i[ok]] - M[j[ok]], axis=1)
    m = (dm > 20) & (dv > 20)          # ignore near-coincident pairs
    if m.sum() < 50:
        return None
    return float(np.median(dv[m] / dm[m]))    # MMC / OMC


def resid_stats(V, M):
    R, t, _ = kabsch(M, V, robust=True)
    r = np.linalg.norm(V - (M @ R.T + t), axis=1)
    return r


def main():
    qa = json.load(open(f"{CACHE}/qtm_align.json"))
    reps = [(p, r) for p, ent in qa.items() for r in ent.get("reps", [])]
    print(f"qtm_align reps: {len(reps)}", flush=True)

    # Wrist-swap reps: their cup track follows the WRIST, so their chord ratio measures a
    # BROKEN track, not scale — they must NOT contribute to the per-participant s_p estimate
    # (they poison P13's s_p and add noise to P19/P08). Still rendered/measured, just excluded
    # from ESTIMATION.
    ws = json.load(open(f"{CACHE}/wrist_swap_sweep.json"))["wrist_swaps"]
    WRIST_SWAP = {w["c3d"] for w in ws}
    print(f"excluding {len(WRIST_SWAP)} wrist-swap reps from scale estimation: "
          f"{sorted(WRIST_SWAP)}", flush=True)

    # ---- pass 1: load every rep, estimate per-rep chord scale (MMC/OMC) ----
    loaded = {}   # vstem -> (V,M,tag,use_head, pid)
    rep_scale = {}
    for pid, r in reps:
        d = load_rep(r["video"], r["c3d"], r["lag"])
        if d is None:
            continue
        V, M, tag, uh = d
        loaded[r["video"]] = (V, M, tag, uh, pid, r["c3d"])
        s = chord_scale(V, M)
        if s is not None and r["c3d"] not in WRIST_SWAP:   # exclude broken tracks from s_p
            rep_scale.setdefault(pid, []).append(s)
    print(f"loaded {len(loaded)} reps across {len(rep_scale)} participants", flush=True)

    # ---- per-participant scale factor s_p (median of rep chord scales) ----
    # s_p = MMC/OMC length ratio. To CORRECT MMC onto OMC we divide MMC by s_p.
    s_p = {p: float(np.median(v)) for p, v in rep_scale.items()}

    # ---- pass 2: measure residuals with and without the per-participant correction ----
    # correction: V' = V / s_p (bring MMC to OMC scale). Kabsch about the head+cup centroid,
    # so scale must be applied about that centroid to be a pure size change.
    rows = []
    for vstem, (V, M, tag, uh, pid, c3d) in loaded.items():
        s = s_p.get(pid, 1.0)
        # no-scale
        r0 = resid_stats(V, M)
        # per-participant scale: shrink MMC about its own centroid by 1/s
        c0 = V.mean(0)
        Vs = c0 + (V - c0) / s
        r1 = resid_stats(Vs, M)
        rows.append({
            "video": vstem, "pid": pid, "c3d": c3d, "s_p": s,
            "rms0": float(np.sqrt((r0 ** 2).mean())),
            "rms1": float(np.sqrt((r1 ** 2).mean())),
            "good0": float((r0 < GOOD_MM).mean()),
            "good1": float((r1 < GOOD_MM).mean()),
            "n": int(len(V)),
        })

    # ---- report ----
    print(f"\n{'pid':<5}{'n_rep':>6}{'s_p':>7}{'%off':>7}   "
          f"{'rms0':>7}{'rms1':>7}   {'good0':>7}{'good1':>7}")
    by_p = {}
    for row in rows:
        by_p.setdefault(row["pid"], []).append(row)
    for pid in sorted(by_p):
        rr = by_p[pid]
        s = s_p.get(pid, 1.0)
        rms0 = np.median([x["rms0"] for x in rr]); rms1 = np.median([x["rms1"] for x in rr])
        g0 = np.mean([x["good0"] for x in rr]); g1 = np.mean([x["good1"] for x in rr])
        flag = "  <== corrected" if abs(s - 1) > 0.03 else ""
        print(f"{pid:<5}{len(rr):>6}{s:>7.3f}{(s-1)*100:>+6.1f}%   "
              f"{rms0:>7.1f}{rms1:>7.1f}   {g0:>6.1%} {g1:>6.1%}{flag}")

    print(f"\n{'OVERALL':<5}")
    all0 = np.array([x["rms0"] for x in rows]); all1 = np.array([x["rms1"] for x in rows])
    g0 = np.array([x["good0"] for x in rows]); g1 = np.array([x["good1"] for x in rows])
    print(f"  median rms   : {np.median(all0):.1f} -> {np.median(all1):.1f} mm")
    print(f"  mean good%   : {g0.mean():.1%} -> {g1.mean():.1%}")

    # focus on the corrected participants
    corr = [p for p in s_p if abs(s_p[p] - 1) > 0.03]
    if corr:
        cr = [x for x in rows if x["pid"] in corr]
        c0 = np.array([x["rms0"] for x in cr]); c1 = np.array([x["rms1"] for x in cr])
        cg0 = np.array([x["good0"] for x in cr]); cg1 = np.array([x["good1"] for x in cr])
        print(f"\nCORRECTED participants {sorted(corr)} ({len(cr)} reps):")
        print(f"  median rms   : {np.median(c0):.1f} -> {np.median(c1):.1f} mm")
        print(f"  mean good%   : {cg0.mean():.1%} -> {cg1.mean():.1%}")

    out = {"scale_per_participant": s_p, "good_mm": GOOD_MM,
           "note": "s_p = median MMC/OMC chord-length ratio; correction divides MMC by s_p "
                   "about its centroid. rms0/good0 = no scale, rms1/good1 = per-participant scale.",
           "reps": rows}
    json.dump(out, open(f"{CACHE}/per_participant_scale.json", "w"), indent=1)
    print(f"\nwrote {CACHE}/per_participant_scale.json")


if __name__ == "__main__":
    main()
