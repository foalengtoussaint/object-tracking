"""Align biomech (markerless, W0) to OMC/QTM (mocap frame) using BOTH head + cup.

Per-rep rigid mocap->W0 fit (Kabsch) on the COMBINED head+cup point cloud, synced by
the known-good lag from qtm_align.json (same sync convention as features.mocap_to_w0).
Head+cup together constrains the rotation that the symmetric cup-alone fit left ambiguous.

Biomech side (W0, mm):
  - head  = keypoints3d[:, 67, :3]           (biomech_<stem>.npz)
  - cup   = track3d_clean3d_refill[...]['rts']  (smoothed consensus cup track)
OMC side (mocap frame, mm), from mocap.load_trial(c3d):
  - head  = head_centroid()   (5-marker robust head cluster)
  - cup   = centroid()        (4 cup markers)

Output: cache/align_head_cup.json — per-rep (R,t,combined rms, head resid, cup resid,
n frames used) + a per-participant summary. Caches the fit; does not rewrite trajectories.
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
EXCLUDE_MM = 20.0          # combined head+cup: a touch looser than cup-only 15mm
CUP_FIELD = "rts"          # smoothed cup track

# --- OMC quality gates (some QTM C3Ds are corrupt: frozen cup, dead/gappy head cluster,
#     broken sync). We MEASURE each C3D and gate it rather than trusting every file. ---
MIN_CUP_TRAVEL_MM = 20.0   # frozen-cup C3Ds have <20mm total cup travel (known corruption)
MIN_HEAD_VALID_FRAC = 0.90 # need ≥90% valid head-cluster frames to trust the head channel
MIN_SYNC_CORR = 0.90       # qtm_align sync correlation below this = unreliable pairing


def bbox_travel(pts):
    g = np.isfinite(pts).all(1)
    if g.sum() < 5:
        return 0.0
    p = pts[g]
    return float(np.linalg.norm(p.max(0) - p.min(0)))


def single_p(stem):
    return re.sub(r'^(P\d+)_\1_', r'\1_', stem)


def biomech_path(video_stem):
    """video_stem like 'P02_P02_drinking_left_...__clean3d_refill' -> biomech npz path."""
    base = video_stem.replace("__clean3d_refill", "")
    for cand in (base, single_p(base)):
        p = f"{CACHE}/biomech_{cand}.npz"
        if os.path.exists(p):
            return p
    return None


def biomech_cup(video_stem):
    p = f"{TRACK}/{video_stem}.json"
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
    """Resample video->VIDEO_FPS and mocap->rate to COMMON_HZ, apply lag; return (v,mo) same len."""
    vr = resample3d(v_arr, VIDEO_FPS)
    mr = resample3d(m_arr, rate)
    if lag >= 0:
        v = vr[lag:]; mo = mr[:len(v)]
    else:
        mo = mr[-lag:]; v = vr[:len(mo)]
    L = min(len(v), len(mo))
    return v[:L], mo[:L]


def fit_head_cup(bm_head, bm_cup, omc_head, omc_cup, rate, lag, exclude=False):
    """Kabsch mocap->W0 on stacked head+cup. Returns dict or None.

    exclude=False (DEFAULT, user choice 2026-07-09): plain robust(Huber) Kabsch on ALL
    head+cup frames, NO hard 20mm drop, NO readmit — the honest whole-rep fit including any
    constant cup offset. residuals reported over ALL frames.
    exclude=True: legacy hard-exclude+readmit at EXCLUDE_MM (drops ~half the cup frames on
    offset reps → flattering rms). Kept for comparison only.
    """
    vh, mh = sync_pair(bm_head, omc_head, rate, lag)
    vc, mc = sync_pair(bm_cup, omc_cup, rate, lag)
    L = min(len(vh), len(vc))
    vh, mh, vc, mc = vh[:L], mh[:L], vc[:L], mc[:L]

    # tag: 0=head, 1=cup so we can split residuals afterward
    V = np.concatenate([vh, vc], 0)
    M = np.concatenate([mh, mc], 0)
    tag = np.concatenate([np.zeros(L, int), np.ones(L, int)])
    ok = ~(np.isnan(V).any(1) | np.isnan(M).any(1))
    if ok.sum() < 10:
        return None
    V, M, tag = V[ok], M[ok], tag[ok]

    R, t, rms = kabsch(M, V, robust=True)
    keep = np.ones(len(M), bool)
    if exclude:
        for _ in range(6):
            r = np.linalg.norm(V - (M @ R.T + t), axis=1)
            nk = r < EXCLUDE_MM
            if nk.sum() < 10 or np.array_equal(nk, keep):
                keep = nk if nk.sum() >= 10 else keep
                break
            keep = nk
            R, t, _ = kabsch(M[keep], V[keep], robust=False)

    resid = np.linalg.norm(V - (M @ R.T + t), axis=1)
    hk = keep & (tag == 0)
    ck = keep & (tag == 1)
    return {
        "R": R.tolist(), "t": t.tolist(),
        "rms_mm": float(np.sqrt(np.mean(resid[keep] ** 2))),
        "head_resid_med_mm": float(np.median(resid[hk])) if hk.any() else None,
        "cup_resid_med_mm": float(np.median(resid[ck])) if ck.any() else None,
        "n_used": int(keep.sum()), "n_head": int(hk.sum()), "n_cup": int(ck.sum()),
        "n_total": int(len(M)),
    }


def main():
    qa = json.load(open(f"{CACHE}/qtm_align.json"))
    results = {}
    per_p = {}
    filtered = []      # reps dropped by an OMC quality gate (with reason)
    n_ok = n_skip_bm = n_skip_fit = 0
    reps = [(p, r) for p, ent in qa.items() for r in ent.get("reps", [])]
    print(f"qtm_align reps: {len(reps)}", flush=True)
    for i, (pid, r) in enumerate(reps):
        vstem = r["video"]
        bmp = biomech_path(vstem)
        if bmp is None:
            n_skip_bm += 1; continue
        cup = biomech_cup(vstem)
        if cup is None:
            n_skip_bm += 1; continue

        b = np.load(bmp, allow_pickle=True)["keypoints3d"]
        head = b[:, J_HEAD, :3].astype(float)
        head[b[:, J_HEAD, 3] < 0.1] = np.nan

        try:
            t3 = load_trial(r["c3d"])
        except Exception:
            n_skip_fit += 1; continue
        omc_cup = t3.centroid()
        omc_head = t3.head_centroid() if t3.has_head() else None
        head_valid = np.isfinite(omc_head).all(1).mean() if omc_head is not None else 0.0
        cup_travel = bbox_travel(omc_cup)

        # --- FLAG-ONLY: nothing is dropped; we tag OMC quality so you filter downstream. ---
        flags = []
        if r.get("cls") == "broken":
            flags.append("cls_broken")
        if (r.get("sync_corr") or 1.0) < MIN_SYNC_CORR:
            flags.append("bad_sync")
        if cup_travel < MIN_CUP_TRAVEL_MM:
            flags.append("frozen_cup")
        if head_valid < MIN_HEAD_VALID_FRAC:
            flags.append("gappy_head")
        if flags:
            filtered.append((vstem, r["c3d"], ",".join(flags)))

        # Head-channel usability is a FIT-CORRECTNESS decision, not a filtering one:
        # a dead/gappy OMC head cluster would corrupt the rep's rotation if fed in.
        # So we still fit cup-only when the head is unusable — but the REP is kept.
        use_head = head_valid >= MIN_HEAD_VALID_FRAC
        head_in = head if use_head else np.full_like(head, np.nan)

        fit = fit_head_cup(head_in, cup,
                           omc_head if omc_head is not None else omc_cup,
                           omc_cup, t3.rate, r["lag"])
        if fit is None:
            n_skip_fit += 1; continue
        fit["c3d"] = r["c3d"]; fit["side"] = r["side"]; fit["lag"] = r["lag"]
        fit["omc_head_valid_frac"] = round(float(head_valid), 3)
        fit["omc_cup_travel_mm"] = round(float(cup_travel), 1)
        fit["sync_corr"] = r.get("sync_corr")
        fit["qtm_cls"] = r.get("cls")
        fit["used_head"] = bool(use_head)
        fit["omc_flags"] = flags        # [] = clean; else list of quality issues
        results[vstem] = fit
        per_p.setdefault(pid, []).append(fit)
        n_ok += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(reps)}] ok={n_ok} skip_bm={n_skip_bm} "
                  f"skip_fit={n_skip_fit} filtered={len(filtered)}", flush=True)

    # per-participant summary
    summary = {}
    for pid, fits in sorted(per_p.items()):
        rms = np.array([f["rms_mm"] for f in fits])
        hd = np.array([f["head_resid_med_mm"] for f in fits if f["head_resid_med_mm"] is not None])
        cd = np.array([f["cup_resid_med_mm"] for f in fits if f["cup_resid_med_mm"] is not None])
        summary[pid] = {
            "n": len(fits),
            "n_flagged": int(sum(1 for f in fits if f["omc_flags"])),
            "n_cup_only": int(sum(1 for f in fits if not f["used_head"])),
            "rms_med_mm": float(np.median(rms)),
            "head_resid_med_mm": float(np.median(hd)) if hd.size else None,
            "cup_resid_med_mm": float(np.median(cd)) if cd.size else None,
            "rms_p90_mm": float(np.percentile(rms, 90)),
        }

    out = {"n_reps": n_ok, "skipped_no_biomech": n_skip_bm, "skipped_fit": n_skip_fit,
           "n_flagged_omc": len(filtered),
           "note": "FLAG-ONLY: no rep is dropped for OMC quality; omc_flags tags issues so you "
                   "filter downstream. Head channel is fit cup-only when OMC head is gappy "
                   "(fit-correctness), but the rep is still aligned+reported.",
           "flag_thresholds": {"min_cup_travel_mm": MIN_CUP_TRAVEL_MM,
                               "min_head_valid_frac": MIN_HEAD_VALID_FRAC,
                               "min_sync_corr": MIN_SYNC_CORR},
           "flagged_omc": [{"video": v, "c3d": c, "flags": why} for v, c, why in filtered],
           "exclude_mm": EXCLUDE_MM, "summary": summary, "reps": results}
    outp = f"{CACHE}/align_head_cup.json"
    json.dump(out, open(outp, "w"), indent=1)
    print(f"\nwrote {outp}", flush=True)
    print(f"aligned {n_ok} reps  (skip_no_biomech={n_skip_bm} skip_fit={n_skip_fit}); "
          f"NONE dropped for OMC quality")
    if filtered:
        from collections import Counter
        cnt = Counter(f for _, _, w in filtered for f in w.split(","))
        print(f"  OMC-flagged reps (kept, tagged): {len(filtered)}  by flag: {dict(cnt)}")
    cuponly = [f for f in results.values() if not f["used_head"]]
    if cuponly:
        print(f"  fit cup-only (gappy/dead OMC head, rep kept): {len(cuponly)}")
    print(f"\n{'pid':<5}{'n':>4}{'rms_med':>9}{'head':>7}{'cup':>7}{'rms_p90':>9}")
    allr = []
    for pid, s in summary.items():
        allr.append(s["rms_med_mm"])
        print(f"{pid:<5}{s['n']:>4}{s['rms_med_mm']:>9.1f}"
              f"{(s['head_resid_med_mm'] or -1):>7.1f}{(s['cup_resid_med_mm'] or -1):>7.1f}"
              f"{s['rms_p90_mm']:>9.1f}")
    print(f"\nOVERALL rms_med across participants: {np.median(allr):.1f}mm")


if __name__ == "__main__":
    main()
