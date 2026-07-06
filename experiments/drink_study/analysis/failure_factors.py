"""Which factor drives per-frame tracking failure: cup SPEED, HAND-ON-CUP, or
CUP-NEAR-MOUTH?

Per frame we know whether the video cup track failed (mocap-video distance >50mm).
We model that against three physical factors, each entered on a LOG scale so the
effect is "really low when far, rising sharply when close" (closeness dominates,
far distances flatten) — a logistic regression on log-distances:

    logit P(fail) = b0 + b_mouth*log_d(cup,mouth) + b_hand*log_d(wrist,cup)
                       + b_speed*log(speed)

Distances are measured in the video/pose frame (where wrist & head live, from the
cached MeTRAbs biomech npz); the failure label comes from the mocap-vs-video error.
Pose is cached for ~23 reps, so the full 3-way model runs on those; speed alone is
available for all 669 (see phase_failure.py for the cohort-wide phase story).

Standardised coefficients + odds ratios say which factor most raises failure odds.

    python failure_factors.py        # fit + print + write cache/failure_factors.json
Run from the repo root (fuse_phases uses a repo-root-relative cache path).
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import json, os, glob, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "drink_study"))
import qtm_align as Q
import segment_cup_only as S
import fuse_phases as F

from _paths import DS
CACHE = DS / "cache"
TRACKDIR = CACHE / "track3d_clean3d_refill"
EPS = 1.0  # mm / (mm/s) floor so log is finite


def _pose_name(video: str):
    npz = set(os.path.basename(f) for f in glob.glob(str(CACHE / "biomech_*.npz")))
    stem = video.replace("__clean3d_refill", "")
    p = stem.split("_")[0]
    single = stem[len(p) + 1:] if stem.startswith(f"{p}_{p}_") else stem
    for n in (stem, single, f"{p}_{single}"):
        if f"biomech_{n}.npz" in npz:
            return n
    return None


def _frame_features(rep):
    """Per-frame (failed, speed, hand_cup_mm, cup_mouth_mm) for one rep, aligned to
    the per-frame mocap-video error. Returns None if no pose / sync rejects."""
    pn = _pose_name(rep["video"])
    if pn is None:
        return None
    vpath = TRACKDIR / (rep["video"] + ".json")
    if not vpath.exists():
        return None
    # per-frame error keyed by VIDEO-track frame (so it lines up with pose/cup-video)
    tr = Q.load_trial(rep["c3d"])
    if not tr.gt_quality()["ok"]:
        return None
    vt = Q.video_track(vpath)
    vr = Q._resample(vt, Q.VIDEO_FPS); mr = Q._resample(tr.centroid(), tr.rate)
    lag, corr = Q._xcorr_lag(Q._speed(vr, Q.COMMON_HZ), Q._speed(mr, Q.COMMON_HZ))
    if corr < Q.MIN_SYNC_CORR:
        return None
    if lag >= 0:
        v = vr[lag:]; mo = mr[:len(v)]; vfi = np.arange(lag, lag + len(v))
    else:
        mo = mr[-lag:]; v = vr[:len(mo)]; vfi = np.arange(len(mo))
    L = min(len(v), len(mo)); v, mo, vfi = v[:L], mo[:L], vfi[:L]
    ok = ~(np.isnan(v).any(1) | np.isnan(mo).any(1))
    v, mo, vfi = v[ok], mo[ok], vfi[ok]
    if len(v) < 10:
        return None
    R, t, _ = Q.kabsch(mo, v, robust=True)
    err = np.linalg.norm(v - (mo @ R.T + t), axis=1)
    failed = (err > Q.FAIL_MM).astype(int)

    # pose distances in the video frame (wrist->cup, head->cup), at the same vfi
    wrist, head, _, _ = F.pose_wrist_head(pn)
    _, cupv = S.load_track(vpath)        # raw video cup track frames
    Lp = min(len(wrist), len(cupv))
    vfi_c = np.clip(vfi, 0, Lp - 1)
    hand_cup = np.linalg.norm(wrist[vfi_c] - cupv[vfi_c], axis=1)
    cup_mouth = np.linalg.norm(head[vfi_c] - cupv[vfi_c], axis=1)

    # cup speed at these frames (on the video cup, present at vfi)
    cupspeed = Q._speed(Q._resample(vt, Q.VIDEO_FPS), Q.COMMON_HZ)[vfi]

    good = np.isfinite(hand_cup) & np.isfinite(cup_mouth) & np.isfinite(cupspeed)
    return dict(failed=failed[good], speed=cupspeed[good],
                hand_cup=hand_cup[good], cup_mouth=cup_mouth[good])


def _video_estimate(vpath: Path, estimate: str) -> np.ndarray:
    """(T,3) cup track from a chosen estimate: 'rts' (smoothed, default), 'kf'
    (forward filter), or 'consensus' (RAW multi-view measurement, no filter)."""
    d = json.loads(vpath.read_text())
    out = []
    for f in d["frames"]:
        p = f.get(estimate)
        out.append(p if p is not None else [np.nan] * 3)
    return np.asarray(out, float)


def _proxy_features(rep, estimate: str = "rts"):
    """Per-frame failure + MOCAP-ONLY proxies for all 669 reps (no pose needed):
      near_mouth  <- cup displacement-from-rest (validated r~-0.99 vs real cup->mouth)
      on_cup      <- in movement window (grasp->release); hand is on the cup throughout
      speed       <- mocap cup speed (real)
    Phases/disp come from the MOCAP cup so they are GT-defined, like phase_failure.py.
    `estimate` chooses which video track the FAILURE is measured against: 'rts'
    (default, smoothed) or 'consensus' (raw measurement, to test KF artifacts)."""
    vpath = TRACKDIR / (rep["video"] + ".json")
    if not vpath.exists():
        return None
    tr = Q.load_trial(rep["c3d"])
    if not tr.gt_quality()["ok"]:
        return None
    vt = _video_estimate(vpath, estimate)
    vr = Q._resample(vt, Q.VIDEO_FPS); mr = Q._resample(tr.centroid(), tr.rate)
    lag, corr = Q._xcorr_lag(Q._speed(vr, Q.COMMON_HZ), Q._speed(mr, Q.COMMON_HZ))
    if corr < Q.MIN_SYNC_CORR:
        return None
    seg = S.segment_cup_only(mr, fps=Q.COMMON_HZ)        # GT phases/disp on mocap grid
    disp, ph, spd = seg["disp"], seg["phase"], seg["speed"]
    in_win = np.isin(ph, [S.P_FWD, S.P_DRINK, S.P_BACK]).astype(float)
    # align error (video frames) to the mocap grid via lag (mocap index)
    if lag >= 0:
        v = vr[lag:]; mo = mr[:len(v)]; mfi = np.arange(len(v))
    else:
        mo = mr[-lag:]; v = vr[:len(mo)]; mfi = np.arange(-lag, -lag + len(mo))
    L = min(len(v), len(mo)); v, mo, mfi = v[:L], mo[:L], mfi[:L]
    ok = ~(np.isnan(v).any(1) | np.isnan(mo).any(1))
    v, mo, mfi = v[ok], mo[ok], mfi[ok]
    if len(v) < 10:
        return None
    R, t, _ = Q.kabsch(mo, v, robust=True)
    err = np.linalg.norm(v - (mo @ R.T + t), axis=1)
    mfi = np.clip(mfi, 0, len(disp) - 1)
    return dict(failed=(err > Q.FAIL_MM).astype(int),
                near_mouth=disp[mfi], on_cup=in_win[mfi], speed=spd[mfi])


def fit_proxy_all(estimate: str = "rts"):
    """Same factor model on ALL 669 reps using the validated mocap proxies.
    `estimate` = 'rts' (smoothed track, default) or 'consensus' (raw measurement)."""
    align = json.load(open(CACHE / "qtm_align.json"))
    rows = {"failed": [], "near_mouth": [], "on_cup": [], "speed": []}
    rep_id = []; n_reps = 0
    for pp in align.values():
        if not (isinstance(pp, dict) and pp.get("ok")):
            continue
        for rep in pp["reps"]:
            f = _proxy_features(rep, estimate=estimate)
            if f is None:
                continue
            for k in rows:
                rows[k].extend(f[k].tolist())
            rep_id.extend([n_reps] * len(f["failed"])); n_reps += 1
    for k in rows:
        rows[k] = np.asarray(rows[k], float)
    rep_id = np.asarray(rep_id, int); y = rows["failed"]; N = len(y)
    print(f"[proxy-all:{estimate}] {n_reps} reps, {N} frames, {y.mean():.1%} failed")

    # near_mouth: displacement is LARGE near mouth already (no -log needed, it IS reach);
    # use log(disp) so it rises smoothly. on_cup is a 0/1 gate. speed log.
    X = np.column_stack([np.log(rows["near_mouth"] + EPS),
                         rows["on_cup"],
                         np.log(rows["speed"] + EPS)])
    names = ["near_mouth (log disp-from-rest)", "on_cup (in-window gate)", "log_speed"]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    clf = LogisticRegression(max_iter=2000).fit(Xs, y)
    odds = np.exp(clf.coef_[0]); auc = roc_auc_score(y, clf.predict_proba(Xs)[:, 1])
    # cluster bootstrap CIs (resample reps)
    uniq = np.unique(rep_id); rng = np.random.default_rng(0); boot = np.full((400, 3), np.nan)
    for bI in range(400):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(rep_id == r) for r in pick])
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        try:
            boot[bI] = np.exp(LogisticRegression(max_iter=1500).fit(Xs[idx], yb).coef_[0])
        except Exception:
            pass
    ci = {names[i]: [float(np.nanpercentile(boot[:, i], 2.5)),
                     float(np.nanpercentile(boot[:, i], 97.5))] for i in range(3)}
    print(f"[proxy-all] AUC {auc:.3f}")
    for i in np.argsort(-np.abs(clf.coef_[0])):
        print(f"  {names[i]:34s} ×{odds[i]:.2f}  CI[{ci[names[i]][0]:.2f},{ci[names[i]][1]:.2f}]")

    # near_mouth nearly SEPARATES the outcome (a per-SD odds ratio explodes and is
    # meaningless), so report the separation directly: failure rate vs displacement,
    # and the share of failures that occur near the mouth.
    disp = rows["near_mouth"]
    edges = np.array([0, 50, 100, 200, 350, 500, 1e9])
    disp_curve = []
    for a_, b_ in zip(edges[:-1], edges[1:]):
        m = (disp >= a_) & (disp < b_)
        if m.sum() > 50:
            disp_curve.append(dict(lo=float(a_), hi=float(min(b_, disp.max())),
                                   n=int(m.sum()), fail=float(y[m].mean())))
    share_near = float((disp[y == 1] > 200).mean())          # frac of failures near mouth
    print(f"[proxy-all] {share_near:.1%} of all failures occur with cup >200mm from rest "
          f"(near the mouth); fail rate at rest (<50mm) "
          f"{disp_curve[0]['fail']:.1%} vs near-mouth (>500mm) {disp_curve[-1]['fail']:.1%}")

    out = dict(estimate=estimate, n_reps=n_reps, n_frames=int(N), fail_rate=float(y.mean()),
               auc=float(auc),
               factors=names, odds_per_sd=odds.tolist(), odds_ci95=ci,
               near_mouth_separates=True, share_failures_near_mouth=share_near,
               disp_fail_curve=disp_curve,
               note="ALL 669 reps, mocap-only proxies: near_mouth=displacement-from-rest "
                    "(validated r~-0.99 vs real cup->mouth on 23 pose reps), on_cup="
                    "in-movement-window gate, speed=mocap cup speed. near_mouth NEARLY "
                    "SEPARATES the outcome so its per-SD odds ratio is inflated/meaningless "
                    "-- report the disp_fail_curve + share_failures_near_mouth instead.")
    suffix = "" if estimate == "rts" else f"_{estimate}"
    json.dump(out, open(CACHE / f"failure_factors_proxy_all{suffix}.json", "w"), indent=2)
    print(f"wrote cache/failure_factors_proxy_all{suffix}.json")
    return out


def fit():
    align = json.load(open(CACHE / "qtm_align.json"))
    rows = {"failed": [], "speed": [], "hand_cup": [], "cup_mouth": []}
    rep_id = []
    n_reps = 0
    for pp in align.values():
        if not (isinstance(pp, dict) and pp.get("ok")):
            continue
        for rep in pp["reps"]:
            f = _frame_features(rep)
            if f is None:
                continue
            for k in rows:
                rows[k].extend(f[k].tolist())
            rep_id.extend([n_reps] * len(f["failed"]))   # cluster label per frame
            n_reps += 1
    for k in rows:
        rows[k] = np.asarray(rows[k], float)
    rep_id = np.asarray(rep_id, int)
    y = rows["failed"]
    N = len(y)
    print(f"{n_reps} pose reps, {N} frames, {y.mean():.1%} failed\n")

    # LOG features: closeness rises sharply near, flattens far.
    #   near_mouth  = -log(cup_mouth)   (large when close)
    #   on_cup      = -log(hand_cup)    (large when hand on cup)
    #   log_speed   =  log(speed)       (sign of coef tells fast- vs slow-driven)
    X = np.column_stack([
        -np.log(rows["cup_mouth"] + EPS),
        -np.log(rows["hand_cup"] + EPS),
        np.log(rows["speed"] + EPS),
    ])
    names = ["near_mouth (-log d_cup_mouth)", "on_cup (-log d_wrist_cup)", "log_speed"]
    # standardise so coefficients are comparable
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xs, y)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, clf.predict_proba(Xs)[:, 1])
    coef = clf.coef_[0]
    odds = np.exp(coef)

    # ---- cluster bootstrap CIs (resample REPS, not frames) -------------------
    # Frames within a rep are highly correlated, so a frame-level bootstrap would
    # badly understate uncertainty. Resample whole reps with replacement; refit;
    # take 2.5/97.5 percentiles of the odds ratios. N=23 reps -> wide, honest CIs.
    B = 1000
    uniq = np.unique(rep_id)
    boot = np.full((B, 3), np.nan)
    rng = np.random.default_rng(0)
    for bI in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(rep_id == r) for r in pick])
        yb = y[idx]
        if yb.min() == yb.max():            # degenerate resample (all one class)
            continue
        try:
            cb = LogisticRegression(max_iter=2000).fit(Xs[idx], yb)
            boot[bI] = np.exp(cb.coef_[0])
        except Exception:
            continue
    ci = {}
    for i in range(3):
        col = boot[:, i][~np.isnan(boot[:, i])]
        ci[names[i]] = [float(np.percentile(col, 2.5)), float(np.percentile(col, 97.5))]

    print(f"logistic on log-distances  (AUC {auc:.3f}, N={N} frames, {n_reps} reps)")
    print(f"{'factor':32s} {'odds/SD':>8s}  {'95% CI (cluster boot)':>22s}")
    order = np.argsort(-np.abs(coef))
    for i in order:
        lo, hi = ci[names[i]]
        print(f"{names[i]:32s} {odds[i]:>8.2f}  [{lo:5.2f}, {hi:5.2f}]")

    # ---- untangle confounded predictors --------------------------------------
    from sklearn.metrics import roc_auc_score as _auc
    def _fit(cols):
        Xx = np.column_stack(cols)
        c = LogisticRegression(max_iter=3000).fit(Xx, y)
        return c.coef_[0], float(_auc(y, c.predict_proba(Xx)[:, 1]))
    uni = {names[i]: float(np.exp(_fit([Xs[:, i]])[0][0])) for i in range(3)}      # each alone
    adj = {names[i]: float(odds[i]) for i in range(3)}                              # full model

    # hand-on-cup as a GATE: failure rate when hand on vs off the cup
    H = rows["hand_cup"]; thr = float(np.percentile(H, 75))
    on = H < thr
    gate = dict(threshold_mm=thr, frac_on=float(on.mean()),
                fail_on=float(y[on].mean()), fail_off=float(y[~on].mean()))

    # speed x mouth-distance interaction (slow vs fast failure within distance bands)
    M = rows["cup_mouth"]; SP = rows["speed"]
    bands = [(0, 150), (150, 300), (300, 600)]
    inter = []
    for lo, hi in bands:
        b = (M >= lo) & (M < hi)
        if b.sum() < 200:
            continue
        s = SP[b]; yy = y[b]; med = np.median(s)
        inter.append(dict(band=f"{lo}-{hi}", n=int(b.sum()),
                          fail_slow=float(yy[s < med].mean()),
                          fail_fast=float(yy[s >= med].mean())))

    corr = {"near_mouth~on_cup": float(np.corrcoef(Xs[:, 0], Xs[:, 1])[0, 1]),
            "on_cup~speed": float(np.corrcoef(Xs[:, 1], Xs[:, 2])[0, 1]),
            "near_mouth~speed": float(np.corrcoef(Xs[:, 0], Xs[:, 2])[0, 1])}

    print("\n--- untangling ---")
    print("univariate odds:", {k.split()[0]: round(v, 1) for k, v in uni.items()})
    print("adjusted  odds:", {k.split()[0]: round(v, 1) for k, v in adj.items()})
    print(f"hand-on gate: fail ON {gate['fail_on']:.0%} vs OFF {gate['fail_off']:.0%}")
    for r in inter:
        print(f"  mouth {r['band']:8s} fail slow {r['fail_slow']:.0%} / fast {r['fail_fast']:.0%}")

    out = dict(n_reps=n_reps, n_frames=int(N), fail_rate=float(y.mean()), auc=float(auc),
               factors=names, std_coef=coef.tolist(), odds_per_sd=odds.tolist(),
               odds_ci95={names[i]: ci[names[i]] for i in range(3)}, n_boot=B,
               univariate_odds=uni, adjusted_odds=adj, gate=gate,
               speed_mouth_interaction=inter, predictor_corr=corr,
               note="logistic on log-distances; near_mouth & on_cup are NEGATIVE log "
                    "distance (larger=closer). on_cup is near-constant 'on' so it acts "
                    "as a GATE (fail only when hand on cup); speed effect is "
                    "NON-MONOTONIC: slow fails at the mouth, fast fails in transport.")
    json.dump(out, open(CACHE / "failure_factors.json", "w"), indent=2)
    print("\nwrote cache/failure_factors.json")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy-all", action="store_true",
                    help="fit on all 669 reps with mocap proxies (no pose)")
    ap.add_argument("--raw", action="store_true",
                    help="with --proxy-all: measure failure against the RAW consensus "
                         "track instead of the RTS-smoothed one (tests KF artifacts)")
    a = ap.parse_args()
    if a.proxy_all:
        fit_proxy_all(estimate="consensus" if a.raw else "rts")
    else:
        fit()
