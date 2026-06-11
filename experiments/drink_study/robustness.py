"""How good must a detector be for the filtered+interpolated track to be useful?

Builds a PREDICTIVE usefulness model. We degrade the cached student detections
along 5 measurable axes, and for each degraded condition record:

  INPUTS  (all measurable from raw detections alone -- no ground truth):
    det_rate     : mean per-camera fraction of frames with a detection
    n_cams       : how many cameras contribute
    reproj_px    : median inter-camera reprojection agreement (the §3 metric)
    inlier_frac  : mean fraction of cams within 30px of the consensus
    tri_rate     : fraction of frames with a >=2-cam triangulation
    fps_eff      : effective frame rate after decimation
  OUTCOME (vs a FROZEN ground-truth track -> only known offline):
    p99_mm       : worst-1% track error of the filtered+interpolated estimate
    good         : p99_mm < CUP_R (35mm)  -- did the track stay on the cup?

Ground truth is the BEST student (pscale_4) clean gating->KF->RTS track on ONE
trial we VISUALLY VERIFIED in the Rerun replay (P01 141546: 870/870 KF coverage,
on-cup), frozen once. Every degraded condition reuses that same trial's raw dets,
so we measure drift from a known-good track -- not a circular per-rep reference.

Then fit a logistic regression INPUTS -> P(good). Result: a rule you can apply
to ANY new model by measuring only its raw-detection metrics (no GT, no sweep),
to predict whether the gating/KF/RTS pipeline will track well -- plus the
per-axis recoverable-quality envelope (the breakpoint on each knob alone).

Reuses the exact gating->KF->RTS pipeline from kf_accuracy.py and the
no-GT agreement metrics from agreement.py. No GPU / no re-inference: degrades
the cached student dets.

Writes cache/robustness.json for the notebook.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from kalman_3d import load_calibration, triangulate_dlt, project, KalmanFilter3D
import kf_accuracy as ka      # gated_consensus, _rts_backward, build_reference, FPS, CUP_R, ...

FPS = ka.FPS
CUP_R = ka.CUP_R              # 35mm usable-track threshold
THR = ka.THR                 # 30px consensus inlier gate
RES = (1920, 1080)
# Ground truth = the BEST student (pscale_4) clean RTS track on ONE trial we
# VISUALLY VERIFIED in the Rerun replay (870/870 KF coverage, on-cup). Frozen
# once as X_truth(t); all degraded conditions are scored as drift from it -- so
# "truth" is an explicit known-good model on a known-good trial, not a circular
# per-rep self-reference.
GT_P = "P01"
GT_STEM = "P01_drinking_right_20231220_141546"
OUT = Path("experiments/drink_study/cache/robustness.json")

# --- degradation axes (each a "how much worse can the detector be" knob) ---
# Ranges pushed PAST the breaking point so each knob's cliff is visible, not just
# its safe region.
DROP = [0.0, 0.4, 0.7, 0.9, 0.95, 0.98, 0.99]   # fraction of detections removed (per cam)
CORRUPT = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]        # fraction of cams pushed OFF the cup each frame
CORRUPT_PX = 120.0                               # how far a corrupted cam is shoved (px)
SIGMA = [0.0, 5.0, 10.0, 20.0, 40.0, 80.0]       # gaussian pixel jitter (the §3b axis)
NCAMS = [2, 3, 4, 6, 10]                         # feed only the n best-covered cams
DECIM = [1, 2, 4, 8, 16]                          # keep every Kth frame (60 -> ~4fps)
N_RANDOM = 240                                   # random multi-axis combos for the predictor fit
SEED = 0


# ---------------------------------------------------------------- degradation
def degrade(dets, cams, rng, drop=0.0, corrupt=0.0, sigma=0.0, n_cams=None, decim=1):
    """Return a degraded copy of `dets` + the effective camera set + fps.

    drop     : per-cam, independently null out this fraction of detections.
    corrupt  : each frame, shove this fraction of the detecting cams by
               CORRUPT_PX in a random direction (simulates a model that
               localizes a wrong/nearby object -> destroys inter-cam agreement).
    sigma    : add N(0,sigma) px jitter to every surviving detection.
    n_cams   : keep only the n best-covered cameras.
    decim    : keep every `decim`-th frame (others become None for all cams).
    """
    n = min(len(v) for v in dets.values())
    use = cams
    if n_cams is not None:
        cov = {c: sum(x is not None for x in dets[c]) for c in cams}
        use = sorted(sorted(cams, key=lambda c: -cov[c])[:n_cams],
                     key=lambda k: int(k.split("_")[1]))
    out = {c: [None] * n for c in use}
    for fr in range(n):
        if fr % decim != 0:
            continue
        present = [c for c in use if dets[c][fr] is not None]
        # drop
        present = [c for c in present if rng.random() >= drop]
        # choose which present cams get corrupted this frame
        ncorr = int(round(corrupt * len(present)))
        corr = set(rng.choice(present, size=ncorr, replace=False)) if ncorr else set()
        for c in present:
            z = np.array(dets[c][fr], float)
            if c in corr:
                ang = rng.uniform(0, 2 * np.pi)
                z = z + CORRUPT_PX * np.array([np.cos(ang), np.sin(ang)])
            if sigma > 0:
                z = z + rng.normal(0, sigma, 2)
            out[c][fr] = (float(z[0]), float(z[1]))
    return out, use, FPS / decim


# ------------------------------------------------------- measurable (no-GT) inputs
def measure(dg, use, calib, fps_eff):
    """No-ground-truth metrics a real deployment could compute from raw dets."""
    n = min(len(v) for v in dg.values())
    sampled = [fr for fr in range(n) if any(dg[c][fr] is not None for c in use)]
    det_rate = float(np.mean([sum(dg[c][fr] is not None for c in use) / len(use)
                              for fr in sampled])) if sampled else 0.0
    reproj, inlier, tri = [], [], 0
    for fr in sampled:
        obs = {c: dg[c][fr] for c in use if dg[c][fr] is not None}
        if len(obs) < 2:
            continue
        tri += 1
        X = triangulate_dlt([calib[c] for c in obs], [np.array(obs[c]) for c in obs])
        e = np.array([float(np.hypot(*(project(calib[c], X)[0] - np.array(obs[c]))))
                      for c in obs])
        reproj.append(float(np.median(e)))
        inlier.append(float(np.mean(e < THR)))
    return {
        "det_rate": round(det_rate, 3),
        "n_cams": len(use),
        "reproj_px": round(float(np.median(reproj)), 2) if reproj else 999.0,
        "inlier_frac": round(float(np.mean(inlier)), 3) if inlier else 0.0,
        "tri_rate": round(tri / len(sampled), 3) if sampled else 0.0,
        "fps_eff": round(fps_eff, 1),
    }


# ------------------------------------------------------------------- outcome
def outcome(dg, use, calib, ref, n):
    """Run the validated gating->KF->RTS pipeline; score worst-1% vs reference."""
    ka.calib, ka.DETS, ka.CAMS, ka.N = calib, dg, use, n
    kf = KalmanFilter3D()
    xs_pred, Ps_pred, xs_upd, Ps_upd, Fs, has = [], [], [], [], [], []
    for fr in range(n):
        t = fr / FPS
        obs = {c: np.array(dg[c][fr], float) for c in use if dg[c][fr] is not None}
        if not kf.initialized:
            # seed from gated consensus; if it can't bootstrap (e.g. 2-cam, which
            # needs >=3 to triangulate a gated seed), fall back to the frozen
            # truth point so we measure *tracking* not *seeding* (matches
            # kf_accuracy's ref-seeded n_cams sweep).
            seed = ka.gated_consensus({c: tuple(v) for c, v in obs.items()})
            if seed is None and ref[fr] is not None and obs:
                seed = ref[fr]
            if seed is not None:
                kf.init(seed, t)
            for L, val in ((xs_pred, kf.x), (Ps_pred, kf.P), (xs_upd, kf.x), (Ps_upd, kf.P)):
                L.append(val.copy() if kf.initialized else None)
            Fs.append(np.eye(6)); has.append(kf.initialized)
            continue
        dt = t - kf.t_last
        kf.predict(t)
        F = np.eye(6); F[:3, 3:] = dt * np.eye(3)
        xs_pred.append(kf.x.copy()); Ps_pred.append(kf.P.copy()); Fs.append(F)
        for c in obs:
            kf.update(calib[c], obs[c], t)
        xs_upd.append(kf.x.copy()); Ps_upd.append(kf.P.copy()); has.append(True)
    xs_s = ka._rts_backward(xs_pred, Ps_pred, xs_upd, Ps_upd, Fs, has)
    est = [x[:3].copy() if x is not None else None for x in xs_s]
    errs = [float(np.linalg.norm(e - r)) for e, r in zip(est, ref)
            if e is not None and r is not None]
    cov = sum(e is not None for e in est) / n
    if not errs:
        return {"coverage": round(cov, 3), "median_mm": None, "p99_mm": None, "good": False}
    p99 = float(np.percentile(errs, 99))
    return {"coverage": round(cov, 3),
            "median_mm": round(float(np.median(errs)), 1),
            "p99_mm": round(p99, 1),
            "good": bool(p99 < CUP_R and cov > 0.5)}   # useful = on-cup tail AND covers the rep


# ------------------------------------------------------------------- sampling
def conditions(rng):
    """Per-axis sweeps (clean baseline on the others) + random multi-axis combos."""
    base = dict(drop=0.0, corrupt=0.0, sigma=0.0, n_cams=None, decim=1)
    out = []
    for v in DROP:    out.append({"axis": "drop", **{**base, "drop": v}})
    for v in CORRUPT: out.append({"axis": "corrupt", **{**base, "corrupt": v}})
    for v in SIGMA:   out.append({"axis": "sigma", **{**base, "sigma": v}})
    for v in NCAMS:   out.append({"axis": "n_cams", **{**base, "n_cams": v}})
    for v in DECIM:   out.append({"axis": "decim", **{**base, "decim": v}})
    for _ in range(N_RANDOM):
        out.append({"axis": "random",
                    "drop": float(rng.choice(DROP)),
                    "corrupt": float(rng.choice(CORRUPT)),
                    "sigma": float(rng.choice(SIGMA)),
                    "n_cams": (None if rng.random() < 0.4 else int(rng.choice(NCAMS))),
                    "decim": int(rng.choice(DECIM))})
    return out


def main():
    # --- freeze the ground-truth track from the verified-good trial ---
    calib = load_calibration(f"data/calib/{GT_P}/calibration.toml", target_size=RES)
    raw = ka.student_dets(GT_P, GT_STEM)
    dets = {c: v for c, v in raw.items() if c in calib}
    cams = sorted(dets, key=lambda k: int(k.split("_")[1]))
    n = min(len(v) for v in dets.values())
    ka.calib, ka.DETS, ka.CAMS, ka.N, ka.GLASS = calib, dets, cams, n, None
    ref = ka.build_reference()                       # pscale_4 clean RTS = X_truth(t)
    ref_cov = sum(r is not None for r in ref)
    print(f"GROUND TRUTH: {GT_P}/{GT_STEM}  {n}f, {len(cams)} cams, "
          f"clean-track coverage {ref_cov}/{n} ({100*ref_cov/n:.0f}%)", flush=True)
    if ref_cov < 0.9 * n:
        print("  WARNING: verified trial's clean track is sparse -- check GT_STEM", flush=True)

    # --- degrade THOSE dets along the axes; score drift from frozen truth ---
    conds = conditions(np.random.default_rng(SEED))
    print(f"running {len(conds)} degraded conditions ...", flush=True)
    rows = []
    for i, cond in enumerate(conds):
        rng = np.random.default_rng(hash((GT_P, i)) % (2**32))
        dg, use, fps_eff = degrade(dets, cams, rng,
                                   drop=cond["drop"], corrupt=cond["corrupt"],
                                   sigma=cond["sigma"], n_cams=cond["n_cams"],
                                   decim=cond["decim"])
        inp = measure(dg, use, calib, fps_eff)
        out = outcome(dg, use, calib, ref, n)
        rows.append({**cond, **inp, **out})
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(conds)} done", flush=True)
    ng = sum(r["good"] for r in rows)
    print(f"{ng}/{len(rows)} conditions stayed on-cup (p99 < {CUP_R:.0f}mm vs frozen truth)", flush=True)

    analyze(rows)


def analyze(rows):
    FEATS = ["det_rate", "n_cams", "reproj_px", "inlier_frac", "tri_rate", "fps_eff"]
    X = np.array([[r[f] for f in FEATS] for r in rows], float)
    y = np.array([r["good"] for r in rows], int)
    print(f"\n=== {len(rows)} conditions, {y.mean()*100:.0f}% stayed on-cup ===", flush=True)

    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score, accuracy_score, r2_score

    # --- (1) LINEAR predictor: INPUTS -> log10(p99_mm), the magnitude/margin ---
    # p99 is heavy-tailed (good ~20mm, catastrophic >1000mm) so we regress in log
    # space; the cup radius maps to a threshold log10(35) on the prediction.
    keep = [i for i, r in enumerate(rows) if r["p99_mm"] is not None]
    Xr = X[keep]
    logp99 = np.log10(np.array([rows[i]["p99_mm"] for i in keep], float))
    lin = make_pipeline(StandardScaler(), LinearRegression())
    pred_log = cross_val_predict(lin, Xr, logp99, cv=5)
    r2 = r2_score(logp99, pred_log)
    lin.fit(Xr, logp99)
    lcoef = lin.named_steps["linearregression"].coef_
    lin_w = sorted(zip(FEATS, lcoef), key=lambda kv: -abs(kv[1]))
    print(f"\nLINEAR predictor  log10(p99_mm) (5-fold CV): R2={r2:.3f}", flush=True)
    print(f"  cup radius = log10({CUP_R:.0f}) = {np.log10(CUP_R):.2f}; "
          f"predict below it -> on-cup. standardized weights:", flush=True)
    for f, w in lin_w:
        print(f"    {f:>12}: {w:+.2f}  (larger {f} -> {'higher' if w>0 else 'lower'} error)", flush=True)

    # --- (2) LOGISTIC predictor: INPUTS -> P(good), the clean decision rule ---
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    proba = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, proba)
    acc = accuracy_score(y, proba > 0.5)
    for r, pp in zip(rows, proba):          # store CV prediction for the plotter
        r["pred_good"] = round(float(pp), 4)
    clf.fit(X, y)
    coef = clf.named_steps["logisticregression"].coef_[0]
    weights = sorted(zip(FEATS, coef), key=lambda kv: -abs(kv[1]))
    print(f"\nLOGISTIC predictor  P(track good) (5-fold CV): AUC={auc:.3f}  acc={acc:.3f}", flush=True)
    print("  standardized weights (which metric predicts 'will track well'):", flush=True)
    for f, w in weights:
        print(f"    {f:>12}: {w:+.2f}", flush=True)

    # --- (3) cheapest SINGLE-metric go/no-go: each metric's solo AUC + best
    #         threshold (Youden's J). Answers "what one number do I check on a
    #         new model?" -- no multi-feature fit needed in deployment. ---
    print("\nSINGLE-METRIC GATE (solo predictive power, best threshold):", flush=True)
    gates = {}
    for j, f in enumerate(FEATS):
        xf = X[:, j]
        # higher-is-better metrics vs lower-is-better (reproj_px): orient the score
        sign = -1.0 if f == "reproj_px" else 1.0
        a = roc_auc_score(y, sign * xf)
        # threshold maximizing TPR-FPR (Youden) -> a usable cutoff
        order = np.argsort(sign * xf)
        thr_best, j_best = None, -1.0
        P, Nn = y.sum(), (1 - y).sum()
        for cut in np.unique(xf):
            pred = (sign * xf >= sign * cut)
            tpr = (pred & (y == 1)).sum() / max(P, 1)
            fpr = (pred & (y == 0)).sum() / max(Nn, 1)
            if tpr - fpr > j_best:
                j_best, thr_best = tpr - fpr, float(cut)
        gates[f] = {"auc": round(float(a), 3), "threshold": round(thr_best, 3),
                    "rule": f"{f} {'<=' if f=='reproj_px' else '>='} {thr_best:.3g}"}
    for f, g in sorted(gates.items(), key=lambda kv: -kv[1]["auc"]):
        print(f"    {f:>12}: AUC={g['auc']:.3f}  rule: {g['rule']}", flush=True)

    # --- per-axis recoverable-quality envelope (breakpoint on each knob alone) ---
    print("\nPER-AXIS ENVELOPE (sweeping one knob, others clean):", flush=True)
    env = {}
    for axis in ["drop", "corrupt", "sigma", "n_cams", "decim"]:
        ax = [r for r in rows if r["axis"] == axis]
        if not ax:
            continue
        key = axis
        ax = sorted(ax, key=lambda r: r[key])
        # median over participants at each level
        levels = sorted({r[key] for r in ax}, key=lambda v: (v is None, v))
        line = []
        for lv in levels:
            cell = [r for r in ax if r[key] == lv]
            good_frac = float(np.mean([r["good"] for r in cell]))
            p99 = np.median([r["p99_mm"] for r in cell if r["p99_mm"] is not None])
            line.append((lv, good_frac, p99))
        env[axis] = line
        s = "  ".join(f"{lv}:{gf:.0%}/{p99:.0f}mm" for lv, gf, p99 in line)
        print(f"  {axis:>8}: {s}", flush=True)
    print("  (level : fraction-on-cup / median worst-1% err)", flush=True)

    OUT.write_text(json.dumps({
        "ground_truth": f"{GT_P}/{GT_STEM} (pscale_4 clean RTS, verified)",
        "cup_radius_mm": CUP_R, "n_conditions": len(rows), "feats": FEATS,
        "linear": {"target": "log10(p99_mm)", "r2": round(float(r2), 3),
                   "log_cup_radius": round(float(np.log10(CUP_R)), 3),
                   "weights": {f: round(float(w), 3) for f, w in zip(FEATS, lcoef)}},
        "logistic": {"target": "P(p99<cup_radius & cov>0.5)", "auc": round(auc, 3),
                     "acc": round(acc, 3),
                     "weights": {f: round(float(w), 3) for f, w in zip(FEATS, coef)}},
        "single_metric_gates": gates,
        "envelope": {a: [[lv, round(gf, 3), round(float(p99), 1)]
                         for lv, gf, p99 in line] for a, line in env.items()},
        "rows": rows,
    }, indent=2))
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
