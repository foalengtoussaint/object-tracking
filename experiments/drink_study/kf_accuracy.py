"""How accurate must detection be for the 3D KF to track the cup well?

Offline analysis across multiple calibrated participants. The reference track
AND the swept input both come from the **student** model (pscale_4, the best
single-class cup student) — the model that actually runs at inference and feeds
the KF — not the teacher, so the budget reflects real deployment detections.

Student detections are run once per participant and cached (GPU is the only cost;
re-runs are cache hits).

Procedure (per participant)
---------------------------
1. Build a CLEAN reference track: per-frame inlier-gated consensus (>=3 cams,
   thr=30px) over ALL cameras, then RTS-smoothed -> the "true" cup path X_ref(t).
   This is our ground truth for the rep (the cameras agree on it geometrically).
2. Sweep two knobs that degrade the detector and ask when the KF loses the cup:
     (a) pixel noise  sigma_px : add N(0,sigma) jitter to every 2D detection.
     (b) #cameras     n_cams   : feed the KF only the n best-covered cameras.
   For each setting, run a fresh KalmanFilter3D over the rep (EKF gate on, the
   same filter used at runtime) and measure, vs X_ref:
     - coverage  : frames with a KF estimate
     - track err : median / p90 3D distance to X_ref (mm)
     - on_glass  : (P01 only) fraction of frames the estimate projects onto the
                   cam_10 static-glass spot — a hard failure = locked onto distractor.
3. Report the breakpoints: the sigma / n_cams beyond which median track error
   exceeds a usable threshold (cup radius ~35 mm), and whether they hold across
   participants.

Writes results to cache/kf_accuracy_multi.json for the notebook.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from kalman_3d import load_calibration, triangulate_dlt, project, KalmanFilter3D

FPS = 60.0
THR = 30.0
CUP_R = 35.0                              # mm, a usable-track threshold
RES = (1920, 1080)
CONF = 0.25
CUP_CLASS = [0]                           # student is single-class cup
PARTICIPANTS = ["P01", "P06", "P19", "P23"]
STUDENT = "experiments/drink_study/runs/pscale_4/weights/best.pt"
CLIPS_ROOT = Path("/home/imove/Documents/clips")
DETCACHE = Path("experiments/drink_study/cache/student_dets")
GLASS_P01 = np.array([1672, 745])         # cam_10 static-glass pixel (P01 only)
OUT = Path("experiments/drink_study/cache/kf_accuracy_multi.json")

# --- per-participant context (set in main before each run) ---
calib: dict = {}
DETS: dict = {}
CAMS: list = []
N: int = 0
GLASS = None                              # glass-check pixel, or None to skip


def rep_stems(p: str, n: int, hand: str = "right") -> list[str]:
    pdir = CLIPS_ROOT / p
    stems = sorted({f.name.rsplit(".", 2)[0]
                    for f in pdir.glob(f"{p}_drinking_{hand}_*.mp4")})
    return stems[:n]


def student_dets(p: str, stem: str) -> dict:
    """pscale_4 student cup-centroids on one rep's 10 cams, cached per participant."""
    DETCACHE.mkdir(parents=True, exist_ok=True)
    cf = DETCACHE / f"{p}_{stem}__pscale_4__c{CONF}.json"
    if cf.exists():
        d = json.loads(cf.read_text())
        return {c: [tuple(x) if x else None for x in v] for c, v in d.items()}
    from ultralytics import YOLO
    from agreement import detect_rep
    pdir = CLIPS_ROOT / p
    rep = {f"cam_{c}": pdir / f"{stem}.{c}.mp4" for c in range(1, 11)
           if f"cam_{c}" in calib and (pdir / f"{stem}.{c}.mp4").exists()}
    print(f"  [{p}] student on {stem} ({len(rep)} cams) ...", flush=True)
    dets = detect_rep(YOLO(STUDENT), rep, CONF, CUP_CLASS, verbose=True)
    cf.write_text(json.dumps(dets))
    return {c: [tuple(x) if x else None for x in v] for c, v in dets.items()}


def gated_consensus(obs, thr=THR, minc=3):
    cur = dict(obs)
    while len(cur) >= 2:
        X = triangulate_dlt([calib[c] for c in cur], [np.array(cur[c]) for c in cur])
        e = {c: float(np.hypot(*(project(calib[c], X)[0] - np.array(cur[c])))) for c in cur}
        w = max(e, key=e.get)
        if e[w] <= thr:
            break
        del cur[w]
    if len(cur) < minc:
        return None
    return triangulate_dlt([calib[c] for c in cur], [np.array(cur[c]) for c in cur])


def _rts_backward(xs_pred, Ps_pred, xs_upd, Ps_upd, Fs, has):
    """Rauch-Tung-Striebel backward pass over stored forward-filter states.
    Returns the smoothed full 6D states (pos+vel) per frame, None where unfilled."""
    n = len(xs_upd)
    xs_s = [x.copy() if x is not None else None for x in xs_upd]
    start = next((i for i, h in enumerate(has) if h), None)
    if start is None:
        return [None] * n
    for k in range(n - 2, start - 1, -1):
        if xs_upd[k] is None:
            continue
        Pp = Ps_pred[k + 1]
        try:
            C = Ps_upd[k] @ Fs[k + 1].T @ np.linalg.inv(Pp)
        except np.linalg.LinAlgError:
            continue
        xs_s[k] = xs_upd[k] + C @ (xs_s[k + 1] - xs_pred[k + 1])
    return xs_s


def build_reference():
    """Clean RTS-smoothed reference cup track over all cams (the 'truth')."""
    raw = [gated_consensus({c: DETS[c][fr] for c in CAMS if DETS[c][fr] is not None})
           for fr in range(N)]
    # forward linear KF on the 3D consensus points, storing for RTS
    kf = KalmanFilter3D()
    xs_pred, Ps_pred, xs_upd, Ps_upd, Fs, has = [], [], [], [], [], []
    for fr in range(N):
        t = fr / FPS
        if not kf.initialized:
            if raw[fr] is not None:
                kf.init(raw[fr], t)
            xs_pred.append(kf.x.copy() if kf.initialized else None)
            Ps_pred.append(kf.P.copy() if kf.initialized else None)
            xs_upd.append(kf.x.copy() if kf.initialized else None)
            Ps_upd.append(kf.P.copy() if kf.initialized else None)
            Fs.append(np.eye(6)); has.append(kf.initialized)
            continue
        dt = t - kf.t_last
        kf.predict(t)
        F = np.eye(6); F[:3, 3:] = dt * np.eye(3)
        xs_pred.append(kf.x.copy()); Ps_pred.append(kf.P.copy()); Fs.append(F)
        if raw[fr] is not None:
            kf.update_3d(raw[fr], t, meas_noise_mm=30.0, gate=1e9)
        xs_upd.append(kf.x.copy()); Ps_upd.append(kf.P.copy()); has.append(True)
    xs_s = _rts_backward(xs_pred, Ps_pred, xs_upd, Ps_upd, Fs, has)
    return [x[:3].copy() if x is not None else None for x in xs_s]


def run_kf(sigma_px=0.0, n_cams=None, seed=0, ref=None, rts=False):
    """Fresh runtime KF over the rep with noisy / reduced detections.

    If `ref` is given, the KF is seeded from the clean reference whenever it is
    uninitialized — this isolates *tracking* ability from *seeding* ability, so
    the n_cams sweep measures whether few cams can hold a lock once started,
    not whether they can bootstrap the gated consensus seed (which needs >=3).

    `rts=False` -> causal forward EKF estimate per frame (the ONLINE tracker).
    `rts=True`  -> forward EKF then an RTS backward pass (the OFFLINE smoother,
    future-aware — the mode used for label cleaning in §4). Same noisy/reduced
    input either way, so the two curves are directly comparable.
    """
    rng = np.random.default_rng(seed)
    use_cams = CAMS
    if n_cams is not None:
        # the n cameras with the most detections (best coverage)
        cov = {c: sum(x is not None for x in DETS[c]) for c in CAMS}
        use_cams = sorted(sorted(CAMS, key=lambda c: -cov[c])[:n_cams],
                          key=lambda k: int(k.split("_")[1]))
    kf = KalmanFilter3D()
    est = []
    # forward-filter states stored for an optional RTS backward pass
    xs_pred, Ps_pred, xs_upd, Ps_upd, Fs, has = [], [], [], [], [], []
    for fr in range(N):
        t = fr / FPS
        obs = {}
        for c in use_cams:
            z = DETS[c][fr]
            if z is None:
                continue
            zz = np.array(z, float)
            if sigma_px > 0:
                zz = zz + rng.normal(0, sigma_px, 2)
            obs[c] = zz
        if not kf.initialized:
            seed_pt = (ref[fr] if ref is not None and ref[fr] is not None
                       else gated_consensus({c: tuple(v) for c, v in obs.items()}))
            if seed_pt is not None:
                kf.init(seed_pt, t)
            est.append(kf.x[:3].copy() if kf.initialized else None)
            xs_pred.append(kf.x.copy() if kf.initialized else None)
            Ps_pred.append(kf.P.copy() if kf.initialized else None)
            xs_upd.append(kf.x.copy() if kf.initialized else None)
            Ps_upd.append(kf.P.copy() if kf.initialized else None)
            Fs.append(np.eye(6)); has.append(kf.initialized)
            continue
        dt = t - kf.t_last
        kf.predict(t)                       # single frame-level predict (sets t_last=t)
        F = np.eye(6); F[:3, 3:] = dt * np.eye(3)
        xs_pred.append(kf.x.copy()); Ps_pred.append(kf.P.copy()); Fs.append(F)
        for c in obs:                       # each update's internal predict is a no-op (dt=0)
            kf.update(calib[c], obs[c], t)
        xs_upd.append(kf.x.copy()); Ps_upd.append(kf.P.copy()); has.append(True)
        est.append(kf.x[:3].copy())
    if not rts:
        return est
    xs_s = _rts_backward(xs_pred, Ps_pred, xs_upd, Ps_upd, Fs, has)
    return [x[:3].copy() if x is not None else None for x in xs_s]


def score(est, ref):
    cov = sum(e is not None for e in est)
    errs = [float(np.linalg.norm(e - r))
            for e, r in zip(est, ref) if e is not None and r is not None]
    # glass-lock check only for participants with a known cam_10 glass spot (P01)
    on_glass = checked = 0
    if GLASS is not None and "cam_10" in calib:
        for e in est:
            if e is None:
                continue
            uv, infr = project(calib["cam_10"], e)
            if infr:
                checked += 1
                on_glass += int(np.hypot(*(uv - GLASS)) < 80)
    return {
        "coverage": round(100 * cov / N, 1),
        # distribution of the PER-FRAME cup track error within this rep
        "p10_mm": round(float(np.percentile(errs, 10)), 1) if errs else None,
        "median_mm": round(float(np.median(errs)), 1) if errs else None,
        "p90_mm": round(float(np.percentile(errs, 90)), 1) if errs else None,
        "p99_mm": round(float(np.percentile(errs, 99)), 1) if errs else None,
        "max_mm": round(float(np.max(errs)), 1) if errs else None,
        "on_glass_pct": round(100 * on_glass / checked, 1) if checked else None,
    }


SIGMAS = [0, 2, 5, 10, 20, 40, 80]
NCAMS = [2, 3, 4, 5, 7, 10]
REPEATS = 5
REPS_PER_P = 1                            # one trial per participant


def run_one_rep(p: str, stem: str) -> dict | None:
    """Set per-rep globals, build student-based reference, run both sweeps."""
    global calib, DETS, CAMS, N, GLASS
    calib = load_calibration(f"data/calib/{p}/calibration.toml", target_size=RES)
    raw = student_dets(p, stem)
    DETS = {c: v for c, v in raw.items() if c in calib}
    if len(DETS) < 3:
        print(f"  [{p}/{stem}] <3 calibrated cams, skip", flush=True)
        return None
    CAMS = sorted(DETS, key=lambda k: int(k.split("_")[1]))
    N = min(len(v) for v in DETS.values())
    GLASS = GLASS_P01 if p == "P01" else None

    ref = build_reference()
    ref_cov = sum(r is not None for r in ref)
    if ref_cov < 0.3 * N:
        print(f"  [{p}/{stem}] reference too sparse ({ref_cov}/{N}), skip", flush=True)
        return None

    def avg_over_seeds(rts, **kw):
        rows = [score(run_kf(seed=s, rts=rts, **kw), ref) for s in range(REPEATS)]
        out = {}
        for k in rows[0]:
            vals = [r[k] for r in rows if r[k] is not None]
            out[k] = round(float(np.mean(vals)), 1) if vals else None
        return out

    def sweeps(rts):
        noise = {s: (avg_over_seeds(rts, sigma_px=s) if s > 0
                     else score(run_kf(sigma_px=0, rts=rts), ref))
                 for s in SIGMAS}
        cams = {nc: score(run_kf(n_cams=nc, ref=ref, rts=rts), ref) for nc in NCAMS}
        return noise, cams

    noise, cams = sweeps(rts=False)           # causal online EKF
    noise_r, cams_r = sweeps(rts=True)        # forward + RTS smoother (offline)

    # --- combined noise x #cameras grid (causal EKF): does the cliff move with N? ---
    # ref-seeded so a small N still bootstraps; median track err per (n_cams, sigma).
    grid = {nc: {s: avg_over_seeds(False, sigma_px=s, n_cams=nc, ref=ref)["median_mm"]
                 for s in SIGMAS} for nc in NCAMS}

    print(f"  [{p}/{stem}] {N}f ref_cov={ref_cov}/{N}  EKF: med@20px={noise[20]['median_mm']} "
          f"med@40px={noise[40]['median_mm']} 2cam={cams[2]['median_mm']}  |  "
          f"RTS: med@20px={noise_r[20]['median_mm']} med@40px={noise_r[40]['median_mm']} "
          f"2cam={cams_r[2]['median_mm']}",
          flush=True)
    return {"participant": p, "rep": stem, "n_frames": N, "ref_coverage": ref_cov,
            "noise_sweep": noise, "cam_sweep": cams,
            "noise_sweep_rts": noise_r, "cam_sweep_rts": cams_r,
            "grid": grid}


def _agg(per_rep, sweep_key, knobs):
    """For each knob, aggregate across reps. Two different things:
    - median_mm/p25/p75 = rep-to-rep spread of each rep's TYPICAL-frame (median) error.
    - pf_p10/pf_p90 = median across reps of each rep's WITHIN-rep per-frame p10/p90,
      i.e. the typical rep's best-10% and worst-10% cup-tracking frames.
    """
    out = {}
    for k in knobs:
        cell = [r[sweep_key][k] for r in per_rep]
        meds = [c["median_mm"] for c in cell if c["median_mm"] is not None]
        pf10 = [c["p10_mm"] for c in cell if c.get("p10_mm") is not None]
        pf90 = [c["p90_mm"] for c in cell if c.get("p90_mm") is not None]
        pf99 = [c["p99_mm"] for c in cell if c.get("p99_mm") is not None]
        gl = [c["on_glass_pct"] for c in cell if c.get("on_glass_pct") is not None]
        out[k] = {"median_mm": round(float(np.median(meds)), 1) if meds else None,
                  "p25_mm": round(float(np.percentile(meds, 25)), 1) if meds else None,
                  "p75_mm": round(float(np.percentile(meds, 75)), 1) if meds else None,
                  "pf_p10_mm": round(float(np.median(pf10)), 1) if pf10 else None,
                  "pf_p90_mm": round(float(np.median(pf90)), 1) if pf90 else None,
                  "pf_p99_mm": round(float(np.median(pf99)), 1) if pf99 else None,
                  "on_glass_pct": round(float(np.mean(gl)), 1) if gl else None,
                  "n_reps": len(meds)}
    return out


def main():
    per_rep = []
    for p in PARTICIPANTS:
        for stem in rep_stems(p, REPS_PER_P):
            r = run_one_rep(p, stem)
            if r is not None:
                per_rep.append(r)

    print(f"\n=== aggregated over {len(per_rep)} reps "
          f"({len({r['participant'] for r in per_rep})} participants) ===", flush=True)

    noise_agg = _agg(per_rep, "noise_sweep", SIGMAS)
    cam_agg = _agg(per_rep, "cam_sweep", NCAMS)
    noise_agg_r = _agg(per_rep, "noise_sweep_rts", SIGMAS)
    cam_agg_r = _agg(per_rep, "cam_sweep_rts", NCAMS)

    def show(title, knobs, agg, agg_r, label):
        print(f"\n{title} (median / worst-10% / worst-1% per-frame)", flush=True)
        print(f"{label:>8} | {'EKF (causal/online)':^26} | {'+RTS (offline smoother)':^26}", flush=True)
        print(f"{'':>8} | {'med':>7} {'p90':>8} {'p99':>9} | {'med':>7} {'p90':>8} {'p99':>9}", flush=True)
        for k in knobs:
            a, b = agg[k], agg_r[k]
            print(f"{k:>8} | {str(a['median_mm']):>7} {str(a['pf_p90_mm']):>8} {str(a['pf_p99_mm']):>9}"
                  f" | {str(b['median_mm']):>7} {str(b['pf_p90_mm']):>8} {str(b['pf_p99_mm']):>9}", flush=True)

    show("(a) pixel-noise budget", SIGMAS, noise_agg, noise_agg_r, "sigma_px")
    show("(b) #cameras (ref-seeded)", NCAMS, cam_agg, cam_agg_r, "n_cams")

    # --- (c) combined grid: median track err per (n_cams, sigma), and the
    #         noise tolerance sigma* (largest sigma whose median < cup radius) per N ---
    grid_agg = {nc: {s: float(np.median([r["grid"][nc][s] for r in per_rep
                                         if r["grid"][nc][s] is not None]))
                     for s in SIGMAS} for nc in NCAMS}

    def sigma_star(curve):
        """Largest sigma keeping median < cup radius; linear-interp the crossing."""
        prev_s, prev_v = None, None
        for s in SIGMAS:
            v = curve[s]
            if v is not None and v > CUP_R:
                if prev_s is None:
                    return 0.0
                # interpolate between prev (<=cup) and this (>cup)
                frac = (CUP_R - prev_v) / (v - prev_v) if v != prev_v else 0.0
                return round(prev_s + frac * (s - prev_s), 1)
            prev_s, prev_v = s, v
        return float(SIGMAS[-1])   # never crossed within swept range

    tol = {nc: sigma_star(grid_agg[nc]) for nc in NCAMS}

    print(f"\n(c) does the noise cliff move with #cameras? median track err (mm) per (N, sigma):", flush=True)
    print("n_cams\\sig " + " ".join(f"{s:>6}" for s in SIGMAS) + "   sigma*(cup-radius)", flush=True)
    for nc in NCAMS:
        cells = " ".join(f"{grid_agg[nc][s]:>6.0f}" for s in SIGMAS)
        print(f"{nc:>6}    {cells}   {tol[nc]:>6} px", flush=True)
    print(f"(sigma* = noise level where the median crosses the {CUP_R:.0f}mm cup radius; higher = more tolerant)", flush=True)

    OUT.write_text(json.dumps({
        "model": "pscale_4 student", "participants": PARTICIPANTS,
        "reps_per_participant": REPS_PER_P, "cup_radius_mm": CUP_R,
        "sigmas": SIGMAS, "ncams": NCAMS,
        "noise_agg": {str(k): v for k, v in noise_agg.items()},
        "cam_agg": {str(k): v for k, v in cam_agg.items()},
        "noise_agg_rts": {str(k): v for k, v in noise_agg_r.items()},
        "cam_agg_rts": {str(k): v for k, v in cam_agg_r.items()},
        "grid_agg": {str(nc): {str(s): grid_agg[nc][s] for s in SIGMAS} for nc in NCAMS},
        "sigma_star": {str(nc): tol[nc] for nc in NCAMS},
        "per_rep": per_rep,
    }, indent=2))
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
