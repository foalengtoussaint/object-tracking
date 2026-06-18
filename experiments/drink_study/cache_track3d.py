"""Cache the full 3D track for every CALIBRATED rep, reusing the cached pscale_4
detections (no GPU, no video). For each rep we replay the exact pipeline the
Rerun replay uses (viz_replay.run_pipeline): gated >=3-cam consensus -> causal
KalmanFilter3D (per-cam gate accept/reject) -> RTS smooth.

Per-rep output -> cache/track3d/{P}_{stem}__pscale_4.json:
  {
    "participant", "stem", "n_frames", "cams",
    "summary": {tri_rate, median_px, consensus_cov, kf_cov},
    "frames": [ {fr, consensus:[x,y,z]|null, kf:[x,y,z]|null, rts:[x,y,z]|null,
                 kept:[cam,...], median_px:float|null,
                 accept:{cam:bool}}  ... ]
  }
3D needs calibration.toml -> only participants with data/calib/<P>/ are processed
(currently P01/P06/P19/P23). Detections come from cache/student_dets/ (see
[[project_pscale4_full_det_cache]]). Verbose per-rep progress.

    python experiments/drink_study/cache_track3d.py            # all calibrated reps
    python experiments/drink_study/cache_track3d.py --participants P19
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np

import kf_accuracy as ka
from kalman_3d import load_calibration, triangulate_dlt, project
from viz_replay import run_pipeline      # consensus -> causal KF (+accept) -> RTS

ROOT = Path(__file__).resolve().parents[2]
DETCACHE = Path("experiments/drink_study/cache/student_dets")
OUTDIR = Path("experiments/drink_study/cache/track3d")
CALIB_DIR = ROOT / "data" / "calib"


def calibrated_participants() -> list[str]:
    if not CALIB_DIR.exists():
        return []
    return sorted(d.name for d in CALIB_DIR.iterdir()
                  if (d / "calibration.toml").exists())


def load_dets(cf: Path, calib: dict) -> dict:
    raw = json.loads(cf.read_text())
    return {c: [tuple(x) if x else None for x in v]
            for c, v in raw.items() if c in calib}


def kept_and_px(obs, calib, thr=ka.THR, minc=3):
    """Which cams survive the gate + the median reprojection px (consensus tightness)."""
    cur = dict(obs)
    while len(cur) >= 2:
        X = triangulate_dlt([calib[c] for c in cur], [np.array(cur[c]) for c in cur])
        e = {c: float(np.hypot(*(project(calib[c], X)[0] - np.array(cur[c])))) for c in cur}
        w = max(e, key=e.get)
        if e[w] <= thr:
            break
        del cur[w]
    if len(cur) < minc:
        return [], None
    X = triangulate_dlt([calib[c] for c in cur], [np.array(cur[c]) for c in cur])
    px = float(np.median([np.hypot(*(project(calib[c], X)[0] - np.array(cur[c]))) for c in cur]))
    return sorted(cur), px


def process_rep(p: str, stem: str, cf: Path, calib: dict) -> dict:
    dets = load_dets(cf, calib)
    cams = sorted(dets, key=lambda k: int(k.split("_")[1]))
    n = min(len(v) for v in dets.values())
    ka.calib, ka.DETS, ka.CAMS, ka.N = calib, dets, cams, n
    consensus, kf_est, kf_accept, rts_est = run_pipeline(calib, dets, cams, n)

    frames, pxs, tri = [], [], 0
    for fr in range(n):
        obs = {c: dets[c][fr] for c in cams if dets[c][fr] is not None}
        kept, px = kept_and_px(obs, calib) if len(obs) >= 2 else ([], None)
        if px is not None:
            tri += 1; pxs.append(px)
        frames.append({
            "fr": fr,
            "consensus": consensus[fr].tolist() if consensus[fr] is not None else None,
            "kf": kf_est[fr].tolist() if kf_est[fr] is not None else None,
            "rts": rts_est[fr].tolist() if rts_est[fr] is not None else None,
            "kept": kept,
            "median_px": px,
            "accept": {c: bool(v) for c, v in kf_accept[fr].items()},
        })
    summary = {
        "tri_rate": tri / n if n else 0.0,
        "median_px": float(np.median(pxs)) if pxs else None,
        "consensus_cov": sum(c is not None for c in consensus) / n if n else 0.0,
        "kf_cov": sum(e is not None for e in kf_est) / n if n else 0.0,
    }
    return {"participant": p, "stem": stem, "n_frames": n, "cams": cams,
            "summary": summary, "frames": frames}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", default=None, help="comma list (default: all calibrated)")
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    args = ap.parse_args()

    parts = calibrated_participants()
    if args.participants:
        want = set(args.participants.split(","))
        parts = [p for p in parts if p in want]
    if not parts:
        raise SystemExit("no calibrated participants found under data/calib/")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"calibrated participants: {parts}", flush=True)

    jobs = []
    for p in parts:
        for cf in sorted(DETCACHE.glob(f"{p}_*__pscale_4__c{ka.CONF}.json")):
            stem = cf.name[len(p) + 1:].replace(f"__pscale_4__c{ka.CONF}.json", "")
            jobs.append((p, stem, cf))
    print(f"{len(jobs)} calibrated reps to process", flush=True)

    t0, rows = time.time(), []
    for k, (p, stem, cf) in enumerate(jobs, 1):
        out = OUTDIR / f"{p}_{stem}__pscale_4.json"
        if out.exists() and not args.force:
            d = json.loads(out.read_text()); s = d["summary"]
            print(f"[{k}/{len(jobs)}] {p} {stem}  (cached) tri={s['tri_rate']:.3f} "
                  f"px={s['median_px']}", flush=True)
            rows.append((p, stem, s)); continue
        calib = load_calibration(f"data/calib/{p}/calibration.toml", target_size=ka.RES)
        res = process_rep(p, stem, cf, calib)
        out.write_text(json.dumps(res))
        s = res["summary"]
        print(f"[{k}/{len(jobs)}] {p} {stem}  n={res['n_frames']}  tri={s['tri_rate']:.3f} "
              f"px={s['median_px']:.2f}  cons_cov={s['consensus_cov']:.3f}  "
              f"-> {out.name}", flush=True)
        rows.append((p, stem, s))

    # cross-rep summary table
    summ = OUTDIR / "_summary.json"
    summ.write_text(json.dumps([{"participant": p, "stem": stem, **s} for p, stem, s in rows]))
    print(f"\n=== per-participant means ===", flush=True)
    for p in sorted({r[0] for r in rows}):
        ss = [s for (pp, _, s) in rows if pp == p]
        tr = np.mean([s["tri_rate"] for s in ss])
        px = np.mean([s["median_px"] for s in ss if s["median_px"] is not None])
        print(f"  {p}: {len(ss)} reps  tri_rate {tr:.3f}  median_px {px:.2f}", flush=True)
    print(f"wrote {summ}  ({len(rows)} reps in {time.time()-t0:.1f}s)", flush=True)
    print("TRACK3D_DONE", flush=True)


if __name__ == "__main__":
    main()
