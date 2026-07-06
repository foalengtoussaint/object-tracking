"""Rebuild 3D cup tracks using the CONSENSUS-ANCHORED KF (no 2D-gated EKF).

Same gated >=3-cam consensus per frame as cache_track3d.py, but instead of the
2D-gated causal EKF (which diverges, see project_consensus_anchored_kf) we feed the
consensus as a 3D measurement to a no-gate linear KF + RTS (kf_consensus). Cannot
diverge, still smooths/interpolates gaps with the dynamics.

Reads detections for a given model from cache/student_dets_<tag>/ and writes
cache/track3d_<tag>/{P}_{stem}__<tag>.json (same schema as cache_track3d: per-frame
consensus/kf/rts/kept/median_px + summary). No GPU.

    python experiments/drink_study/cache_track3d_consensus.py --tag clean3d_fill
    python experiments/drink_study/cache_track3d_consensus.py --tag clean3d_refill
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np

import kf_accuracy as ka
from kalman_3d import load_calibration, triangulate_dlt, project
from kf_consensus import kf_rts_on_consensus
from cache_track3d import load_dets, kept_and_px, calibrated_participants

ROOT = Path(__file__).resolve().parents[2]


def process_rep(p, stem, cf, calib, tag):
    dets = load_dets(cf, calib)
    cams = sorted(dets, key=lambda k: int(k.split("_")[1]))
    n = min(len(v) for v in dets.values())
    ka.calib, ka.DETS, ka.CAMS, ka.N = calib, dets, cams, n

    # gated >=3-cam consensus per frame (same as cache_track3d / run_pipeline)
    consensus = [ka.gated_consensus({c: dets[c][fr] for c in cams
                                     if dets[c][fr] is not None}) for fr in range(n)]
    cons_arr = np.array([c if c is not None else [np.nan] * 3 for c in consensus], float)

    # consensus-anchored KF (causal) + RTS (smoothed) — cannot diverge
    if np.isfinite(cons_arr).all(1).sum() >= 2:
        kf_arr, rts_arr = kf_rts_on_consensus(cons_arr, fps=ka.FPS, return_both=True)
    else:
        kf_arr = rts_arr = np.full((n, 3), np.nan)

    frames, pxs, tri = [], [], 0
    for fr in range(n):
        obs = {c: dets[c][fr] for c in cams if dets[c][fr] is not None}
        kept, px = kept_and_px(obs, calib) if len(obs) >= 2 else ([], None)
        if px is not None:
            tri += 1; pxs.append(px)
        frames.append({
            "fr": fr,
            "consensus": consensus[fr].tolist() if consensus[fr] is not None else None,
            "kf": kf_arr[fr].tolist() if np.isfinite(kf_arr[fr]).all() else None,
            "rts": rts_arr[fr].tolist() if np.isfinite(rts_arr[fr]).all() else None,
            "kept": kept,
            "median_px": px,
            "accept": {},          # no per-camera gating in the anchored filter
        })
    summary = {
        "tri_rate": tri / n if n else 0.0,
        "median_px": float(np.median(pxs)) if pxs else None,
        "consensus_cov": sum(c is not None for c in consensus) / n if n else 0.0,
        "kf_cov": sum(np.isfinite(x).all() for x in kf_arr) / n if n else 0.0,
    }
    return {"participant": p, "stem": stem, "n_frames": n, "cams": cams,
            "summary": summary, "frames": frames}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="clean3d_fill | clean3d_refill | pscale_4 ...")
    ap.add_argument("--conf", default="0.25")
    ap.add_argument("--participants", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    detdir = Path(f"experiments/drink_study/cache/student_dets_{args.tag}")
    if not detdir.exists():          # pscale_4 lives in the bare student_dets/
        detdir = Path("experiments/drink_study/cache/student_dets")
    outdir = Path(f"experiments/drink_study/cache/track3d_{args.tag}")
    outdir.mkdir(parents=True, exist_ok=True)

    parts = calibrated_participants()
    if args.participants:
        want = set(args.participants.split(","))
        parts = [p for p in parts if p in want]
    print(f"[{args.tag}] consensus-anchored KF | det={detdir} | calibrated: {parts}", flush=True)

    jobs = []
    for p in parts:
        for cf in sorted(detdir.glob(f"{p}_*__{args.tag}__c{args.conf}.json")):
            stem = cf.name[len(p) + 1:].replace(f"__{args.tag}__c{args.conf}.json", "")
            jobs.append((p, stem, cf))
    print(f"[{args.tag}] {len(jobs)} reps to process", flush=True)

    t0, rows = time.time(), []
    for k, (p, stem, cf) in enumerate(jobs, 1):
        out = outdir / f"{p}_{stem}__{args.tag}.json"
        if out.exists() and not args.force:
            d = json.loads(out.read_text()); rows.append((p, stem, d["summary"])); continue
        calib = load_calibration(f"data/calib/{p}/calibration.toml", target_size=ka.RES)
        res = process_rep(p, stem, cf, calib, args.tag)
        out.write_text(json.dumps(res)); s = res["summary"]
        rows.append((p, stem, s))
        if k % 50 == 0 or k == len(jobs):
            print(f"[{args.tag}][{k}/{len(jobs)}] {p} {stem} tri={s['tri_rate']:.3f} "
                  f"px={s['median_px']:.2f} ({time.time()-t0:.0f}s)", flush=True)

    summ = outdir / "_summary.json"
    summ.write_text(json.dumps([{"participant": p, "stem": s, **ss} for p, s, ss in rows]))
    print(f"\n[{args.tag}] === per-participant means ===", flush=True)
    for p in sorted({r[0] for r in rows}):
        ss = [s for (pp, _, s) in rows if pp == p]
        tr = np.mean([s["tri_rate"] for s in ss])
        px = np.mean([s["median_px"] for s in ss if s["median_px"] is not None])
        print(f"  {p}: {len(ss)} reps  tri_rate {tr:.3f}  median_px {px:.2f}", flush=True)
    print(f"[{args.tag}] wrote {len(rows)} reps in {time.time()-t0:.1f}s", flush=True)
    print("TRACK3D_CONSENSUS_DONE", flush=True)


if __name__ == "__main__":
    main()
