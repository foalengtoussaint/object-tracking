"""Cohort comparison of the 2-cam gap-rescue: OLD (hard >=3-cam) vs SOFT (inflated R)
vs FULL (rescued 2-cam counts as much as >=3-cam consensus).

For every refill-detection rep of a calibrated participant:
  - run rescue_2cam.run() twice (soft R, full R) -> rts_old, rts_soft, rts_full
  - cup-only segment each -> does it find a drink dwell?
  - coverage = fraction of frames with consensus-or-rescue
  - SOFT-vs-FULL track divergence on rescued frames (large = full weight yanking)

Reports:
  - drink-phase recovery: #reps with a drink dwell under each mode
  - reps where rescue ADDS a drink dwell that OLD missed (the win)
  - mean coverage gain
  - reps with large soft-vs-full divergence (the safety tell for full weight)

    python experiments/drink_study/rescue_cohort.py
    python experiments/drink_study/rescue_cohort.py --participants P10,P24
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import kf_accuracy as ka
import segment_cup_only as sc
from kalman_3d import load_calibration
from cache_track3d import load_dets, calibrated_participants
from kf_consensus import R_MM
import rescue_2cam as R

DET = Path("experiments/drink_study/cache/student_dets_clean3d_refill")
FPS = 60.0


def to_arr(a):
    return np.array([x if x is not None else [np.nan] * 3 for x in a], float)


def drink_info(track):
    """Interp track -> cup-only segment -> (has_drink, drink_frames)."""
    xyz = to_arr(track) if isinstance(track, list) else track.copy()
    v = np.isfinite(xyz).all(1); idx = np.flatnonzero(v)
    if len(idx) < 10:
        return False, 0
    for ax in range(3):
        xyz[:, ax] = np.interp(np.arange(len(xyz)), idx, xyz[idx, ax])
    s = sc.segment_cup_only(xyz)
    df = sum(e - st for nm, st, e in s["intervals"] if nm == "drinking")
    return len(s["drink_runs"]) > 0, int(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", default=None)
    ap.add_argument("--div-thr", type=float, default=80.0,
                    help="soft-vs-full divergence (mm) above which to flag a rep")
    args = ap.parse_args()

    parts = calibrated_participants()
    if args.participants:
        want = set(args.participants.split(","))
        parts = [p for p in parts if p in want]

    jobs = []
    for p in parts:
        for cf in sorted(DET.glob(f"{p}_*__clean3d_refill__c0.25.json")):
            if "_summary" in cf.name:
                continue
            stem = cf.name[len(p) + 1:].replace("__clean3d_refill__c0.25.json", "")
            jobs.append((p, stem, cf))
    print(f"[cohort] {len(jobs)} reps over {len(parts)} participants", flush=True)

    rows = []; t0 = time.time(); calib_cache = {}
    for k, (p, stem, cf) in enumerate(jobs, 1):
        if p not in calib_cache:
            calib_cache[p] = load_calibration(f"data/calib/{p}/calibration.toml", target_size=ka.RES)
        calib = calib_cache[p]
        dets = load_dets(cf, calib)
        cams = sorted(dets, key=lambda c: int(c.split("_")[1]))
        n = min(len(v) for v in dets.values())
        if n < 30:
            continue
        soft = R.run(dets, cams, calib, n, r_rescue=R.R_RESCUE)
        full = R.run(dets, cams, calib, n, r_rescue=R_MM)

        cov_hard = int(np.isfinite(soft["cons"]).all(1).sum())
        nresc = len(soft["rescued"])
        d_old, f_old = drink_info(soft["rts_old"])
        d_soft, f_soft = drink_info(soft["rts_new"])
        d_full, f_full = drink_info(full["rts_new"])

        # soft-vs-full divergence on rescued frames
        so = to_arr(soft["rts_new"]); fu = to_arr(full["rts_new"])
        resc = [fr for fr, *_ in soft["rescued"]]
        if resc:
            dd = np.linalg.norm(fu[resc] - so[resc], axis=1)
            div_med, div_max = float(np.median(dd)), float(dd.max())
        else:
            div_med = div_max = 0.0

        rows.append(dict(name=f"{p}_{stem}", n=n, cov_hard=cov_hard, nresc=nresc,
                         cov_hard_pct=cov_hard / n, cov_resc_pct=(cov_hard + nresc) / n,
                         d_old=d_old, d_soft=d_soft, d_full=d_full,
                         f_old=f_old, f_soft=f_soft, f_full=f_full,
                         div_med=div_med, div_max=div_max))
        if k % 50 == 0 or k == len(jobs):
            print(f"[cohort][{k}/{len(jobs)}] {p} {stem}  resc={nresc} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    Rn = len(rows)
    drink = lambda key: sum(r[key] for r in rows)
    print(f"\n=== COHORT ({Rn} reps) ===", flush=True)
    print(f"  mean hard coverage:    {np.mean([r['cov_hard_pct'] for r in rows])*100:5.1f}%")
    print(f"  mean rescued coverage: {np.mean([r['cov_resc_pct'] for r in rows])*100:5.1f}%  "
          f"(+{np.mean([r['cov_resc_pct']-r['cov_hard_pct'] for r in rows])*100:.1f} pts)")
    print(f"  mean 2-cam rescues/rep: {np.mean([r['nresc'] for r in rows]):.1f}")
    print()
    print(f"  reps WITH a drink dwell:")
    print(f"    OLD (hard gate): {drink('d_old'):3d}/{Rn}")
    print(f"    SOFT rescue:     {drink('d_soft'):3d}/{Rn}")
    print(f"    FULL rescue:     {drink('d_full'):3d}/{Rn}")
    # recoveries: drink appears that OLD missed
    rec_soft = [r for r in rows if r["d_soft"] and not r["d_old"]]
    rec_full = [r for r in rows if r["d_full"] and not r["d_old"]]
    lost_soft = [r for r in rows if r["d_old"] and not r["d_soft"]]
    print()
    print(f"  drink dwell RECOVERED by rescue (OLD missed -> rescue found):")
    print(f"    SOFT: {len(rec_soft)}   FULL: {len(rec_full)}")
    print(f"  drink dwell LOST by soft rescue (OLD had -> soft missed): {len(lost_soft)}")
    print()
    flag = sorted([r for r in rows if r["div_max"] >= args.div_thr],
                  key=lambda r: -r["div_max"])
    print(f"  reps where FULL weight yanks the track (soft-vs-full max >= {args.div_thr}mm): {len(flag)}")
    for r in flag[:20]:
        print(f"    {r['name']:44s} div med={r['div_med']:4.0f} max={r['div_max']:4.0f}mm  "
              f"resc={r['nresc']:2d}  drink old/soft/full={int(r['d_old'])}/{int(r['d_soft'])}/{int(r['d_full'])}")
    print()
    print(f"  reps recovered by SOFT (first 25):")
    for r in rec_soft[:25]:
        print(f"    {r['name']:44s} f_soft={r['f_soft']:3d} f_full={r['f_full']:3d} resc={r['nresc']}")

    out = Path("experiments/drink_study/cache/rescue_cohort.json")
    out.write_text(json.dumps(rows, indent=1, default=float))
    print(f"\n  wrote {out}  ({time.time()-t0:.0f}s total)")
    print("RESCUE_COHORT_DONE", flush=True)


if __name__ == "__main__":
    main()
