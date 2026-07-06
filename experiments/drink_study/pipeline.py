"""drink_study end-to-end pipeline — the single entry point for the whole flow.

Read THIS file to understand the pipeline: raw clips -> detections -> 3D track ->
segmentation -> drink-dwell. Each stage is CACHE-FIRST: if the stage's artifact for a
rep already exists under cache/, it is reused and NO GPU/inference runs. A stage only
computes when its cache is cold, and it announces that before doing any heavy work.

    # inspect the whole DAG for one rep (all stages should report "cache")
    python experiments/drink_study/pipeline.py --rep P07_P07_drinking_left_20240124_142839

    # inspect a stage in isolation
    python experiments/drink_study/pipeline.py --rep <rep> --stage dwell

    # DAG summary across every rep that has a fused track
    python experiments/drink_study/pipeline.py --summary

The rep key is a fused-track stem, e.g. `P07_P07_drinking_left_20240124_142839__clean3d_refill`
(the `__clean3d_refill` suffix is optional on input — it is filled in from the cache).

Stages and their cached artifacts (VARIANT defaults to clean3d_refill, the reproject-fill
model that wins every axis — see docs/context/project_reproject_fill_filter.md):

  1 detections   cache/student_dets_<VARIANT>/<rep>__c0.25.json   (per-cam YOLO boxes)
  2 track3d      cache/track3d_<VARIANT>/<rep>.json                (consensus-KF 3D cup track)
                 cache/lopo_fused/<rep>.npz['fused']               (fused track used downstream)
  3 segment      lib/segment_cup_only.segment_cup_only(track)      (5 phases incl. drink dwell)
  4 dwell        cache/learn_seg_mouth.json  (proxy21 predicted dwell, ms error vs truth)
                 cache/mouth_dwell.json      (mocap cup->head-centroid TRUTH dwell)

This is orchestration only — the algorithms live in lib/. Nothing here re-runs the ~12h
GPU detection job; cold detection/track stages print how to populate them (cache_scripts/)
rather than silently launching hours of inference.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

# --- lib path shim (see _paths.py): make spine importable by bare name from anywhere ---
for _q in Path(__file__).resolve().parents:
    if (_q / "lib" / "segment_cup_only.py").exists():
        sys.path.insert(0, str(_q / "lib")); sys.path.insert(0, str(_q)); break
import numpy as np
from _paths import CACHE

VARIANT = "clean3d_refill"          # the shipped detection/track variant (reproject-fill)


def _norm_rep(rep: str) -> str:
    """Accept a rep with or without the __<VARIANT> suffix; return the canonical stem."""
    return rep if rep.endswith(f"__{VARIANT}") else f"{rep}__{VARIANT}"


# ---------------------------------------------------------------- stage inspectors
# Each returns (ok: bool, source: 'cache'|'cold', detail: str). Cache-first: they only
# READ. Computing a cold stage is deliberately gated behind an explicit message so we
# never trigger GPU inference as a side effect of "just inspecting the pipeline".

def stage_dets(rep: str):
    f = CACHE / f"student_dets_{VARIANT}" / f"{rep}__c0.25.json"
    if f.exists():
        n = len(json.loads(f.read_text()))
        return True, "cache", f"{f.name} ({n} frames)"
    return False, "cold", (f"missing {f.name} — populate via "
                           f"cache_scripts/cache_dets_model.py (GPU, ~hours for all reps)")


def stage_track(rep: str):
    tj = CACHE / f"track3d_{VARIANT}" / f"{rep}.json"
    npz = CACHE / "lopo_fused" / f"{rep}.npz"
    if npz.exists():
        fused = np.load(npz, allow_pickle=True)["fused"]
        vis = int(np.isfinite(np.asarray(fused, float)).all(1).sum())
        return True, "cache", f"lopo_fused/{npz.name} (fused {len(fused)}f, {vis} visible)"
    if tj.exists():
        return True, "cache", f"track3d_{VARIANT}/{tj.name} (no fused npz)"
    return False, "cold", (f"missing track — populate via "
                           f"lib/cache_track3d_consensus.py")


def stage_segment(rep: str):
    """Segment the fused track into phases (cheap, CPU) — computes live from the cached track."""
    npz = CACHE / "lopo_fused" / f"{rep}.npz"
    if not npz.exists():
        return False, "cold", "no fused track to segment (run stage 2 first)"
    import segment_cup_only as SC
    d = np.load(npz, allow_pickle=True)
    xyz = np.asarray(d["fused"], float)
    phases = SC.segment_cup_only(xyz)
    drinks = [p for p in phases if p[0] == "drink"] if phases else []
    return True, "computed", f"{len(phases)} phases, {len(drinks)} drink(s): {drinks}"


def stage_dwell(rep: str):
    seg = json.loads((CACHE / "learn_seg_mouth.json").read_text())
    row = seg.get("perrep", {}).get(rep)
    cols = seg.get("perrep_cols", [])
    out = []
    if row is not None:
        d = dict(zip(cols, row))
        out.append(f"proxy21 err {d.get('proxy21'):.0f}ms vs base17 {d.get('base17'):.0f}ms "
                   f"(truth=mocap cup->head)")
    else:
        out.append("rep not in learn_seg_mouth.json perrep")
    md = CACHE / "mouth_dwell.json"
    if md.exists():
        m = json.loads(md.read_text())
        rec = m.get(rep) if isinstance(m, dict) else None
        if isinstance(rec, dict) and "dwell_s" in rec:
            out.append(f"truth dwell {rec['dwell_s']:.2f}s")
    ok = row is not None
    return ok, "cache" if ok else "cold", "; ".join(out)


STAGES = [("dets", "1 detections", stage_dets),
          ("track", "2 3D track", stage_track),
          ("segment", "3 segmentation", stage_segment),
          ("dwell", "4 dwell", stage_dwell)]


def run_rep(rep: str, only: str | None):
    rep = _norm_rep(rep)
    print(f"\n=== pipeline · {rep} · variant={VARIANT} ===")
    all_ok = True
    for key, label, fn in STAGES:
        if only and key != only:
            continue
        try:
            ok, src, detail = fn(rep)
        except Exception as e:
            ok, src, detail = False, "error", f"{type(e).__name__}: {e}"
        mark = "OK " if ok else "-- "
        print(f"  {mark}[{src:>8}] {label:<16} {detail}")
        all_ok &= ok
    print("  " + ("all stages resolved from cache/compute" if all_ok
                  else "some stages cold — see messages above (no GPU was launched)"))
    return all_ok


def summary():
    """DAG coverage across every rep that has a fused track."""
    reps = sorted(p.stem for p in (CACHE / "lopo_fused").glob(f"*__{VARIANT}.npz"))
    print(f"=== pipeline coverage · {len(reps)} reps with fused track · variant={VARIANT} ===")
    seg = json.loads((CACHE / "learn_seg_mouth.json").read_text())
    have_dwell = set(seg.get("perrep", {}))
    ndet = sum((CACHE / f"student_dets_{VARIANT}" / f"{r}__c0.25.json").exists() for r in reps)
    ndw = sum(r in have_dwell for r in reps)
    print(f"  stage 1 detections : {ndet}/{len(reps)}")
    print(f"  stage 2 fused track: {len(reps)}/{len(reps)} (this is the rep set)")
    print(f"  stage 4 dwell score: {ndw}/{len(reps)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rep", help="fused-track stem (with or without __clean3d_refill suffix)")
    ap.add_argument("--stage", choices=[k for k, _, _ in STAGES],
                    help="inspect only this stage")
    ap.add_argument("--summary", action="store_true", help="DAG coverage across all reps")
    a = ap.parse_args()
    if a.summary:
        summary(); return
    if not a.rep:
        ap.error("give --rep <stem> (or --summary)")
    run_rep(a.rep, a.stage)


if __name__ == "__main__":
    main()
