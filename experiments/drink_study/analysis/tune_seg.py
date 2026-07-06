"""LOPO parameter search for the cup-only phase segmenter (drink dwell).

Truth = the mocap cup track segmented by the SAME geometric segmenter (mocap is
the same signal family as our video cup, just sub-mm clean). The objective is to
pick segmenter params so the VIDEO-track drink span matches the MOCAP-track drink
span. Cache-only, NO GPU, NO retraining -- reloads cache/lopo_fused/<video>.npz.

Why param search can help (measured, tune_seg motivation):
  in-dwell median cup speed  true 45   fused 59   kf 72  (mm/s)
  in-dwell frames above the DRINK_SPEED=120 gate: true 0.2%  fused 14%  kf 26%
So the video track floats faster in the dwell and the 120 gate wrongly cuts those
frames -> dwell UNDER-estimate. Truth is insensitive to the gate (0.2% either way),
so raising the gate recovers real dwell frames without moving the truth. But a too-
high gate lets transport edges in (boundary error), so we search jointly, LOPO.

Scored on the `--track` variant (default fused, the segmentation winner). Reports
per-fold best params by cross-validated |dur|err so no single-split winner sneaks in.

    python experiments/drink_study/tune_seg.py [--track fused|hard_kf|kf]
Run from repo root. ~seconds. Writes cache/tune_seg.json.
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import argparse, glob, itertools, json, time
from pathlib import Path
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "drink_study"))
import segment_cup_only as S

from _paths import CACHE as _C
CACHE = _C / "lopo_fused"
HZ = 60.0

# search grid -- centred on the current defaults (see segment_cup_only.py):
#   FWD_ON 15, BACK_OFF 10, DRINK_SPEED 120, DRINK_DISP_PAD 90, BUTTER_HZ 6
GRID = dict(
    drink_speed=[130, 150, 170, 190, 220],
    drink_disp_pad=[120, 150, 180, 220],
    fwd_on=[15, 20],
    back_off=[12, 14, 18],
    butter_hz=[5.0, 6.0, 8.0],
    min_phase=[5, 8],
)
DEFAULT = dict(drink_speed=120, drink_disp_pad=90, fwd_on=15, back_off=10)


def span(xyz, **kw):
    r = S.segment_cup_only(xyz, fps=HZ, **kw)
    runs = r["drink_runs"]
    return (runs[0][0], runs[-1][1]) if runs else None


def load_reps():
    reps = []
    for f in sorted(glob.glob(str(CACHE / "*.npz"))):
        d = np.load(f, allow_pickle=True)
        reps.append(dict(pid=str(d["pid"]), video=str(d["video"]),
                         true=d["true"], kf=d["kf"], hard_kf=d["hard_kf"],
                         fused=d["fused"]))
    return reps


def score(reps, track, params):
    """Return (mean|dur|err_ms, mean_boundary_ms, mean_bias_ms) over reps with a truth dwell."""
    de, bd = [], []
    for r in reps:
        tsp = span(r["true"])                      # truth uses DEFAULT gate (fixed)
        if tsp is None:
            continue
        vsp = span(r[track], **params)
        if vsp is None:
            de.append((tsp[1] - tsp[0]) / HZ * 1000)   # miss = whole dwell missed
            bd.append((tsp[1] - tsp[0]) / HZ * 1000)
            continue
        td = (tsp[1] - tsp[0]) / HZ * 1000
        vd = (vsp[1] - vsp[0]) / HZ * 1000
        de.append(vd - td)
        bd.append((abs(vsp[0] - tsp[0]) + abs(vsp[1] - tsp[1])) / 2 / HZ * 1000)
    de = np.array(de); bd = np.array(bd)
    return float(np.abs(de).mean()), float(bd.mean()), float(de.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="fused", choices=["fused", "hard_kf", "kf"])
    args = ap.parse_args()

    reps = load_reps()
    pids = sorted({r["pid"] for r in reps})
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    print(f"{len(reps)} reps, {len(pids)} participants, {len(combos)} param combos, "
          f"track={args.track}", flush=True)

    d_abs, d_bnd, d_bias = score(reps, args.track, DEFAULT)
    print(f"DEFAULT ({DEFAULT}): |dur|={d_abs:.0f}ms  boundary={d_bnd:.0f}ms  "
          f"bias={d_bias:+.0f}ms", flush=True)

    # precompute per-combo per-participant scores once (grid x pid), then LOPO-select
    t0 = time.time()
    # global scores of every combo (for the honest pooled optimum)
    combo_abs = np.zeros(len(combos))
    per_pid = {p: np.zeros(len(combos)) for p in pids}
    for ci, params in enumerate(combos):
        # score split by participant so we can both pool and hold out
        de_by = {p: [] for p in pids}
        for r in reps:
            tsp = span(r["true"])
            if tsp is None:
                continue
            vsp = span(r[args.track], **params)
            td = (tsp[1] - tsp[0]) / HZ * 1000
            vd = (vsp[1] - vsp[0]) / HZ * 1000 if vsp else 0.0
            de_by[r["pid"]].append(abs(vd - td) if vsp else td)
        allde = [x for p in pids for x in de_by[p]]
        combo_abs[ci] = np.mean(allde)
        for p in pids:
            per_pid[p][ci] = np.mean(de_by[p]) if de_by[p] else np.nan
        if (ci + 1) % 40 == 0 or ci == len(combos) - 1:
            print(f"  scored {ci+1}/{len(combos)} combos ({time.time()-t0:.0f}s)", flush=True)

    best_pool = int(np.argmin(combo_abs))
    print(f"\nPOOLED best: {combos[best_pool]}  |dur|={combo_abs[best_pool]:.0f}ms", flush=True)

    # LOPO: for each held-out pid, pick the combo that's best on the OTHER pids,
    # then evaluate it on the held-out pid -> honest cross-validated |dur|.
    held_abs = []
    picks = {}
    for held in pids:
        trn_scores = np.zeros(len(combos))
        for ci in range(len(combos)):
            vals = [per_pid[p][ci] for p in pids if p != held and not np.isnan(per_pid[p][ci])]
            trn_scores[ci] = np.mean(vals)
        bi = int(np.argmin(trn_scores))
        picks[held] = combos[bi]
        if not np.isnan(per_pid[held][bi]):
            held_abs.append(per_pid[held][bi])
    lopo_abs = float(np.mean(held_abs))
    # how stable is the pick across folds?
    from collections import Counter
    key = lambda c: tuple(sorted(c.items()))
    cnt = Counter(key(p) for p in picks.values())
    top_pick, top_n = cnt.most_common(1)[0]
    print(f"LOPO cross-validated |dur|={lopo_abs:.0f}ms  "
          f"(most-common fold pick chosen {top_n}/{len(pids)} folds)", flush=True)
    print(f"most-common pick: {dict(top_pick)}", flush=True)

    # final reported params = the pooled optimum; report its full metrics
    fa, fb, fbias = score(reps, args.track, combos[best_pool])
    print(f"\nPOOLED-best full metrics: |dur|={fa:.0f}ms boundary={fb:.0f}ms bias={fbias:+.0f}ms", flush=True)

    out = dict(track=args.track, n=len(reps),
               default=DEFAULT, default_metrics=dict(abs=d_abs, boundary=d_bnd, bias=d_bias),
               pooled_best=combos[best_pool],
               pooled_best_metrics=dict(abs=fa, boundary=fb, bias=fbias),
               lopo_abs=lopo_abs, most_common_pick=dict(top_pick), fold_agreement=f"{top_n}/{len(pids)}")
    json.dump(out, open(_C / "tune_seg.json", "w"), indent=2)
    print(f"wrote cache/tune_seg.json", flush=True)


if __name__ == "__main__":
    main()
