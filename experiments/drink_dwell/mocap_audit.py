"""Audit the RAW mocap cup markers of a trial for gap-fill artifacts that would corrupt the
truth dwell — WITHOUT the despiking `centroid()` applies (we want to SEE the raw damage):

  JUMPS         a per-frame centroid step > jump_mm (non-physical teleport; real drink <~10mm/fr)
  STRAIGHT runs a run of near-constant velocity (|jerk|~0) longer than straight_min frames =
                QTM linearly INTERPOLATED across a marker dropout (fabricated straight-line motion)
  MISSING       raw NaN frames per cup marker (what QTM had to gap-fill)

The point: if a worst rep's cup→head distance is driven by a straight interpolated segment or a
teleport, the TRUTH is wrong (bad mocap), not the model. Prints per-marker + centroid findings,
and whether any artifact overlaps the truth dwell window.

    python experiments/drink_dwell/mocap_audit.py P070034          # one trial
    python experiments/drink_dwell/mocap_audit.py --worst 12       # audit the worst-N reps
"""
from __future__ import annotations
import sys as _s, pathlib as _p
_s.path.insert(0, str(_p.Path(__file__).resolve().parent))
import argparse, json
import numpy as np
from mocap import load_trial, CUP_MARKERS
from truth import dwell_truth

HERE = _p.Path(__file__).resolve().parent
_DS = HERE.parents[0] / "drink_study"
RESULTS = HERE / "cache" / "results.json"
ALIGN = _DS / "cache" / "qtm_align.json"

JUMP_MM = 50.0          # per-frame step above this = non-physical
STRAIGHT_MIN = 8        # a straight run this long (frames) = suspected interpolation
STRAIGHT_JERK = 0.5     # mm/fr^2: |2nd diff of position| below this = "straight"


def _runs(mask):
    out = []; i = 0; T = len(mask)
    while i < T:
        if mask[i]:
            j = i
            while j < T and mask[j]:
                j += 1
            out.append((i, j)); i = j
        else:
            i += 1
    return out


def straight_runs(xyz, min_len=STRAIGHT_MIN, jerk_thr=STRAIGHT_JERK):
    """Runs where the trajectory is (near-)linear: 2nd difference ~ 0 across all axes AND the
    marker is actually moving (>2mm/fr, so we don't flag a stationary rest as 'interpolated')."""
    v = np.diff(xyz, axis=0)                                  # (T-1,3) velocity
    jerk = np.linalg.norm(np.diff(v, axis=0), axis=1)         # (T-2,) |2nd diff|
    speed = np.linalg.norm(v, axis=1)[:-1]                    # (T-2,)
    straight = (jerk < jerk_thr) & (speed > 2.0)
    return [(s + 1, e + 1) for s, e in _runs(straight) if e - s >= min_len]


def jumps(xyz, thr=JUMP_MM):
    step = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    return [(int(i), float(step[i])) for i in np.where(step > thr)[0]]


def audit(c3d, dwell_span_mocap=None):
    tr = load_trial(c3d)
    idx = {l: i for i, l in enumerate(tr.labels)}
    print(f"\n=== {c3d}  rate={tr.rate:.0f}Hz  T={tr.n_frames} ===")
    # per cup marker (RAW, no despike)
    for mk in CUP_MARKERS:
        if mk not in idx:
            print(f"  {mk:6}: ABSENT"); continue
        raw = tr.markers[:, idx[mk]]
        miss = int(np.isnan(raw[:, 0]).sum())
        filled = raw.copy()
        for k in range(3):                                   # interp to expose interpolation shape
            col = filled[:, k]; g = ~np.isnan(col)
            if g.sum() > 2:
                filled[:, k] = np.interp(np.arange(len(col)), np.where(g)[0], col[g])
        jp = jumps(filled); st = straight_runs(filled)
        print(f"  {mk:6}: missing {miss:3}/{tr.n_frames}  jumps {len(jp)}  straight_runs {len(st)}"
              + (f"  {st[:4]}" if st else ""))
    # centroid (raw, NOT despiked) — this is what the truth uses (via cup_to_head)
    cup = tr.centroid(despike=False)
    jp = jumps(cup); st = straight_runs(cup)
    print(f"  CENTROID(raw): jumps {len(jp)}"
          + (f" @ {[(i, round(s)) for i, s in jp[:5]]}" if jp else "")
          + f"  straight_runs {len(st)}" + (f" {st[:5]}" if st else ""))
    # does any artifact fall inside the truth dwell window?
    dw = dwell_truth(tr)
    if dw.span:
        s, e = dw.span
        in_dwell = [(a, b) for a, b in st if b > s and a < e] + \
                   [(i, round(v)) for i, v in jp if s <= i <= e]
        flag = "  <-- ARTIFACT INSIDE DWELL WINDOW" if in_dwell else ""
        print(f"  truth dwell mocap-frames ({s},{e}) dur={dw.dur_s:.2f}s{flag}")
        if in_dwell:
            print(f"     overlapping: {in_dwell}")
    return tr, dw


def worst_c3ds(n):
    al = json.load(open(ALIGN)); m = {}
    for v in al.values():
        if isinstance(v, dict) and v.get("ok"):
            for r in v["reps"]:
                m[r["video"]] = r["c3d"]
    d = json.load(open(RESULTS)); ip = d["perrep_cols"].index("proxy21")
    rows = sorted(((v[ip], k) for k, v in d["perrep"].items()), reverse=True)[:n]
    return [(p, k, m.get(k)) for p, k in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trial", nargs="?", help="C3D stem, e.g. P070034")
    ap.add_argument("--worst", type=int, help="audit the worst-N proxy21 reps")
    a = ap.parse_args()
    if a.worst:
        rows = worst_c3ds(a.worst)
        print(f"auditing {len(rows)} worst reps' mocap for straight-lines / jumps ...")
        for err, video, c3d in rows:
            print(f"\n########## proxy21 err {err:.0f}ms  {video[:34]}  c3d={c3d} ##########")
            if c3d:
                audit(c3d)
    elif a.trial:
        audit(a.trial)
    else:
        ap.error("give a C3D stem or --worst N")


if __name__ == "__main__":
    main()
