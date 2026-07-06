"""Reconcile the two independent phase segmenters into one phase track.

  CUP-ONLY (segment_cup_only on our refill+rescue 3D cup track): clean on the
      transport phases (the cup is the end-effector), but blind at the drink dwell
      because the cup is occluded at the mouth (interpolated/rescued, see the
      apex churn that flips the drinking gate).
  POSE  (container segment_drink_task, cached in biomech_*.npz phase_intervals):
      its drink phase is "hand stationary NEAR MOUTH" with near-mouth relative to
      each recording's min hand-mouth distance -> occlusion-robust exactly where
      the cup fails. It is already a cup+pose fusion.

Reconciliation rule (per trial, on a common timeline):
  * map pose's 7 phases -> the 5 cup phases (reaching->rest_pre, returning->rest_post)
  * DRINK dwell: trust POSE (hand-near-mouth beats interpolated cup). Take pose's
    drinking interval.
  * TRANSPORT/REST boundaries: trust CUP where the two agree within TOL frames;
    where they disagree, keep cup's onset/offset (clean end-effector motion) but
    FLAG it.
  * Report per-phase boundary deltas + overall agreement (IoU of the drinking
    interval, frame-wise phase agreement).

    python experiments/drink_study/reconcile_phases.py            # all 4 cached trials
    python experiments/drink_study/reconcile_phases.py P23_P23_drinking_right_20240716_151359
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import segment_cup_only as sc

CACHE = Path("experiments/drink_study/cache")
TRACK = CACHE / "track3d_clean3d_refill"
FPS = 60.0
TOL = 6                                     # frames (~0.1s) within which boundaries "agree"
PHASES5 = ["rest_pre", "forward_transport", "drinking", "back_transport", "rest_post"]
POSE2CUP = {"rest_pre": "rest_pre", "reaching": "rest_pre",
            "forward_transport": "forward_transport", "drinking": "drinking",
            "back_transport": "back_transport", "returning": "rest_post",
            "rest_post": "rest_post"}


def cup_intervals(trial):
    """segment_cup_only on the rescued refill track (with 2-cam rescue if present)."""
    rf = CACHE / f"rescue2cam_{trial}_full.json"
    if rf.exists():
        rts = json.loads(rf.read_text())["rts_new"]
    else:
        d = json.loads((TRACK / f"{trial}__clean3d_refill.json").read_text())
        rts = [f["rts"] for f in d["frames"]]
    xyz = np.array([p if p else [np.nan] * 3 for p in rts], float)
    v = np.isfinite(xyz).all(1); idx = np.flatnonzero(v)
    for a in range(3):
        xyz[:, a] = np.interp(np.arange(len(xyz)), idx, xyz[idx, a])
    return sc.segment_cup_only(xyz)["intervals"], len(xyz)


def pose_intervals(trial):
    """pose phase_intervals from the cached biomech npz, collapsed to the 5 cup phases."""
    b = np.load(CACHE / f"biomech_{trial}.npz", allow_pickle=True)
    raw = [(POSE2CUP[n], int(s), int(e)) for n, s, e in b["phase_intervals"]]
    # merge adjacent same-name (reaching+rest_pre etc.)
    merged = []
    for n, s, e in raw:
        if merged and merged[-1][0] == n and merged[-1][2] == s:
            merged[-1] = (n, merged[-1][1], e)
        else:
            merged.append((n, s, e))
    return merged, int(b["phase_intervals"][-1][2])


def phase_array(intervals, T):
    a = np.array(["rest_pre"] * T, dtype=object)
    for n, s, e in intervals:
        a[s:min(e, T)] = n
    return a


def span(intervals, name):
    runs = [(s, e) for n, s, e in intervals if n == name]
    if not runs:
        return None
    return runs[0][0], runs[-1][1]


def iou(a, b):
    if a is None or b is None:
        return 0.0 if (a or b) else 1.0
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union else 0.0


def reconcile(trial):
    cup_iv, Tc = cup_intervals(trial)
    pose_iv, Tp = pose_intervals(trial)
    T = min(Tc, Tp)

    cup_drink = span(cup_iv, "drinking")
    pose_drink = span(pose_iv, "drinking")
    cup_grasp = (span(cup_iv, "forward_transport") or (None, None))[0], \
                (span(cup_iv, "back_transport") or span(cup_iv, "forward_transport") or (None, None))[1]

    # === reconciled timeline ===
    # drink dwell from POSE (occlusion-robust); transport onset/offset from CUP.
    # onset = first cup motion (start of cup forward_transport); offset = end of cup back/forward
    fwd = span(cup_iv, "forward_transport"); back = span(cup_iv, "back_transport")
    onset = fwd[0] if fwd else (pose_drink[0] if pose_drink else 0)
    offset = (back[1] if back else (fwd[1] if fwd else (pose_drink[1] if pose_drink else T)))
    drink = pose_drink or cup_drink
    rec = []
    if onset > 0:
        rec.append(("rest_pre", 0, onset))
    if drink and drink[0] > onset:
        rec.append(("forward_transport", onset, drink[0]))
        rec.append(("drinking", drink[0], min(drink[1], offset) if offset else drink[1]))
        d_end = min(drink[1], offset) if offset else drink[1]
        if offset and offset > d_end:
            rec.append(("back_transport", d_end, offset))
    else:
        rec.append(("forward_transport", onset, offset or T))
    if offset and offset < T:
        rec.append(("rest_post", offset, T))

    # === agreement diagnostics ===
    ca = phase_array(cup_iv, T); pa = phase_array(pose_iv, T)
    frame_agree = float(np.mean(ca == pa))
    diag = dict(
        trial=trial, T=T,
        cup_drink=cup_drink, pose_drink=pose_drink, drink_iou=round(iou(cup_drink, pose_drink), 2),
        cup_has_drink=cup_drink is not None, pose_has_drink=pose_drink is not None,
        onset_cup=onset, frame_agree=round(frame_agree, 2),
        reconciled=[(n, s, e) for n, s, e in rec],
    )
    return diag


def fmt(s):
    return f"{s/FPS:.2f}s" if s is not None else "--"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trials", nargs="*")
    args = ap.parse_args()
    if args.trials:
        trials = args.trials
    else:
        trials = [Path(f).name[len("biomech_"):-4]
                  for f in sorted(glob.glob(str(CACHE / "biomech_*.npz")))]
    print(f"reconciling {len(trials)} trials (drink dwell from POSE, transport from CUP)\n")
    rows = []
    for t in trials:
        try:
            d = reconcile(t)
        except FileNotFoundError as e:
            print(f"  {t}: SKIP ({e})"); continue
        rows.append(d)
        cd, pd = d["cup_drink"], d["pose_drink"]
        print(f"{t}")
        print(f"   cup drink:  {fmt(cd[0]) if cd else '--':>6} - {fmt(cd[1]) if cd else '--':>6}   "
              f"pose drink: {fmt(pd[0]) if pd else '--':>6} - {fmt(pd[1]) if pd else '--':>6}   "
              f"IoU={d['drink_iou']}")
        print(f"   frame-wise phase agreement: {d['frame_agree']*100:.0f}%")
        print(f"   reconciled: " + " | ".join(f"{n}:{fmt(s)}-{fmt(e)}" for n, s, e in d["reconciled"]))
        print()
    # summary
    di = [r["drink_iou"] for r in rows]
    print(f"=== {len(rows)} trials ===")
    print(f"  cup found drink:  {sum(r['cup_has_drink'] for r in rows)}/{len(rows)}   "
          f"pose found drink: {sum(r['pose_has_drink'] for r in rows)}/{len(rows)}")
    print(f"  median drink IoU (cup vs pose): {np.median(di):.2f}")
    print(f"  mean frame-wise phase agreement: {np.mean([r['frame_agree'] for r in rows])*100:.0f}%")
    out = CACHE / "reconcile_phases.json"
    out.write_text(json.dumps(rows, indent=1, default=str))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
