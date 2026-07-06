"""Acceptance protocol: decide whether a model is good enough to trust.

The robustness study showed the failures that matter are NOT what detector
metrics (mAP/recall) measure: the catastrophic cases had det~.92 and tri=1.0 on
all 10 cams yet the FUSED track was >1m wrong (confident-and-wrong). So we judge
the model the way it's deployed -- through the gating->KF pipeline -- and we
specifically hunt the confident-wrong mode.

Four gates, run as a funnel (cheap no-GT checks first):

  GATE 0  PRECONDITION  (no GT)  -- the one hard floor from the envelope study:
          >=3 cameras must land in the POST-GATE consensus inlier set per frame.
          This is the corrected agreement metric (measured on the cameras the
          >=3-cam gate actually keeps, NOT the raw set -- that was the metric bug).
  GATE 1  DETECTOR SANITY  (labels)  -- per-cam recall/precision; necessary not
          sufficient. Flags cam_10-glass memorization (the documented distractor).
  GATE 2  FUSED-TRACK ACCURACY  (needs GT)  -- run gating->KF->RTS, score p99
          drift vs frozen truth; pass = p99<cup_radius AND coverage>0.5.
          *** PENDING: ground-truth source undecided (see --help). Stubbed. ***
  GATE 3  CONFIDENT-WRONG AUDIT  (no GT here; uses post-gate agreement as proxy)
          -- split would-be failures into graceful vs catastrophic and require a
          runtime guard that flags the catastrophic ones.

Usage:
    .venv/bin/python experiments/drink_study/acceptance.py P01 P01_drinking_right_20231220_141546
    # add --thresholds to print the pass bars; --json to dump the report.

Runs from the cached student dets (no GPU). Gate 2 currently reports PENDING.
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from kalman_3d import load_calibration, triangulate_dlt, project
import kf_accuracy as ka

RES = (1920, 1080)
THR = ka.THR          # 30px consensus inlier gate
CUP_R = ka.CUP_R      # 35mm usable-track radius
GLASS_CAM = "cam_10"  # documented static-distractor cup (see project_cam10_apparent_size)

# --- pass bars (one place to tune the protocol) ---
BARS = {
    "g0_min_inlier_cams": 3,        # the hard floor
    "g0_frac_frames_ok": 0.95,      # >=this fraction of detected frames must hit it
    "g1_min_recall": 0.5,           # per-cam recall floor (necessary, not sufficient)
    "g1_max_glass_eject_rate": 0.5,  # if cam_10 is ejected from consensus more than
                                     # this often (despite high coverage) -> it's
                                     # tracking the static glass, not the cup
    "g3_max_lowagree_streak_s": 0.5,  # the KF coasts through brief <3-cam gaps; what
                                      # hurts is a SUSTAINED low-agreement run. Ceiling
                                      # in seconds (at FPS) before drift is a real risk.
}


def post_gate_inliers(obs, calib, thr=THR, minc=3):
    """Replicate kf_accuracy.gated_consensus' ejection loop, but RETURN the set of
    cameras that survive (the real inlier set), not just the point.

    Iteratively drops the worst-reprojecting camera until all kept cams are within
    `thr` px of their common triangulation. This is the corrected agreement metric.
    """
    cur = dict(obs)
    while len(cur) >= 2:
        X = triangulate_dlt([calib[c] for c in cur], [np.array(cur[c]) for c in cur])
        e = {c: float(np.hypot(*(project(calib[c], X)[0] - np.array(cur[c])))) for c in cur}
        w = max(e, key=e.get)
        if e[w] <= thr:
            break
        del cur[w]
    return set(cur) if len(cur) >= minc else set()


def gate0_precondition(dets, calib, cams):
    """The one hard floor: per frame, how many cams land in the post-gate inlier
    set. Fully no-GT. Returns the corrected agreement distribution + pass/fail."""
    n = min(len(v) for v in dets.values())
    counts = []
    for fr in range(n):
        obs = {c: dets[c][fr] for c in cams if dets[c][fr] is not None}
        if len(obs) < 2:
            continue
        counts.append(len(post_gate_inliers(obs, calib)))
    counts = np.array(counts) if counts else np.array([0])
    frac_ok = float(np.mean(counts >= BARS["g0_min_inlier_cams"]))
    return {
        "gate": "0 precondition (>=3 post-gate agreeing cams)",
        "median_inlier_cams": int(np.median(counts)),
        "frac_frames_>=3_inliers": round(frac_ok, 3),
        "n_triangulable_frames": int(len(counts)),
        "pass": bool(frac_ok >= BARS["g0_frac_frames_ok"]
                     and np.median(counts) >= BARS["g0_min_inlier_cams"]),
        "note": "CORRECTED agreement metric: counts cams the >=3-cam gate KEEPS, "
                "not raw dets (the raw-set metric was non-discriminative).",
    }


def gate1_detector_sanity(dets, calib, cams):
    """Per-cam coverage + a cam_10-glass memorization flag.

    Coverage alone CAN'T tell "memorized the static glass" from "genuinely sees
    the cup" -- both give high coverage. The real glass signature
    (project_cam10_apparent_size) is that cam_10's detection DISAGREES in 3D, so
    the consensus keeps EJECTING it. We measure: of the frames where cam_10
    detects AND a >=3-cam consensus exists, how often is cam_10 NOT in the kept
    inlier set. High eject-rate despite high coverage = tracking the glass."""
    n = min(len(v) for v in dets.values())
    cov = {c: round(sum(x is not None for x in dets[c]) / n, 3) for c in cams}
    seen = ejected = 0
    if GLASS_CAM in cams:
        for fr in range(n):
            if dets[GLASS_CAM][fr] is None:
                continue
            obs = {c: dets[c][fr] for c in cams if dets[c][fr] is not None}
            keep = post_gate_inliers(obs, calib)
            if not keep:                      # no >=3 consensus this frame -> skip
                continue
            seen += 1
            if GLASS_CAM not in keep:
                ejected += 1
    eject_rate = round(ejected / seen, 3) if seen else 0.0
    glass_suspect = (cov.get(GLASS_CAM, 0) >= 0.9
                     and eject_rate > BARS["g1_max_glass_eject_rate"])
    return {
        "gate": "1 detector sanity (coverage + 3D-disagreement glass flag)",
        "per_cam_coverage": cov,
        "min_cam_coverage": round(min(cov.values()), 3),
        "cam_10_eject_rate": eject_rate,
        "cam_10_glass_suspect": bool(glass_suspect),
        "pass": bool(min(cov.values()) >= BARS["g1_min_recall"] and not glass_suspect),
        "note": "NECESSARY, NOT SUFFICIENT -- detector metrics pass even the "
                "confident-wrong failures. Glass flag now uses 3D consensus "
                "ejection, not raw coverage (high coverage != tracking the cup).",
    }


def gate2_fused_accuracy():
    """Run gating->KF->RTS, score p99 drift vs frozen truth. PENDING: the
    ground-truth source is undecided (self-track is circular). Stubbed until then."""
    return {
        "gate": "2 fused-track accuracy vs ground truth",
        "pass": None,
        "status": "PENDING",
        "note": "Ground-truth source undecided. Options: (a) independent "
                "hand-verified held-out trials [strongest], (b) pscale_4 self-track "
                "[circular], (c) cross-model+geometry consensus [blind on hard "
                "frames]. Wire up once chosen; pass = p99<35mm AND coverage>0.5.",
    }


def gate3_confident_wrong(dets, calib, cams):
    """Confident-wrong audit WITHOUT GT: use the post-gate inlier count as the
    runtime guard signal. A frame is a 'confident-wrong RISK' when a 2-cam (or
    fewer) consensus still triangulates SOMETHING (the pipeline would emit a
    position) but <3 cams actually agree -- the exact signature of the catastrophic
    family. The guard = suppress/flag any frame with <3 post-gate inliers."""
    n = min(len(v) for v in dets.values())
    emitted = risk = 0
    for fr in range(n):
        obs = {c: dets[c][fr] for c in cams if dets[c][fr] is not None}
        if len(obs) < 2:
            continue
        emitted += 1                                   # pipeline would output a point
        if len(post_gate_inliers(obs, calib)) < 3:     # but <3 agree -> confident-wrong risk
            risk += 1
    rate = round(risk / emitted, 3) if emitted else 0.0
    return {
        "gate": "3 confident-wrong audit + runtime guard",
        "frames_emitting_a_position": int(emitted),
        "confident_wrong_risk_frames": int(risk),
        "confident_wrong_risk_rate": rate,
        "guard": "suppress any frame with <3 post-gate-agreeing cameras",
        "pass": bool(rate <= BARS["g3_max_catastrophic_rate"]),
        "note": "The guard is the corrected agreement metric used as a runtime "
                "veto. No cheap raw-det threshold separated good/bad (false_positives.py); "
                "the post-gate inlier count does.",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("participant", help="e.g. P01")
    ap.add_argument("stem", help="clip stem, e.g. P01_drinking_right_20231220_141546")
    ap.add_argument("--json", metavar="PATH", default=None, help="dump full report")
    ap.add_argument("--thresholds", action="store_true", help="print the pass bars")
    args = ap.parse_args()

    if args.thresholds:
        print("PASS BARS:", json.dumps(BARS, indent=2))

    calib = load_calibration(f"data/calib/{args.participant}/calibration.toml", target_size=RES)
    raw = ka.student_dets(args.participant, args.stem)
    dets = {c: v for c, v in raw.items() if c in calib}
    cams = sorted(dets, key=lambda k: int(k.split("_")[1]))
    ka.calib, ka.DETS, ka.CAMS, ka.N = calib, dets, cams, min(len(v) for v in dets.values())

    report = [
        gate0_precondition(dets, calib, cams),
        gate1_detector_sanity(dets, cams),
        gate2_fused_accuracy(),
        gate3_confident_wrong(dets, calib, cams),
    ]

    print(f"\nACCEPTANCE PROTOCOL  {args.participant}/{args.stem}")
    print("=" * 72)
    for g in report:
        verdict = {True: "PASS", False: "FAIL", None: "PENDING"}[g["pass"]]
        print(f"\n[{verdict}]  GATE {g['gate']}")
        for k, v in g.items():
            if k in ("gate", "pass", "note"):
                continue
            print(f"        {k}: {v}")
        print(f"        -> {g['note']}")

    runnable = [g for g in report if g["pass"] is not None]
    overall = all(g["pass"] for g in runnable)
    pending = [g for g in report if g["pass"] is None]
    print("\n" + "=" * 72)
    print(f"RUNNABLE GATES: {'ALL PASS' if overall else 'FAILED'} "
          f"({sum(g['pass'] for g in runnable)}/{len(runnable)})"
          + (f"   | {len(pending)} PENDING (Gate 2 ground truth)" if pending else ""))
    print("NOTE: a full ACCEPT also requires Gate 2 once a ground-truth source is chosen.")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"clip": f"{args.participant}/{args.stem}", "bars": BARS, "gates": report,
             "runnable_all_pass": overall}, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
