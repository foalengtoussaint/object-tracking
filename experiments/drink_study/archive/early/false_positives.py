"""Drill into the predictor's FALSE POSITIVES -- the deployment risk.

A false positive here = the no-GT usefulness predictor says "this track is GOOD"
(pred_good >= 0.5) but the post-pipeline track is actually BAD (p99 >= cup radius).
These are the dangerous calls: a confident-but-wrong position we'd trust in
deployment. (A false NEGATIVE just wastes a usable track -- annoying, not unsafe.)

Reads cache/robustness.json (no recompute). Prints:
  1. the confusion matrix at P=0.5 and the FP rate among predicted-good calls
  2. every FP, split into the two families (camera-disagreement vs too-sparse)
  3. why the cheap agreement metrics fail to flag them (the metric bug), and
  4. how well candidate runtime guards would suppress them.

Writes cache/false_positives.json for the animation / notebook.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
D = json.loads((HERE / "cache" / "robustness.json").read_text())
CUP = D["cup_radius_mm"]
ROWS = D["rows"]
OUT = HERE / "cache" / "false_positives.json"
THRESH = 0.5


def label(r):
    pred_good = r.get("pred_good", 0.0) >= THRESH
    return ("TP" if r["good"] else "FP") if pred_good else ("FN" if r["good"] else "TN")


def fmt(r):
    nc = r["n_cams"]
    return (f"axis={r['axis']:7} drop={r['drop']:.2f} corr={r['corrupt']:.2f} "
            f"sig={r['sigma']:4.0f} nc={str(nc):>4} dec={r['decim']:2} | "
            f"det={r['det_rate']:.2f} tri={r['tri_rate']:.2f} inl={r['inlier_frac']:.3f} "
            f"reproj={r['reproj_px']:6.1f} | pred={r['pred_good']:.2f} p99={r['p99_mm']:.0f}mm")


def main():
    conf = {k: [r for r in ROWS if label(r) == k] for k in ("TP", "FP", "FN", "TN")}
    tp, fp, fn, tn = (conf[k] for k in ("TP", "FP", "FN", "TN"))
    npred = len(tp) + len(fp)

    print("=" * 78)
    print("FALSE-POSITIVE ANALYSIS  (predicted GOOD but actually BAD = confident-wrong)")
    print("=" * 78)
    print(f"\nConfusion @ P(good)>={THRESH}:  TP={len(tp)}  FP={len(fp)}  "
          f"FN={len(fn)}  TN={len(tn)}")
    print(f"FP rate among the {npred} predicted-GOOD calls: "
          f"{len(fp)/max(1,npred):.0%}  <-- this is the deployment risk")
    print(f"FP median actual error: {np.median([r['p99_mm'] for r in fp]):.0f}mm "
          f"(worst {max(r['p99_mm'] for r in fp):.0f}mm) -- confident AND far-wrong")

    # --- two families: camera-disagreement (dangerous) vs too-sparse (mild) ---
    loud = [r for r in fp if r["inlier_frac"] < 0.1]
    sparse = [r for r in fp if r["inlier_frac"] >= 0.1]
    print("\n" + "-" * 78)
    print("TWO FP FAMILIES")
    print("-" * 78)
    print(f"\n(A) CAMERA-DISAGREEMENT  ({len(loud)} FPs, median "
          f"{np.median([r['p99_mm'] for r in loud]):.0f}mm) -- THE DANGEROUS ONE.")
    print("    Every cam detects & triangulates (det~.92, tri=1.0, n=10 all look great)")
    print("    but they localize DIFFERENT points -> fusion locks onto a phantom.")
    print("    Cause: high corrupt-cam fraction or large pixel jitter.")
    for r in sorted(loud, key=lambda r: -r["p99_mm"]):
        print("      " + fmt(r))
    print(f"\n(B) TOO-SPARSE  ({len(sparse)} FPs, median "
          f"{np.median([r['p99_mm'] for r in sparse]):.0f}mm) -- milder, near-threshold.")
    print("    Extreme dropout: few triangulations, KF coasts then drifts just past cup.")
    for r in sorted(sparse, key=lambda r: -r["p99_mm"]):
        print("      " + fmt(r))

    # --- why the cheap agreement metric does NOT flag the dangerous family ---
    print("\n" + "-" * 78)
    print("WHY inlier_frac DOESN'T CATCH THEM  (the metric bug)")
    print("-" * 78)
    good = [r for r in ROWS if r["good"]]
    y = np.array([r["good"] for r in ROWS], int)
    print(f"\n  inlier_frac solo AUC over all conditions: "
          f"{roc_auc_score(y, [r['inlier_frac'] for r in ROWS]):.3f}  (~coin flip)")
    print(f"  median inlier_frac:  GOOD tracks={np.median([r['inlier_frac'] for r in good]):.3f}   "
          f"disagreement-FP={np.median([r['inlier_frac'] for r in loud]):.3f}")
    print("  -> GOOD is not higher than the FPs. inlier_frac as stored does not separate.")
    print("  ROOT CAUSE: inlier_frac/reproj_px are measured on the RAW degraded dets")
    print("  (corrupted cams included) BEFORE the pipeline's own >=3-cam consensus gate,")
    print("  so they score 'do all raw cams agree' -- which a corrupt cam destroys even")
    print("  when the pipeline would have correctly EJECTED it. They also saturate with")
    print("  exactly-determined (3-cam) geometry, where a good track shows reproj>1000px.")
    print("  FIX (open): compute agreement on the POST-gate inlier set (RANSAC inlier")
    print("  count / consensus residual), not the raw camera set.")

    # --- candidate runtime guards: how well each suppresses the FPs ---
    print("\n" + "-" * 78)
    print("CANDIDATE RUNTIME GUARDS  (extra check on top of pred_good>=0.5)")
    print("-" * 78)
    pred_good_rows = tp + fp
    tot_good = len(tp)

    def evalg(name, keep):
        kept = [r for r in pred_good_rows if keep(r)]
        kg = sum(r["good"] for r in kept)
        kb = len(kept) - kg
        print(f"  {name:46} -> keeps {kg}/{tot_good} good, {kb} FP remain")
        return {"name": name, "kept_good": kg, "fp_remain": kb}

    guards = [
        evalg("none (baseline pred>=.5)", lambda r: True),
        evalg("AND inlier_frac >= 0.1", lambda r: r["inlier_frac"] >= 0.1),
        evalg("AND reproj_px <= 100", lambda r: r["reproj_px"] <= 100),
        evalg("AND tri_rate >= 0.1", lambda r: r["tri_rate"] >= 0.1),
        evalg("AND inlier>=.1 AND tri>=.1", lambda r: r["inlier_frac"] >= 0.1 and r["tri_rate"] >= 0.1),
    ]
    print("\n  None is clean: any threshold that kills the disagreement FPs also kills")
    print("  good sparse/3-cam tracks. CONCLUSION: the cheap raw-det metrics can't")
    print("  separate confident-wrong here -- you need (1) a post-gate agreement metric,")
    print("  and/or (2) a runtime post-fusion reprojection-residual spike detector that")
    print("  watches the ACCEPTED track, not the raw detections.")

    OUT.write_text(json.dumps({
        "threshold": THRESH, "cup_radius_mm": CUP,
        "confusion": {k: len(v) for k, v in conf.items()},
        "fp_rate_among_predicted_good": round(len(fp) / max(1, npred), 3),
        "families": {
            "camera_disagreement": [r for r in loud],
            "too_sparse": [r for r in sparse],
        },
        "guards": guards,
        "false_positives": sorted(fp, key=lambda r: -r["p99_mm"]),
    }, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
