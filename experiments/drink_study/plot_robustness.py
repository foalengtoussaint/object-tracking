"""Visualize the robustness/usefulness model from cache/robustness.json.

Layout:
  TOP ROW  -- per-axis ENVELOPE small-multiples: one mini-panel per degradation
              knob, each with its OWN real x-axis (no normalization). Left y =
              worst-1% track error (mm, log) with the cup-radius line; right y =
              % conditions staying on-cup. Shows at a glance that every knob is
              flat-safe EXCEPT n_cams, which is catastrophic at 2 cams.
  BOTTOM L -- which raw-detection metric PREDICTS success: logistic standardized
              weights + each metric's solo go/no-go AUC.
  BOTTOM R -- actual worst-1% error vs inlier_frac, coloured good/bad: the
              currency is camera AGREEMENT, not precision/density.

No recompute -- reads the cached JSON. Writes cache/robustness.png.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).resolve().parent
D = json.loads((HERE / "cache" / "robustness.json").read_text())
CUP = D["cup_radius_mm"]
OUT = HERE / "cache" / "robustness.png"

# knob -> (x-axis label, how to format a level for ticks)
AXES = [
    ("drop",    "dropout fraction",        lambda v: f"{v:.0%}"),
    ("corrupt", "corrupted-cam fraction",  lambda v: f"{v:.0%}"),
    ("sigma",   "pixel jitter  sigma (px)", lambda v: f"{v:.0f}"),
    ("n_cams",  "# cameras fed",           lambda v: f"{v:.0f}"),
    ("decim",   "keep every K-th frame",   lambda v: f"1/{v:.0f}"),
]

fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 5, figure=fig, height_ratios=[1, 1.15], hspace=0.42, wspace=0.55)
fig.suptitle(
    "Detector-robustness envelope & usefulness predictor   "
    f"(ground truth: P01 verified trial, {D['n_conditions']} conditions)",
    fontsize=14, fontweight="bold")

# ---------------------------------------- TOP ROW: per-knob small multiples
for j, (axis, xlabel, fmt) in enumerate(AXES):
    a = fig.add_subplot(gs[0, j])
    pts = D["envelope"][axis]
    xs = [p[0] for p in pts]
    good = [p[1] * 100 for p in pts]
    err = [p[2] for p in pts]
    xpos = list(range(len(xs)))            # even spacing; real values on ticks

    # left axis: worst-1% error (log)
    a.plot(xpos, err, "o-", color="#1f77b4", lw=2, ms=6, zorder=3)
    a.axhline(CUP, color="crimson", ls="--", lw=1.4, zorder=2)
    a.set_yscale("log")
    a.set_ylim(8, 2e4)
    a.set_xticks(xpos)
    a.set_xticklabels([fmt(x) for x in xs], fontsize=8, rotation=0)
    a.set_xlabel(xlabel, fontsize=9)
    if j == 0:
        a.set_ylabel("worst-1% err (mm, log)", color="#1f77b4", fontsize=9)
    a.tick_params(axis="y", labelcolor="#1f77b4")
    a.set_title(axis, fontsize=11, fontweight="bold")
    a.grid(alpha=0.25, which="both")

    # right axis: % on-cup
    a2 = a.twinx()
    a2.plot(xpos, good, "s--", color="#2a9d8f", lw=1.4, ms=5, alpha=0.85, zorder=1)
    a2.set_ylim(-5, 108)
    if j == len(AXES) - 1:
        a2.set_ylabel("% on-cup", color="#2a9d8f", fontsize=9)
    a2.tick_params(axis="y", labelcolor="#2a9d8f", labelsize=8)

    # flag any level that busts the cup radius
    for x, e, g in zip(xpos, err, good):
        if e > CUP:
            a.annotate("FAIL", (x, e), color="crimson", fontsize=8,
                       fontweight="bold", ha="center", va="bottom")

# shared legend for the top row
from matplotlib.lines import Line2D
fig.legend(handles=[
    Line2D([], [], color="#1f77b4", marker="o", label="worst-1% track err (mm)"),
    Line2D([], [], color="#2a9d8f", marker="s", ls="--", label="% conditions on-cup"),
    Line2D([], [], color="crimson", ls="--", label=f"cup radius {CUP:.0f} mm (usable threshold)"),
], loc="upper right", fontsize=9, ncol=1, bbox_to_anchor=(0.995, 0.965))

# ---------------------------------- BOTTOM LEFT: which metric predicts success
a = fig.add_subplot(gs[1, :2])
feats = D["feats"]
logw = [D["logistic"]["weights"][f] for f in feats]
gate_auc = [D["single_metric_gates"][f]["auc"] for f in feats]
order = np.argsort(np.abs(logw))
y = np.arange(len(feats))
a.barh(y, [logw[i] for i in order],
       color=["#2a9d8f" if logw[i] > 0 else "#e76f51" for i in order])
a.set_yticks(y)
a.set_yticklabels([feats[i] for i in order])
a.axvline(0, color="k", lw=0.8)
a.set_xlabel("logistic standardized weight   (+ → predicts a GOOD track)")
a.set_title(f"Which raw-detection metric predicts success?   logistic AUC={D['logistic']['auc']:.2f}",
            fontsize=11, fontweight="bold")
for i, idx in enumerate(order):
    a.text(0.02, i, f"  solo go/no-go AUC {gate_auc[idx]:.2f}", va="center",
           fontsize=8, transform=a.get_yaxis_transform(), color="#333")
a.grid(alpha=0.3, axis="x")

# ------------ BOTTOM RIGHT: predicted P(good) from cheap metrics vs actual error
# Tests the WHOLE logistic predictor: does P(good) computed from raw-detection
# metrics (no ground truth) line up with the actual post-pipeline track error?
a = fig.add_subplot(gs[1, 2:])
rows = D["rows"]
pred = np.array([r.get("pred_good", np.nan) for r in rows])
p99 = np.array([r["p99_mm"] if r["p99_mm"] is not None else np.nan for r in rows])
good = np.array([r["good"] for r in rows])
m = ~np.isnan(p99) & ~np.isnan(pred)
sc = a.scatter(pred[m], np.clip(p99[m], 1, 1e4), c=good[m].astype(float),
               cmap="RdYlGn", s=30, alpha=0.8, edgecolor="k", linewidth=0.3)
a.axhline(CUP, color="crimson", ls="--", lw=1.5, label=f"cup radius {CUP:.0f} mm")
a.axvline(0.5, color="gray", ls=":", lw=1.3, label="decision threshold P=0.5")
a.set_yscale("log")
a.set_xlim(-0.03, 1.03)
a.set_xlabel("predicted P(track good)  — from cheap no-GT metrics (5-fold CV)")
a.set_ylabel("actual worst-1% track error (mm, log)")
a.set_title(f"Does the predictor work?  high P(good) → low error   (AUC {D['logistic']['auc']:.2f})",
            fontsize=11, fontweight="bold")
# shade the two "correct" quadrants for readability
a.axvspan(-0.03, 0.5, ymin=0, ymax=1, color="crimson", alpha=0.04)
a.axvspan(0.5, 1.03, ymin=0, ymax=1, color="green", alpha=0.04)
a.legend(fontsize=8, loc="upper center")
a.grid(alpha=0.3, which="both")
cb = fig.colorbar(sc, ax=a, fraction=0.045)
cb.set_label("actual: track good (1) / bad (0)")

fig.savefig(OUT, dpi=110, bbox_inches="tight")
print(f"wrote {OUT}")
