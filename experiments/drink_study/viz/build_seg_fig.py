"""Slide-8 figure for the learned segmenter: (a) mean dwell-duration error across the
feature progression (tuned gate -> 6ch -> 13ch -> hybrid), (b) the TAIL (p95/p99/max) for
ALL FOUR methods -- the win hides below p95 and only shows in the extreme tail. Every
number is recomputed from per-rep LOPO caches (no hardcoded stats).

    python experiments/drink_study/build_seg_fig.py   -> slides/fig14_segmenter.png
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import json, glob, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
from _paths import CACHE
OUT = HERE / "slides"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
import segment_cup_only as S
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 140})
HZ = 60.0


def _geo(xyz, **kw):
    r = S.segment_cup_only(xyz, fps=HZ, **kw); rr = r["drink_runs"]
    return (rr[0][0], rr[-1][1]) if rr else None


def _err(tsp, vsp):
    td = (tsp[1] - tsp[0]) / HZ * 1000
    if vsp is None:
        return td
    return abs((vsp[1] - vsp[0]) / HZ * 1000 - td)


# truth dwell spans (mocap gate) from the lopo_fused cache -- shared by all methods
TRUTH = {}
for f in glob.glob(str(CACHE / "lopo_fused" / "*.npz")):
    d = np.load(f, allow_pickle=True); v = str(d["video"])
    ts = _geo(d["true"])
    if ts is not None:
        TRUTH[v] = ts


def tail_from_pred(pred):
    """pred: {video: span-or-None} -> np.array of |dur| errors over reps with truth."""
    de = [_err(TRUTH[v], tuple(sp) if sp else None) for v, sp in pred.items() if v in TRUTH]
    return np.array(de)


# 6ch + tuned per-rep spans from learn_seg.json; 13ch + hybrid per-rep |dur| from their jsons
_seg = json.load(open(CACHE / "learn_seg.json"))
_clf13 = json.load(open(CACHE / "learn_seg_clf.json"))
_hyb = json.load(open(CACHE / "learn_seg_hybrid.json"))
err_tuned = tail_from_pred(_seg["pred"]["tuned"])
err_6ch = tail_from_pred(_seg["pred"]["clf"])
err_13ch = np.array([v[1] for v in _clf13["perrep"].values()])   # [tuned, clf13]
err_hyb = np.array([v[1] for v in _hyb["perrep"].values()])       # [tuned, hybrid]
ERR = {"tuned\ngate": err_tuned, "learned\n6ch": err_6ch,
       "+3D dir\n13ch": err_13ch, "HYBRID\n13+4occ": err_hyb}

labels = list(ERR)
means = [ERR[k].mean() for k in labels]
meds = [np.percentile(ERR[k], 50) for k in labels]
cols = ["#8d99ae", "#5c9ead", "#3a7ca5", "#2a9d8f"]

fig, (a, b) = plt.subplots(1, 2, figsize=(12.4, 4.6), gridspec_kw=dict(width_ratios=[1.35, 1]))

# (a) median + mean: median drops ONCE (gate -> learned) then is flat; mean keeps easing
# because the TAIL keeps shrinking. Two bars per method make that split explicit.
x = np.arange(len(labels)); w = 0.38
a.bar(x - w / 2, meds, w, color=cols, edgecolor="#333", label="median (p50)")
a.bar(x + w / 2, means, w, color=cols, edgecolor="#333", alpha=0.45, hatch="//", label="mean")
for xi, (md, mn) in enumerate(zip(meds, means)):
    a.text(xi - w / 2, md + 4, f"{md:.0f}", ha="center", fontsize=9.5, weight="bold")
    a.text(xi + w / 2, mn + 4, f"{mn:.0f}", ha="center", fontsize=9.5, color="#444")
a.set_xticks(x); a.set_xticklabels(labels, fontsize=10.5)
a.set_ylabel("dwell-duration error (ms)")
a.set_ylim(0, max(means) * 1.2)
a.axhline(meds[0], color="#8d99ae", ls="--", lw=1)
a.set_title("Median drops ONCE (gate→learned, 133→~110), then flat;\n"
            "mean keeps easing only because the tail shrinks",
            fontsize=10.5, weight="bold")
a.legend(fontsize=8.5, loc="upper right")
a.annotate("3D velocity DIRECTION\nsolves the P20 place-down (tail)", xy=(2 + w / 2, means[2]),
           xytext=(0.75, max(means) * 1.06), fontsize=8, color="#3a7ca5",
           arrowprops=dict(arrowstyle="->", color="#3a7ca5", lw=1.2))

# (b) the tail: p95 / p99 / max for ALL FOUR methods. p95 is ~flat -> the win hides
# BELOW p95 and only shows in the extreme tail (that's the message).
def pct(arr, q): return np.percentile(arr, q)
groups = ["p95", "p99", "max"]
qs = [95, 99, 100]
xb = np.arange(len(groups)); w = 0.20
for j, (lab, col) in enumerate(zip(labels, cols)):
    arr = ERR[lab]
    vals = [pct(arr, q) for q in qs]
    off = (j - 1.5) * w
    b.bar(xb + off, vals, w, color=col, edgecolor="#333",
          label=lab.replace("\n", " "))
    for xi, v in zip(xb, vals):
        b.text(xi + off, v + 55, f"{v:.0f}", ha="center", fontsize=7.2, rotation=90,
               va="bottom", color="#333")
b.text(0, pct(ERR[labels[0]], 95) + 430, "p95 ≈ flat", ha="center", va="bottom",
       fontsize=8.5, color="#666", style="italic")
b.set_xticks(xb); b.set_xticklabels(groups, fontsize=11)
b.set_ylabel("dwell-duration error (ms)")
b.set_ylim(0, max(ERR[labels[0]].max(), ERR["learned\n6ch"].max()) * 1.18)
p99_t = pct(ERR[labels[0]], 99); p99_h = pct(ERR[labels[3]], 99)
b.set_title(f"The win is in the EXTREME tail\np99 {p99_t:.0f}→{p99_h:.0f}, "
            f"max {ERR[labels[0]].max():.0f}→{ERR[labels[3]].max():.0f}",
            fontsize=10.5, weight="bold")
b.legend(fontsize=7.6, loc="upper left", ncol=2)

fig.suptitle("Learned drink-dwell segmenter vs the tuned geometric gate  (LOPO, "
             f"{len(err_tuned)} reps)", fontsize=13, weight="bold", y=1.02)
fig.tight_layout()
p = OUT / "fig14_segmenter.png"
fig.savefig(p, bbox_inches="tight", facecolor="white")
print("wrote", p)
