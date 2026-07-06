"""THE WIN, one figure: base17 (video-only) vs proxy21 (+head distance) dwell error.

Reads cache/results.json (from run.py) and makes a 3-panel summary:
  1. CDF of per-rep |duration error| — proxy21 curve sits left of base17 (better everywhere).
  2. Paired per-rep improvement — for each rep, base17 err vs proxy21 err (below diagonal = head helps).
  3. Headline table — mean / p50 / p90 / p99 / max for both, + how many reps improved.

    python experiments/drink_dwell/summary.py
Writes ../drink_study/slides/dwell_summary.png
"""
from __future__ import annotations
import sys as _s, pathlib as _p
_s.path.insert(0, str(_p.Path(__file__).resolve().parent))
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = _p.Path(__file__).resolve().parent
RESULTS = HERE / "cache" / "results.json"
OUT = HERE.parents[0] / "drink_study" / "slides" / "dwell_summary.png"

C_BASE, C_PROX = "#777", "#c0392b"


def main():
    d = json.load(open(RESULTS))
    cols = d["perrep_cols"]
    ib, ip = cols.index("base17"), cols.index("proxy21")
    rows = np.array(list(d["perrep"].values()), float)
    base, prox = rows[:, ib], rows[:, ip]
    n = len(rows)

    fig = plt.figure(figsize=(15, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 0.9], wspace=0.28)

    # --- panel 1: CDF ---
    ax = fig.add_subplot(gs[0, 0])
    for arr, c, lab in [(base, C_BASE, "base17 (video only)"), (prox, C_PROX, "proxy21 (+head dist)")]:
        xs = np.sort(arr); ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, color=c, lw=2.2, label=lab)
    ax.axvline(0, color="#ccc", lw=0.8)
    ax.set_xlim(0, 600); ax.set_ylim(0, 1.0)
    ax.set_xlabel("|dwell duration error| (ms)"); ax.set_ylabel("fraction of reps ≤ x")
    ax.set_title("Error CDF — proxy21 left of base17 = better", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.grid(alpha=0.25)

    # --- panel 2: paired scatter (base vs proxy per rep) ---
    ax = fig.add_subplot(gs[0, 1])
    lim = 700
    b = np.clip(base, 0, lim); p = np.clip(prox, 0, lim)
    better = prox < base - 1; worse = prox > base + 1
    ax.scatter(b[better], p[better], s=10, c=C_PROX, alpha=0.5, label=f"head helps ({better.sum()})")
    ax.scatter(b[worse], p[worse], s=10, c=C_BASE, alpha=0.5, label=f"head hurts ({worse.sum()})")
    ax.scatter(b[~better & ~worse], p[~better & ~worse], s=8, c="#bbb", alpha=0.5,
               label=f"tie ({(~better & ~worse).sum()})")
    ax.plot([0, lim], [0, lim], color="#333", lw=1, ls="--")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("base17 error (ms)"); ax.set_ylabel("proxy21 error (ms)")
    ax.set_title("Per-rep: below diagonal = head distance wins", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    ax.grid(alpha=0.25)

    # --- panel 3: headline table ---
    ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
    s = d["summary"]
    metrics = ["mean", "p50", "p90", "p99", "max"]
    cell = [[m, f"{s['base17'][m]:.0f}", f"{s['proxy21'][m]:.0f}"] for m in metrics]
    tbl = ax.table(cellText=cell, colLabels=["", "base17", "proxy21"],
                   cellLoc="center", loc="center", bbox=[0.0, 0.30, 1.0, 0.62])
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 1.5)
    for (r, c), cc in tbl.get_celld().items():
        cc.set_edgecolor("#ddd")
        if r == 0:
            cc.set_facecolor("#f2f2f2"); cc.set_text_props(fontweight="bold")
        if c == 2 and r > 0:
            cc.set_text_props(color=C_PROX, fontweight="bold")
    ax.text(0.5, 0.20, f"n = {n} reps (LOPO)", ha="center", fontsize=10, transform=ax.transAxes)
    ax.text(0.5, 0.10, f"head distance improves {better.sum()}/{n} "
            f"({100*better.sum()/n:.0f}%)", ha="center", fontsize=10.5,
            color=C_PROX, fontweight="bold", transform=ax.transAxes)
    ax.set_title("drink-dwell error (ms)", fontsize=11, fontweight="bold")

    fig.suptitle("Drink-dwell: adding cup→head distance (proxy21) vs video-only (base17)",
                 fontsize=14, fontweight="bold", y=1.02)
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}", flush=True)
    print(f"  base17 mean {s['base17']['mean']:.0f}  proxy21 mean {s['proxy21']['mean']:.0f}  "
          f"({better.sum()}/{n} improved)", flush=True)


if __name__ == "__main__":
    main()
