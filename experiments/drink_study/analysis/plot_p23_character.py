"""Characterize HOW P23's cup trajectories differ from P01/P06/P19.

From cache/track3d (no GPU/video). Per-rep kinematic features, paths normalized
to first cup position. Highlights the features that separate P23: lower lift,
shorter duration, higher peak speed (a faster, shallower drink).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE = Path("experiments/drink_study/cache/track3d")
FPS = 60.0
PCOLOR = {"P01": "#1f77b4", "P06": "#2ca02c", "P19": "#ff7f0e", "P23": "#d62728"}


def load():
    rows = {}
    for fp in sorted(CACHE.glob("*__pscale_4.json")):
        d = json.loads(fp.read_text()); p = d["participant"]
        rts = np.array([f["rts"] for f in d["frames"] if f["rts"] is not None], float)
        if len(rts) < 3:
            continue
        norm = rts - rts[0]
        steps = np.linalg.norm(np.diff(rts, axis=0), axis=1)
        rows.setdefault(p, []).append({
            "norm": norm,
            "lift": float(np.ptp(rts[:, 2])),
            "dur": len(rts) / FPS,
            "peak_speed": float(np.percentile(steps * FPS, 99)),
            "reach_xy": float(np.linalg.norm(norm[:, :2], axis=1).max()),
            "z": rts[:, 2] - rts[0, 2],
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/drink_study/cache/p23_character.png")
    a = ap.parse_args()
    rows = load()
    parts = sorted(rows)
    fig = plt.figure(figsize=(17, 10)); gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.28)

    # 1. normalized z vs time, P23 vs rest
    ax = fig.add_subplot(gs[0, 0])
    for p in parts:
        for r in rows[p]:
            ax.plot(np.arange(len(r["z"])) / FPS, r["z"], color=PCOLOR[p],
                    alpha=0.8 if p == "P23" else 0.25, lw=1.6 if p == "P23" else 0.7)
    ax.set(title="Cup lift (Δz) vs time — P23 red", xlabel="s", ylabel="Δz (mm)")

    # 2. normalized top-down
    ax = fig.add_subplot(gs[0, 1])
    for p in parts:
        for r in rows[p]:
            ax.plot(r["norm"][:, 0], r["norm"][:, 1], color=PCOLOR[p],
                    alpha=0.8 if p == "P23" else 0.2, lw=1.4 if p == "P23" else 0.6)
    ax.scatter([0], [0], c="k", marker="x", s=60, zorder=5)
    ax.set(title="Path top-down (normalized to start)", xlabel="Δx (mm)", ylabel="Δy (mm)")
    ax.axis("equal")

    # 3-6. feature distributions (box + jitter), P23 last/red
    feats = [("lift", "lift height (mm)  — P23 LOWER"),
             ("dur", "duration (s)  — P23 SHORTER"),
             ("peak_speed", "peak speed (mm/s)  — P23 FASTER"),
             ("reach_xy", "horiz. reach (mm)")]
    cells = [gs[0, 2], gs[1, 0], gs[1, 1], gs[1, 2]]
    for (k, title), cell in zip(feats, cells):
        ax = fig.add_subplot(cell)
        data = [[r[k] for r in rows[p]] for p in parts]
        ax.boxplot(data, labels=parts, showfliers=False, widths=0.5)
        for i, p in enumerate(parts, 1):
            x = np.random.default_rng(0).normal(i, 0.06, len(rows[p]))
            ax.scatter(x, [r[k] for r in rows[p]], color=PCOLOR[p],
                       s=22, alpha=0.8, edgecolor="k", linewidth=0.3, zorder=3)
        ax.set_title(title)

    handles = [plt.Line2D([], [], color=c, lw=3, label=p) for p, c in PCOLOR.items()]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("P23 characterization: a faster, shallower drink "
                 "(lower lift, shorter duration, higher peak speed)", y=0.99, fontsize=13)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=110, bbox_inches="tight")
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
