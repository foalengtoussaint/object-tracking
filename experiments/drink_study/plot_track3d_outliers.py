"""Outlier dashboard over the cached 3D cup trajectories (cache/track3d/).

Two outlier axes on one figure:
  MOTION   -- is the cup TRAJECTORY anomalous vs the rest? (path length, lift
              height, bbox volume, peak speed) -> robust z-score (MAD) per feature.
  QUALITY  -- is the 3D TRACK unreliable? (tri_rate, median_px, consensus gaps).

Panels:
  1. all RTS trajectories overlaid (z vs time), outliers highlighted in red
  2. all RTS trajectories in 3D (top-down x-y), outliers red
  3. quality scatter: tri_rate vs median_px, point size = #frames, outliers red
  4. motion feature scatter: lift height vs path length, outliers red
  5. per-rep robust-z bar (max |z| across motion features) -- the ranking

A rep is flagged if any motion feature |robust-z| > 3.5, or tri_rate < mean-2std,
or median_px > mean+2std. Reads cache only; no GPU/video.

    python experiments/drink_study/plot_track3d_outliers.py
    ... --out experiments/drink_study/cache/track3d_outliers.png
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
PCOLOR = {"P01": "#1f77b4", "P06": "#2ca02c", "P19": "#ff7f0e", "P23": "#9467bd"}


def robust_z(x):
    x = np.asarray(x, float)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) or 1e-9
    return 0.6745 * (x - med) / mad


def rep_features(d):
    """Kinematic features from a rep's RTS track."""
    rts = np.array([f["rts"] for f in d["frames"] if f["rts"] is not None], float)
    if len(rts) < 3:
        return None
    steps = np.linalg.norm(np.diff(rts, axis=0), axis=1)
    speed = steps * FPS                                   # mm/s
    return {
        "n": d["n_frames"],
        "path_len": float(steps.sum()),
        "lift": float(np.ptp(rts[:, 2])),                 # vertical range (the drink)
        "vol": float(np.prod(np.ptp(rts, axis=0))),       # motion bbox volume
        "peak_speed": float(np.percentile(speed, 99)),    # robust peak
        "rts": rts,
        "tri_rate": d["summary"]["tri_rate"],
        "median_px": d["summary"]["median_px"] or np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/drink_study/cache/track3d_outliers.png")
    args = ap.parse_args()

    files = sorted(p for p in CACHE.glob("*__pscale_4.json"))
    reps = []
    for fp in files:
        d = json.loads(fp.read_text())
        ft = rep_features(d)
        if ft is None:
            continue
        ft["participant"] = d["participant"]; ft["stem"] = d["stem"]
        reps.append(ft)
    print(f"{len(reps)} reps loaded", flush=True)

    MOTION = ["path_len", "lift", "vol", "peak_speed"]
    Z = {k: robust_z([r[k] for r in reps]) for k in MOTION}
    for i, r in enumerate(reps):
        r["zmax"] = max(abs(Z[k][i]) for k in MOTION)
        r["zwhich"] = max(MOTION, key=lambda k: abs(Z[k][i]))

    tri = np.array([r["tri_rate"] for r in reps])
    px = np.array([r["median_px"] for r in reps])
    tri_lo, px_hi = tri.mean() - 2 * tri.std(), np.nanmean(px) + 2 * np.nanstd(px)
    for r in reps:
        r["motion_out"] = r["zmax"] > 3.5
        r["qual_out"] = (r["tri_rate"] < tri_lo) or (r["median_px"] > px_hi)
        r["out"] = r["motion_out"] or r["qual_out"]

    outs = [r for r in reps if r["out"]]
    print(f"flagged {len(outs)} outliers:", flush=True)
    for r in sorted(outs, key=lambda r: -r["zmax"]):
        tag = []
        if r["motion_out"]:
            tag.append(f"motion:{r['zwhich']} z={r['zmax']:.1f}")
        if r["qual_out"]:
            tag.append(f"quality tri={r['tri_rate']:.3f} px={r['median_px']:.2f}")
        print(f"  {r['participant']} {r['stem'][-15:]}  {' | '.join(tag)}", flush=True)

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.24)

    def col(r):
        return "red" if r["out"] else PCOLOR.get(r["participant"], "gray")

    # 1. z vs time overlay
    ax = fig.add_subplot(gs[0, 0])
    for r in reps:
        t = np.arange(len(r["rts"])) / FPS
        ax.plot(t, r["rts"][:, 2], color=col(r), alpha=0.85 if r["out"] else 0.3,
                lw=1.6 if r["out"] else 0.7)
    ax.set(title="Cup height (z) vs time — outliers red", xlabel="s", ylabel="z (mm)")

    # 2. top-down x-y
    ax = fig.add_subplot(gs[0, 1])
    for r in reps:
        ax.plot(r["rts"][:, 0], r["rts"][:, 1], color=col(r),
                alpha=0.85 if r["out"] else 0.25, lw=1.6 if r["out"] else 0.6)
    ax.set(title="Trajectory top-down (x-y)", xlabel="x (mm)", ylabel="y (mm)")
    ax.axis("equal")

    # 3. quality scatter
    ax = fig.add_subplot(gs[0, 2])
    for r in reps:
        ax.scatter(r["tri_rate"], r["median_px"], s=max(20, r["n"] / 8),
                   color=col(r), alpha=0.8, edgecolor="k", linewidth=0.3)
    ax.axvline(tri_lo, ls="--", c="red", lw=1); ax.axhline(px_hi, ls="--", c="red", lw=1)
    ax.set(title="Quality: tri_rate vs median_px (size=#frames)",
           xlabel="tri_rate", ylabel="median_px")

    # 4. motion feature scatter
    ax = fig.add_subplot(gs[1, 0])
    for r in reps:
        ax.scatter(r["lift"], r["path_len"], color=col(r), alpha=0.8,
                   edgecolor="k", linewidth=0.3, s=40)
    ax.set(title="Motion: lift height vs path length", xlabel="lift (mm)",
           ylabel="path length (mm)")

    # 5. robust-z ranking bar
    ax = fig.add_subplot(gs[1, 1:])
    order = sorted(range(len(reps)), key=lambda i: reps[i]["zmax"])
    ax.bar(range(len(reps)), [reps[i]["zmax"] for i in order],
           color=["red" if reps[i]["out"] else PCOLOR.get(reps[i]["participant"], "gray")
                  for i in order])
    ax.axhline(3.5, ls="--", c="red", lw=1, label="z=3.5 threshold")
    ax.set(title="Per-rep max motion robust-z (sorted)", ylabel="max |robust-z|",
           xlabel="rep (sorted)")
    ax.legend()
    # label the worst few
    for i in order[-5:]:
        ax.annotate(f"{reps[i]['participant']} ..{reps[i]['stem'][-6:]}",
                    (order.index(i), reps[i]["zmax"]), fontsize=7, rotation=90,
                    va="bottom", ha="center")

    handles = [plt.Line2D([], [], color=c, lw=3, label=p) for p, c in PCOLOR.items()]
    handles.append(plt.Line2D([], [], color="red", lw=3, label="OUTLIER"))
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False)
    fig.suptitle(f"3D cup-trajectory outlier dashboard — {len(reps)} calibrated reps "
                 f"(P01/P06/P19/P23), {len(outs)} flagged", y=0.995, fontsize=13)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
