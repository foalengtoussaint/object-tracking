"""Shape-based trajectory-outlier dashboard over cache/track3d/.

Unlike plot_track3d_outliers.py (4 scalar features), this compares the cup PATH
SHAPE, normalized so position-on-the-table doesn't matter:

  normalize : subtract each rep's first cup position  -> traj - traj[0]
              (removes WHERE; keeps reach direction, lift, scale)

Two complementary outlier views, cross-checked:
  PCA  : resample each rep to N points, flatten (N*3 vector), PCA -> 2D.
         Outlier = Mahalanobis distance from the PC cloud center (chi2 cutoff).
  DTW  : pairwise Dynamic Time Warping distance (time-warp tolerant -> a pause
         mid-drink doesn't fake an outlier). Per-rep score = mean DTW to all
         others; flag by robust-z. Self-contained numpy DTW (no new deps).

Agreement between PCA and DTW = a trustworthy outlier; disagreement usually means
a time-warp artifact (PCA-only) worth eyeballing.

    python experiments/drink_study/plot_track3d_shape.py
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import chi2
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE = Path("experiments/drink_study/cache/track3d")
NRES = 50                                      # resample length for PCA
PCOLOR = {"P01": "#1f77b4", "P06": "#2ca02c", "P19": "#ff7f0e", "P23": "#9467bd"}


def robust_z(x):
    x = np.asarray(x, float)
    med = np.median(x); mad = np.median(np.abs(x - med)) or 1e-9
    return 0.6745 * (x - med) / mad


def resample(traj, n):
    """Arc-length resample a (T,3) path to (n,3) so speed differences don't matter."""
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(traj, axis=0), axis=1))]
    if seg[-1] == 0:
        return np.repeat(traj[:1], n, axis=0)
    u = seg / seg[-1]
    grid = np.linspace(0, 1, n)
    return np.stack([np.interp(grid, u, traj[:, k]) for k in range(3)], axis=1)


def dtw(a, b):
    """Classic DTW on two (n,3)/(m,3) paths, euclidean local cost. Returns distance."""
    na, nb = len(a), len(b)
    D = np.full((na + 1, nb + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, na + 1):
        ai = a[i - 1]
        for j in range(1, nb + 1):
            c = np.linalg.norm(ai - b[j - 1])
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[na, nb]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/drink_study/cache/track3d_shape.png")
    args = ap.parse_args()

    reps = []
    for fp in sorted(CACHE.glob("*__pscale_4.json")):
        d = json.loads(fp.read_text())
        rts = np.array([f["rts"] for f in d["frames"] if f["rts"] is not None], float)
        if len(rts) < 3:
            continue
        norm = rts - rts[0]                                  # <-- relative to first cup position
        reps.append({"participant": d["participant"], "stem": d["stem"],
                     "norm": norm, "rs": resample(norm, NRES)})
    print(f"{len(reps)} reps", flush=True)

    # --- PCA on flattened resampled paths ---
    X = np.stack([r["rs"].ravel() for r in reps])           # (R, NRES*3)
    pcs = PCA(n_components=2).fit_transform(X)
    mu = pcs.mean(0); cov = np.cov(pcs.T); inv = np.linalg.inv(cov)
    md = np.array([np.sqrt((p - mu) @ inv @ (p - mu)) for p in pcs])
    md_cut = np.sqrt(chi2.ppf(0.975, df=2))                 # 97.5% Mahalanobis cutoff
    for i, r in enumerate(reps):
        r["pc"] = pcs[i]; r["md"] = md[i]; r["pca_out"] = md[i] > md_cut

    # --- DTW mean-distance score ---
    R = len(reps)
    Dm = np.zeros((R, R))
    for i in range(R):
        for j in range(i + 1, R):
            Dm[i, j] = Dm[j, i] = dtw(reps[i]["rs"], reps[j]["rs"])
    dscore = Dm.sum(1) / (R - 1)
    dz = robust_z(dscore)
    for i, r in enumerate(reps):
        r["dtw"] = dscore[i]; r["dz"] = dz[i]; r["dtw_out"] = dz[i] > 3.5

    for r in reps:
        r["both"] = r["pca_out"] and r["dtw_out"]
        r["any"] = r["pca_out"] or r["dtw_out"]

    print("\n=== shape outliers (relative-to-start) ===", flush=True)
    for r in sorted(reps, key=lambda r: -r["md"]):
        if not r["any"]:
            continue
        flags = []
        if r["pca_out"]: flags.append(f"PCA md={r['md']:.2f}")
        if r["dtw_out"]: flags.append(f"DTW z={r['dz']:.1f}")
        mark = "  <<BOTH" if r["both"] else ""
        print(f"  {r['participant']} {r['stem'][-15:]}  {' | '.join(flags)}{mark}", flush=True)

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(18, 10)); gs = fig.add_gridspec(2, 3, hspace=0.27, wspace=0.24)

    def col(r):
        if r["both"]: return "red"
        if r["any"]: return "darkorange"
        return PCOLOR.get(r["participant"], "gray")

    # 1. normalized z vs time
    ax = fig.add_subplot(gs[0, 0])
    for r in reps:
        ax.plot(np.arange(len(r["norm"])) / 60.0, r["norm"][:, 2], color=col(r),
                alpha=0.9 if r["any"] else 0.3, lw=1.7 if r["any"] else 0.7)
    ax.set(title="Normalized cup height (z - z0) vs time", xlabel="s", ylabel="Δz (mm)")

    # 2. normalized top-down x-y
    ax = fig.add_subplot(gs[0, 1])
    for r in reps:
        ax.plot(r["norm"][:, 0], r["norm"][:, 1], color=col(r),
                alpha=0.9 if r["any"] else 0.25, lw=1.7 if r["any"] else 0.6)
    ax.scatter([0], [0], c="k", marker="x", s=60, zorder=5, label="start")
    ax.set(title="Normalized path top-down (x-x0, y-y0)", xlabel="Δx (mm)", ylabel="Δy (mm)")
    ax.axis("equal"); ax.legend()

    # 3. PCA scatter
    ax = fig.add_subplot(gs[0, 2])
    for r in reps:
        ax.scatter(*r["pc"], color=col(r), s=45, alpha=0.85, edgecolor="k", linewidth=0.3)
    th = np.linspace(0, 2 * np.pi, 100)
    L = np.linalg.cholesky(cov)
    ell = (mu[:, None] + md_cut * L @ np.array([np.cos(th), np.sin(th)])).T
    ax.plot(ell[:, 0], ell[:, 1], "r--", lw=1, label="97.5% ellipse")
    ax.set(title="PCA of normalized paths (shape space)", xlabel="PC1", ylabel="PC2"); ax.legend()

    # 4. DTW mean-distance ranking
    ax = fig.add_subplot(gs[1, 0])
    order = np.argsort(dscore)
    ax.bar(range(R), dscore[order], color=[col(reps[i]) for i in order])
    ax.set(title="Mean DTW distance to all other reps (sorted)", ylabel="mean DTW", xlabel="rep")

    # 5. PCA-md vs DTW-z agreement
    ax = fig.add_subplot(gs[1, 1])
    for r in reps:
        ax.scatter(r["md"], r["dz"], color=col(r), s=45, alpha=0.85, edgecolor="k", linewidth=0.3)
    ax.axvline(md_cut, ls="--", c="red", lw=1); ax.axhline(3.5, ls="--", c="red", lw=1)
    ax.set(title="Agreement: PCA Mahalanobis vs DTW robust-z",
           xlabel="PCA Mahalanobis dist", ylabel="DTW robust-z")
    for r in reps:
        if r["any"]:
            ax.annotate(f"{r['participant']}..{r['stem'][-6:]}", (r["md"], r["dz"]),
                        fontsize=6, ha="left", va="bottom")

    # 6. 3D of normalized paths (outliers only emphasized)
    ax = fig.add_subplot(gs[1, 2], projection="3d")
    for r in reps:
        ax.plot(r["norm"][:, 0], r["norm"][:, 1], r["norm"][:, 2], color=col(r),
                alpha=0.9 if r["any"] else 0.15, lw=1.6 if r["any"] else 0.5)
    ax.set(title="Normalized paths 3D", xlabel="Δx", ylabel="Δy", zlabel="Δz")

    handles = [plt.Line2D([], [], color=c, lw=3, label=p) for p, c in PCOLOR.items()]
    handles += [plt.Line2D([], [], color="darkorange", lw=3, label="one method"),
                plt.Line2D([], [], color="red", lw=3, label="BOTH methods")]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False)
    nb = sum(r["both"] for r in reps); na = sum(r["any"] for r in reps)
    fig.suptitle(f"Shape-based trajectory outliers (normalized to first cup position) — "
                 f"{R} reps, {na} flagged ({nb} by both PCA+DTW)", y=0.995, fontsize=13)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
