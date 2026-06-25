"""Dashboard: where given reps sit in the cup-only segmentation / trajectory space.

Default target set = the reps that produced no drink-dwell (cup_only_all.json).
Builds one figure:
  A. PCA shape scatter (cohort grey, targets colored by drink-status, labeled)
  B. DTW mean-distance ranking (targets highlighted)
  C. PCA-Mahalanobis vs DTW-z agreement (targets labeled)
  D. normalized top-down paths: cohort faint, targets bold
  E. normalized height (Δz) vs time: cohort faint, targets bold
  F..  per-target segmentation timelines (cup speed + disp + phase bands)

The pairwise DTW matrix is cached to cache/seg_dtw.npy (keyed by rep list) so
re-runs are instant.

    python experiments/drink_study/dashboard_segspace.py
    python experiments/drink_study/dashboard_segspace.py --reps P10_..._153258
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import chi2
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import segment_cup_only as sc

TRACK = Path("experiments/drink_study/cache/track3d")
ALL = Path("experiments/drink_study/cache/cup_only_all.json")
DTW_CACHE = Path("experiments/drink_study/cache/seg_dtw.npy")
NRES = 50


def resample(traj, n):
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(traj, axis=0), axis=1))]
    if seg[-1] == 0:
        return np.repeat(traj[:1], n, axis=0)
    u = seg / seg[-1]; g = np.linspace(0, 1, n)
    return np.stack([np.interp(g, u, traj[:, k]) for k in range(3)], 1)


def dtw_cost(C):
    n, m = C.shape; D = np.full((n + 1, m + 1), np.inf); D[0, 0] = 0
    for i in range(1, n + 1):
        Di, Dp, Ci = D[i], D[i - 1], C[i - 1]
        for j in range(1, m + 1):
            Di[j] = Ci[j - 1] + min(Dp[j], Di[j - 1], Dp[j - 1])
    return D[n, m]


def robust_z(x):
    x = np.asarray(x, float); med = np.median(x); mad = np.median(np.abs(x - med)) or 1e-9
    return 0.6745 * (x - med) / mad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="*", default=None, help="target stems (default: no-drink set)")
    ap.add_argument("--out", default="experiments/drink_study/cache/segspace_dashboard.png")
    args = ap.parse_args()

    allr = json.loads(ALL.read_text())
    drink = {r["rep"]: r["drink"] for r in allr}
    targets = set(args.reps) if args.reps else {r["rep"] for r in allr if not r["drink"]}

    reps = []
    for fp in sorted(TRACK.glob("P*__pscale_4.json")):
        d = json.loads(fp.read_text())
        rts = np.array([f["rts"] for f in d["frames"] if f["rts"] is not None], float)
        if len(rts) < 3:
            continue
        name = fp.stem
        norm = rts - rts[0]
        reps.append({"name": name, "p": d["participant"], "norm": norm,
                     "rs": resample(norm, NRES), "tgt": name in targets,
                     "drink": drink.get(name, True)})
    R = len(reps)
    print(f"{R} reps, {sum(r['tgt'] for r in reps)} targets", flush=True)

    # PCA
    X = np.stack([r["rs"].ravel() for r in reps])
    pcs = PCA(n_components=2).fit_transform(X)
    mu = pcs.mean(0); cov = np.cov(pcs.T); inv = np.linalg.inv(cov)
    md = np.array([np.sqrt((p - mu) @ inv @ (p - mu)) for p in pcs])
    mdcut = np.sqrt(chi2.ppf(0.975, df=2))

    # DTW (cached)
    names = [r["name"] for r in reps]
    cache_ok = False
    if DTW_CACHE.exists():
        z = np.load(DTW_CACHE, allow_pickle=True).item()
        if z.get("names") == names:
            Dm = z["Dm"]; cache_ok = True; print("DTW from cache", flush=True)
    if not cache_ok:
        print("computing pairwise DTW...", flush=True)
        Dm = np.zeros((R, R))
        for i in range(R):
            for j in range(i + 1, R):
                Dm[i, j] = Dm[j, i] = dtw_cost(cdist(reps[i]["rs"], reps[j]["rs"]))
            if i % 50 == 0:
                print(f"  dtw row {i}/{R}", flush=True)
        np.save(DTW_CACHE, {"names": names, "Dm": Dm})
    dscore = Dm.sum(1) / (R - 1); dz = robust_z(dscore)
    for i, r in enumerate(reps):
        r["pc"] = pcs[i]; r["md"] = md[i]; r["dz"] = dz[i]

    tgt = [r for r in reps if r["tgt"]]
    tgt.sort(key=lambda r: -r["md"])

    def tcol(r):
        return "#d62728" if not r["drink"] else "#1f9e1f"   # red=no-drink, green=drink

    # ---- figure ----
    ntl = len(tgt)
    ncols = 3
    tl_rows = int(np.ceil(ntl / ncols))
    fig = plt.figure(figsize=(17, 6 + 2.4 * tl_rows))
    gs = fig.add_gridspec(2 + tl_rows, ncols, hspace=0.55, wspace=0.28)

    # A PCA scatter
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(pcs[:, 0], pcs[:, 1], s=12, c="#cccccc", alpha=0.6)
    for r in tgt:
        ax.scatter(*r["pc"], s=70, c=tcol(r), edgecolor="k", zorder=5)
        ax.annotate(r["name"].split("_")[0] + "_" + r["name"][-15:-10], r["pc"],
                    fontsize=6, ha="left")
    th = np.linspace(0, 2 * np.pi, 100); L = np.linalg.cholesky(cov)
    ell = (mu[:, None] + mdcut * L @ np.array([np.cos(th), np.sin(th)])).T
    ax.plot(ell[:, 0], ell[:, 1], "r--", lw=1)
    ax.set(title="A. PCA shape space (97.5% ellipse)", xlabel="PC1", ylabel="PC2")

    # B DTW ranking
    ax = fig.add_subplot(gs[0, 1])
    order = np.argsort(dscore)
    cols = ["#cccccc"] * R
    for k, i in enumerate(order):
        if reps[i]["tgt"]:
            cols[k] = tcol(reps[i])
    ax.bar(range(R), dscore[order], color=cols)
    ax.axhline(np.median(dscore) + 3.5 / 0.6745 * (np.median(np.abs(dscore - np.median(dscore)))),
               ls="--", c="r", lw=1)
    ax.set(title="B. DTW mean-dist ranking (targets colored)", xlabel="rep (sorted)", ylabel="mean DTW")

    # C agreement
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(md, dz, s=12, c="#cccccc", alpha=0.6)
    ax.axvline(mdcut, ls="--", c="r", lw=1); ax.axhline(3.5, ls="--", c="r", lw=1)
    for r in tgt:
        ax.scatter(r["md"], r["dz"], s=70, c=tcol(r), edgecolor="k", zorder=5)
        ax.annotate(r["name"][-15:-10], (r["md"], r["dz"]), fontsize=6)
    ax.set(title="C. PCA md vs DTW z (red=no-drink)", xlabel="PCA Mahalanobis", ylabel="DTW robust-z")

    # D top-down paths
    ax = fig.add_subplot(gs[1, 0])
    for r in reps:
        if not r["tgt"]:
            ax.plot(r["norm"][:, 0], r["norm"][:, 1], c="#dddddd", lw=0.5, alpha=0.5)
    for r in tgt:
        ax.plot(r["norm"][:, 0], r["norm"][:, 1], c=tcol(r), lw=1.6)
    ax.scatter([0], [0], c="k", marker="x", s=50, zorder=6)
    ax.set(title="D. Normalized path top-down", xlabel="Δx (mm)", ylabel="Δy (mm)"); ax.axis("equal")

    # E Δz vs time
    ax = fig.add_subplot(gs[1, 1])
    for r in reps:
        if not r["tgt"]:
            ax.plot(np.arange(len(r["norm"])) / 60, r["norm"][:, 2], c="#dddddd", lw=0.5, alpha=0.5)
    for r in tgt:
        ax.plot(np.arange(len(r["norm"])) / 60, r["norm"][:, 2], c=tcol(r), lw=1.6)
    ax.set(title="E. Normalized height Δz vs time", xlabel="s", ylabel="Δz (mm)")

    # F legend / note panel
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    ax.text(0, 0.95, "Targets (sorted by PCA md):", fontsize=9, weight="bold", va="top")
    for k, r in enumerate(tgt):
        ax.text(0, 0.88 - k * 0.07,
                f"{'●' if not r['drink'] else '○'} {r['name'].replace('__pscale_4','')}\n"
                f"     md={r['md']:.1f}  dtwz={r['dz']:.1f}  drink={r['drink']}",
                fontsize=7, va="top", color=tcol(r))

    # per-target timelines
    for k, r in enumerate(tgt):
        ax = fig.add_subplot(gs[2 + k // ncols, k % ncols])
        _, xyz = sc.load_track(r["name"])
        s = sc.segment_cup_only(xyz)
        t = np.arange(len(xyz)) / sc.FPS
        ax.plot(t, s["speed"], c="#333", lw=0.9)
        ax2 = ax.twinx(); ax2.plot(t, s["disp"], c="#7a4ce8", ls="--", lw=0.9)
        for nm, a, b in s["intervals"]:
            ax.axvspan(a / sc.FPS, b / sc.FPS, color=sc.PHASE_COLORS[nm], alpha=0.30, lw=0)
        ax.set_title(f"{r['name'].replace('__pscale_4','')}  (drink={r['drink']})", fontsize=7)
        ax.set_xlim(0, t[-1]); ax.tick_params(labelsize=6); ax2.tick_params(labelsize=6)

    fig.suptitle(f"Segmentation-space dashboard — {len(tgt)} target reps vs {R}-rep cohort "
                 f"(red=no drink-dwell, green=drink)", y=0.998, fontsize=13)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
