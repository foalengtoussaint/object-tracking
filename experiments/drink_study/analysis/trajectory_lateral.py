"""How much does each cup path wander LEFT/RIGHT (out of its reach-lift plane)?

A drink reach is ~planar (forward reach + vertical lift = a sagittal plane). The
medio-lateral (side-to-side) excursion should be small. The world frame is the
per-session board frame, so we find the plane per-rep with PCA:

  PC1, PC2 = the reach-lift plane (largest spread)
  PC3      = out-of-plane thickness = LEFT/RIGHT wander  <-- should be small

Metrics per rep:
  lat_mm   = extent along PC3 (mm)         -- absolute side-to-side travel
  lat_frac = PC3 extent / PC1 extent       -- relative to the main motion
PLANAR := lat_mm < LAT_ABS and lat_frac < LAT_FRAC.

    python experiments/drink_study/trajectory_lateral.py
"""
from __future__ import annotations
import json, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRACK = Path("experiments/drink_study/cache/track3d")
FPS = 60.0
LAT_ABS = 120.0      # mm out-of-plane
LAT_FRAC = 0.25      # PC3/PC1


def load_xyz(d):
    rts = np.array([fr["rts"] if fr["rts"] else [np.nan] * 3 for fr in d["frames"]], float)
    valid = np.isfinite(rts).all(1)
    if valid.sum() < 10:
        return None
    idx = np.flatnonzero(valid)
    for ax in range(3):
        rts[:, ax] = np.interp(np.arange(len(rts)), idx, rts[idx, ax])
    return rts


def main():
    files = sorted(f for f in glob.glob(str(TRACK / "P*__pscale_4.json")) if "_summary" not in f)
    rows = []
    for f in files:
        d = json.loads(Path(f).read_text())
        xyz = load_xyz(d)
        if xyz is None:
            continue
        C = xyz - xyz.mean(0)
        # PCA via SVD
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        proj = C @ Vt.T                       # coords in PC frame
        ext = proj.max(0) - proj.min(0)       # extent along PC1,PC2,PC3
        if ext[0] < 50:                       # cup barely moved
            continue
        lat_mm = float(ext[2]); lat_frac = float(ext[2] / ext[0])
        rows.append({"name": Path(f).stem, "p": d["participant"], "ext": ext,
                     "lat_mm": lat_mm, "lat_frac": lat_frac, "proj": proj,
                     "planar": lat_mm < LAT_ABS and lat_frac < LAT_FRAC})
    R = len(rows)
    npl = sum(r["planar"] for r in rows)
    lat = np.array([r["lat_mm"] for r in rows]); frac = np.array([r["lat_frac"] for r in rows])
    print(f"{R} reps | PLANAR (little left/right): {npl} ({100*npl/R:.0f}%)", flush=True)
    print(f"left/right extent mm : median {np.median(lat):.0f}  90th {np.percentile(lat,90):.0f}  max {lat.max():.0f}")
    print(f"PC3/PC1 fraction     : median {np.median(frac):.2f}  90th {np.percentile(frac,90):.2f}  max {frac.max():.2f}")
    print("\nworst (most left/right wander), by lat_mm:")
    for r in sorted(rows, key=lambda r: -r["lat_mm"])[:12]:
        print(f"  {r['name'].replace('__pscale_4',''):42s} lat={r['lat_mm']:5.0f}mm  frac={r['lat_frac']:.2f}  "
              f"(reach PC1={r['ext'][0]:.0f}mm)", flush=True)

    from collections import defaultdict
    byp = defaultdict(list)
    for r in rows:
        byp[r["p"]].append(r["lat_mm"])
    print("\nper-participant median left/right wander (mm):")
    print("  " + "  ".join(f"{p}:{np.median(v):.0f}" for p, v in sorted(byp.items())))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].hist(lat, bins=30, color="#4c9be8"); ax[0].axvline(LAT_ABS, ls="--", c="r")
    ax[0].set(title="left/right (PC3) extent (mm), ideal small", xlabel="mm", ylabel="reps")
    ax[1].hist(frac, bins=30, color="#f0a23b"); ax[1].axvline(LAT_FRAC, ls="--", c="r")
    ax[1].set(title="PC3/PC1 fraction (lateral vs main motion)", xlabel="fraction", ylabel="reps")
    # PC1 (reach) vs PC3 (lateral) projection — clean reps hug the horizontal axis
    for r in rows:
        c = "#bbbbbb" if r["planar"] else "#d62728"
        ax[2].plot(r["proj"][:, 0], r["proj"][:, 2], color=c,
                   lw=0.5 if r["planar"] else 1.3, alpha=0.35 if r["planar"] else 0.9)
    ax[2].set(title="path in reach(PC1) vs left-right(PC3) plane (red=not planar)",
              xlabel="PC1 reach (mm)", ylabel="PC3 left/right (mm)"); ax[2].axis("equal")
    fig.tight_layout()
    out = "experiments/drink_study/cache/trajectory_lateral.png"
    fig.savefig(out, dpi=110); print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
