"""Per-trial diagnostic visualization for the flagged drink_study reps.

For each target trial, one PNG with four panels:
  A. segmentation timeline: cup speed + disp-from-rest + phase bands, with the
     cup-LOST frames (kept<3 cams = interpolated) shaded grey.
  B. camera coverage: # cams kept by triangulation per frame (the gaps).
  C. sagittal view: path in the validated (front, up) plane  -- the up-and-down.
  D. top-down view : path in the validated (front, left/right) plane -- lateral.

The (up, front, lateral) frame is recovered per rep from the physical endpoints
(rest -> apex = up) + the trajectory's best-fit plane (PC3 = left/right), the
construction validated earlier (lift perpendicular to PC3 to ~0.4 deg).

    python experiments/drink_study/viz_flagged.py            # all geometric+seg flagged
    python experiments/drink_study/viz_flagged.py P10_..._153258
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import segment_cup_only as sc

CACHE = Path("experiments/drink_study/cache")
TRACK = CACHE / "track3d"
OUT = CACHE / "flagged_viz"
FPS = 60.0


def frame_axes(rts):
    """Validated (up, front, lateral) frame from rest->apex + best-fit plane."""
    rest = np.median(rts[:30], 0)
    disp = np.linalg.norm(rts - rest, axis=1)
    apex = rts[int(np.argmax(disp))]
    up = apex - rest; up = up / (np.linalg.norm(up) + 1e-9)
    C = rts - rts.mean(0)
    Vt = np.linalg.svd(C, full_matrices=False)[2]
    pc1 = Vt[0]
    front = pc1 - (pc1 @ up) * up; front = front / (np.linalg.norm(front) + 1e-9)
    lateral = np.cross(up, front); lateral = lateral / (np.linalg.norm(lateral) + 1e-9)
    return rest, up, front, lateral


def viz(track_stem: str, flags: dict, out: Path):
    d = json.loads((TRACK / f"{track_stem}.json").read_text())
    fr = d["frames"]
    rts_raw = np.array([f["rts"] if f["rts"] else [np.nan] * 3 for f in fr], float)
    kept = np.array([len(f["kept"]) for f in fr])
    vmask = np.isfinite(rts_raw).all(1)
    idx = np.flatnonzero(vmask)
    rts = rts_raw.copy()
    for a in range(3):
        rts[:, a] = np.interp(np.arange(len(rts)), idx, rts[idx, a])

    _, seg_xyz = sc.load_track(track_stem)
    s = sc.segment_cup_only(seg_xyz)
    t = np.arange(len(rts)) / FPS
    lost = kept < 3
    rest, up, front, lateral = frame_axes(rts)
    C = rts - rest
    f_up = C @ up; f_front = C @ front; f_lat = C @ lateral
    # longest lost run
    best = cur = 0
    for x in lost:
        cur = cur + 1 if x else 0; best = max(best, cur)

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22)

    # A timeline
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, s["speed"], color="#333", lw=1.0, label="speed mm/s")
    ax2 = ax.twinx(); ax2.plot(t, s["disp"], color="#7a4ce8", ls="--", lw=1.0, label="disp mm")
    for nm, a, b in s["intervals"]:
        ax.axvspan(a / FPS, b / FPS, color=sc.PHASE_COLORS[nm], alpha=0.25, lw=0)
    # shade cup-lost
    for a, b in _runs(lost):
        ax.axvspan(a / FPS, b / FPS, color="k", alpha=0.12, lw=0)
    ax.set(title="A. segmentation (grey hatch = cup lost / interpolated)",
           xlabel="s", ylabel="speed mm/s"); ax2.set_ylabel("disp mm")
    ax.set_xlim(0, t[-1])

    # B coverage
    ax = fig.add_subplot(gs[0, 1])
    ax.fill_between(t, kept, step="mid", color="#4c9be8", alpha=0.7)
    ax.axhline(3, ls="--", c="r", lw=1, label="min 3 cams")
    ax.set(title=f"B. cameras kept (longest lost run {best/FPS:.1f}s)",
           xlabel="s", ylabel="# cams"); ax.set_ylim(0, 10); ax.legend(fontsize=8)

    # C sagittal front-up
    ax = fig.add_subplot(gs[1, 0])
    sc_pts = ax.scatter(f_front, f_up, c=t, cmap="viridis", s=10)
    ax.plot(f_front, f_up, color="#999", lw=0.5, alpha=0.5)
    ax.scatter([0], [0], c="k", marker="x", s=60, label="rest")
    ax.set(title="C. sagittal: front vs up (the up-and-down)",
           xlabel="front (mm)", ylabel="up (mm)"); ax.axis("equal"); ax.legend(fontsize=8)
    plt.colorbar(sc_pts, ax=ax, label="t (s)", fraction=0.046)

    # D top-down front-lateral
    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(f_front, f_lat, c=t, cmap="viridis", s=10)
    ax.plot(f_front, f_lat, color="#999", lw=0.5, alpha=0.5)
    ax.scatter([0], [0], c="k", marker="x", s=60)
    ax.set(title=f"D. top-down: front vs left/right (lat={flags.get('lat_mm',0):.0f}mm)",
           xlabel="front (mm)", ylabel="left/right (mm)"); ax.axis("equal")

    fl = [m for m in ("updown", "lateral", "drink_fail", "shape") if flags.get(m)]
    fig.suptitle(f"{track_stem.replace('__pscale_4','')}   flags: {', '.join(fl)}   "
                 f"| md={flags.get('pca_md',0):.1f} dtwz={flags.get('dtw_z',0):.1f} "
                 f"detour={flags.get('detour',0):.2f} peaks={flags.get('n_peaks')} "
                 f"lost={best/FPS:.1f}s", fontsize=12)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}", flush=True)


def _runs(mask):
    out, i, T = [], 0, len(mask)
    while i < T:
        if mask[i]:
            j = i + 1
            while j < T and mask[j]:
                j += 1
            out.append((i, j)); i = j
        else:
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reps", nargs="*")
    args = ap.parse_args()
    rows = {r["trial"]: r for r in json.loads((CACHE / "flagged_trials.json").read_text())}
    if args.reps:
        targets = [r.replace("__pscale_4", "") for r in args.reps]
    else:
        targets = [t for t, r in rows.items() if r["updown"] or r["lateral"] or r["drink_fail"]]
    print(f"{len(targets)} trials to visualize", flush=True)
    for tr in sorted(targets):
        flags = rows.get(tr, {})
        viz(f"{tr}__pscale_4", flags, OUT / f"{tr}.png")


if __name__ == "__main__":
    main()
