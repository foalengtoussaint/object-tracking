"""How clean is each cup trajectory? A good drink = ONE up-and-down.

Two interpretable shape metrics on the RTS 3D track per rep:
  n_peaks  : prominent peaks in displacement-from-rest disp(t). Ideal = 1
             (rise to mouth, fall back). >1 = multiple excursions / track jumps.
  detour   : 3D arc-length / (2 * peak_disp). Ideal ~1.0 (straight up, straight
             down). Higher = wiggly path / object-hopping / jitter.

CLEAN := exactly 1 prominent peak AND detour < DETOUR_MAX.

    python experiments/drink_study/trajectory_cleanliness.py
"""
from __future__ import annotations
import json, glob
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRACK = Path("experiments/drink_study/cache/track3d")
FPS = 60.0
DETOUR_MAX = 1.6
PROM_FRAC = 0.05          # peak prominence = frac of disp range; low so a sip-time
                          # wobble (reproj drifting up/down while the cup is lost)
                          # registers as an extra peak instead of being merged away


def smooth(x, w=9):
    half = w // 2
    return np.array([np.median(x[max(0, i - half):i + half + 1]) for i in range(len(x))])


def analyze(rts):
    valid = np.isfinite(rts).all(1)
    idx = np.flatnonzero(valid)
    xyz = rts.copy()
    for ax in range(3):
        xyz[:, ax] = np.interp(np.arange(len(rts)), idx, rts[idx, ax])
    rest = np.median(xyz[:30], 0)
    disp = smooth(np.linalg.norm(xyz - rest, axis=1))
    rng = disp.max() - disp.min()
    peak = disp.max()
    if peak < 50:                     # cup barely moved -> degenerate
        return 0, np.nan, disp, peak
    prom = PROM_FRAC * rng
    pk, _ = find_peaks(disp, prominence=prom, distance=int(0.4 * FPS))
    arc = np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()
    detour = arc / (2 * peak)
    return len(pk), detour, disp, peak


def main():
    files = sorted(f for f in glob.glob(str(TRACK / "P*__pscale_4.json")) if "_summary" not in f)
    rows = []
    for f in files:
        d = json.loads(Path(f).read_text())
        rts = np.array([fr["rts"] if fr["rts"] else [np.nan] * 3 for fr in d["frames"]], float)
        if np.isfinite(rts).all(1).sum() < 10:
            continue
        npk, det, disp, peak = analyze(rts)
        clean = (npk == 1) and (det < DETOUR_MAX)
        rows.append({"name": Path(f).stem, "p": d["participant"], "npk": npk,
                     "detour": det, "disp": disp, "peak": peak, "clean": clean})
    R = len(rows)
    nclean = sum(r["clean"] for r in rows)
    print(f"{R} reps | CLEAN single up-down: {nclean} ({100*nclean/R:.0f}%)", flush=True)
    from collections import Counter
    print("n_peaks distribution:", dict(sorted(Counter(r["npk"] for r in rows).items())))
    det = np.array([r["detour"] for r in rows if np.isfinite(r["detour"])])
    print(f"detour ratio: median {np.median(det):.2f}  90th pct {np.percentile(det,90):.2f}  max {det.max():.2f}")
    print("\nworst (not clean), by detour:")
    for r in sorted([r for r in rows if not r["clean"]], key=lambda r: -(r["detour"] if np.isfinite(r["detour"]) else 0))[:12]:
        print(f"  {r['name'].replace('__pscale_4',''):42s} peaks={r['npk']} detour={r['detour']:.2f} peakdisp={r['peak']:.0f}mm", flush=True)

    # per-participant clean rate
    from collections import defaultdict
    byp = defaultdict(list)
    for r in rows:
        byp[r["p"]].append(r["clean"])
    print("\nper-participant clean rate:")
    for p in sorted(byp):
        v = byp[p]; print(f"  {p}: {sum(v)}/{len(v)} ({100*sum(v)/len(v):.0f}%)")

    # figure
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for r in rows:
        t = np.arange(len(r["disp"])) / FPS
        ax[0].plot(t, r["disp"], color=("#bbbbbb" if r["clean"] else "#d62728"),
                   lw=0.5 if r["clean"] else 1.2, alpha=0.4 if r["clean"] else 0.9)
    ax[0].set(title=f"disp-from-rest vs time (red = not clean single up-down)",
              xlabel="s", ylabel="disp (mm)")
    ax[1].hist([r["npk"] for r in rows], bins=range(0, 7), align="left", color="#4c9be8", rwidth=0.8)
    ax[1].set(title="# prominent peaks per rep (ideal=1)", xlabel="n_peaks", ylabel="reps")
    ax[2].hist(det, bins=30, color="#f0a23b")
    ax[2].axvline(DETOUR_MAX, ls="--", c="r"); ax[2].set(title="detour ratio (ideal~1.0)", xlabel="arc/(2*peak)", ylabel="reps")
    fig.tight_layout()
    out = "experiments/drink_study/cache/trajectory_cleanliness.png"
    fig.savefig(out, dpi=110)
    print(f"\nwrote {out}", flush=True)
    json.dump([{k: r[k] for k in ("name", "p", "npk", "detour", "peak", "clean")} for r in rows],
              open("experiments/drink_study/cache/trajectory_cleanliness.json", "w"), default=float)


if __name__ == "__main__":
    main()
