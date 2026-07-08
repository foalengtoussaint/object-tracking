"""Per-rep TIMELINE of the MMC-vs-OMC velocity angle — is the error a CLUSTER at the drink
with clean non-drink around it (like the distance plot), or noisy throughout?

Draws, for the N worst reps (by median moving-frame angle), the per-frame angle over time
with the drink-phase span shaded. Also overlays cup speed (grey) so you can see that the
spikes sit where the cup actually moves. Cache-only.  -> slides/mmc_omc_angle_timeline.png
"""
from __future__ import annotations
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # drink_dwell root (archived)
import features as F
from mocap import load_trial, resample as resample3d, VIDEO_FPS
from truth import dwell_truth
from velfit import fit_source

OUT = Path(__file__).resolve().parent / "slides" / "mmc_omc_angle_timeline.png"
SPEED_MM_S = 80.0
HZ = 60.0


def _synced_pair(cup_world, mocap_centroid, rate, lag, R, t):
    vr = resample3d(cup_world, VIDEO_FPS)
    mr = resample3d(mocap_centroid, rate) @ R.T + t
    if lag >= 0:
        v = vr[lag:]; mo = mr[:len(v)]
    else:
        mo = mr[-lag:]; v = vr[:len(mo)]
    L = min(len(v), len(mo))
    return v[:L], mo[:L]


def _vel(xyz):
    d = np.diff(xyz, axis=0) * HZ
    d = np.vstack([d, d[-1:]])
    return d, np.linalg.norm(d, axis=1)


def rep_timeline(npz):
    video = str(npz["video"])
    idx = F.align_index()
    if video not in idx:
        return None
    r = idx[video]
    tr = load_trial(r["c3d"])
    fused = np.asarray(npz["fused"], float)
    src = fit_source(npz)      # KF-RTS masked to detected frames
    fit = F.mocap_to_w0(src, tr.centroid(), tr.rate, r["lag"])
    if fit is None:
        return None
    R, t, _ = fit
    mmc, omc = _synced_pair(fused, tr.centroid(), tr.rate, r["lag"], R, t)
    vm, sm = _vel(mmc)
    vo, so = _vel(omc)
    moving = (sm > SPEED_MM_S) & (so > SPEED_MM_S) & np.isfinite(sm) & np.isfinite(so)
    if moving.sum() < 5:
        return None
    cosang = np.sum(vm * vo, axis=1) / (sm * so + 1e-9)
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    ang[~moving] = np.nan                              # only meaningful when moving
    dm = np.zeros(len(ang), bool)
    dw = dwell_truth(tr)
    sp = dw.span_at(len(ang)) if dw.span else None
    if sp:
        dm[sp[0]:sp[1]] = True
    med = float(np.nanmedian(ang[moving]))
    return dict(video=video, ang=ang, drink=dm, speed=sm, med=med,
                nd_med=float(np.nanmedian(ang[moving & ~dm])) if (moving & ~dm).any() else np.nan,
                dr_med=float(np.nanmedian(ang[moving & dm])) if (moving & dm).any() else np.nan)


def main():
    files = sorted(glob.glob(str(F.FUSED_DIR / "*.npz")))
    print(f"scanning {len(files)} reps for angle timelines...\n", flush=True)
    reps = []
    for i, f in enumerate(files):
        try:
            a = rep_timeline(np.load(f, allow_pickle=True))
        except Exception:
            a = None
        if a is not None:
            reps.append(a)
        if (i + 1) % 150 == 0 or i == len(files) - 1:
            print(f"  [{i+1}/{len(files)}] {len(reps)} paired", flush=True)

    reps.sort(key=lambda z: -z["med"])
    worst = reps[:9]
    print("\n  worst 9 (median moving-angle):")
    for a in worst:
        print(f"  {a['video'][:34]:<36} med={a['med']:5.1f}  "
              f"non-drink={a['nd_med']:5.1f}  drink={a['dr_med']:5.1f}", flush=True)

    # population summary of the drink-vs-nondrink RATIO to answer the pattern question directly
    nd = np.array([a["nd_med"] for a in reps if np.isfinite(a["nd_med"])])
    dr = np.array([a["dr_med"] for a in reps if np.isfinite(a["dr_med"])])
    print(f"\n  ACROSS ALL {len(reps)} reps (per-rep medians):")
    print(f"    non-drink angle: median {np.median(nd):.1f}deg")
    print(f"    drink  angle:    median {np.median(dr):.1f}deg")
    both = [(a["nd_med"], a["dr_med"]) for a in reps
            if np.isfinite(a["nd_med"]) and np.isfinite(a["dr_med"])]
    worse_in_drink = sum(1 for n, d in both if d > n + 5)
    print(f"    reps where drink is >5deg WORSE than non-drink: {worse_in_drink}/{len(both)}"
          f"  ({100*worse_in_drink/len(both):.0f}%)", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.parent.mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(16, 9))
    for ax, a in zip(axes.ravel(), worst):
        x = np.arange(len(a["ang"]))
        ax.plot(x, a["ang"], color="#cc3311", lw=1.0, label="angle")
        # shade drink span
        dm = a["drink"]
        if dm.any():
            s = np.where(dm)[0]
            ax.axvspan(s[0], s[-1], color="#88bbdd", alpha=0.35, label="drink")
        # speed on a twin axis (context: spikes where cup moves)
        ax2 = ax.twinx()
        ax2.plot(x, a["speed"], color="0.6", lw=0.6, alpha=0.7)
        ax2.set_ylim(0, max(300, np.nanmax(a["speed"]) * 1.05)); ax2.set_yticks([])
        ax.axhline(30, color="k", ls=":", lw=0.7)
        ax.set_ylim(0, 180); ax.set_ylabel("angle (deg)", fontsize=8)
        ax.set_title(f"{a['video'][:30]}\nnd {a['nd_med']:.0f}  drink {a['dr_med']:.0f}",
                     fontsize=8)
    fig.suptitle("MMC-OMC velocity angle over time — worst 9 reps "
                 "(blue = drink span, grey = cup speed, dotted = 30deg)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, dpi=110)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
