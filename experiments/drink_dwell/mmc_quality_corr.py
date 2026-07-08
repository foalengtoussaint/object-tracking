"""Across ALL reps: is the MMC<->OMC disagreement driven by MMC (video-track) quality?

Per frame we have an INDEPENDENT MMC-quality signal from the track3d JSON (ncams = how many
cameras agreed, median_px = reprojection spread) -- no mocap involved. We correlate that against
the MMC<->OMC velocity-angle disagreement (under the session fit). If disagreement lives where the
video track is unreliable (few cams / high px), the residual is a TRACKING problem, not alignment.

Per rep: corr(angle, ncams), corr(angle, med_px), and median angle on BAD-MMC vs GOOD-MMC frames.
Then the population distribution + a scatter.  Cache-only.  -> slides/mmc_quality_corr.png
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F
from mocap import load_trial
from session_align import alignment_for
import agreement as AG            # shared sync/angle/quality math

OUT = Path(__file__).resolve().parent / "slides" / "mmc_quality_corr.png"


def rep(npz):
    v = str(npz["video"]); idx = F.align_index()
    if v not in idx:
        return None
    r = idx[v]; tr = load_trial(r["c3d"]); lag = r["lag"]
    fused = np.asarray(npz["fused"], float)
    al = alignment_for(v, "session", npz=npz, tr=tr)
    if al is None:
        return None
    R, t, info = al
    v_s, mo_s = AG.sync_tracks(fused, tr.centroid(), tr.rate, lag)
    ang, mv = AG.velocity_angles(v_s, mo_s @ R.T + t)
    q = AG.mmc_quality(v)
    if q is None:
        return None
    ncams, mpx = q
    n = len(ang)
    nc = AG.align_quality_to_grid(ncams, lag, n)
    mp = AG.align_quality_to_grid(mpx, lag, n)
    m2 = mv & np.isfinite(nc)[:len(mv)] & np.isfinite(mp)[:len(mv)]
    A = ang[m2]; NC = nc[:len(mv)][m2]; MP = mp[:len(mv)][m2]
    if len(A) < 15 or NC.std() < 1e-6:
        return None
    cN = float(np.corrcoef(A, NC)[0, 1])
    cP = float(np.corrcoef(A, MP)[0, 1]) if MP.std() > 1e-6 else np.nan
    bad = (NC < 4) | (MP > np.nanpercentile(MP, 75))
    ang_bad = float(np.median(A[bad])) if bad.any() else np.nan
    ang_good = float(np.median(A[~bad])) if (~bad).any() else np.nan
    return dict(video=v, cN=cN, cP=cP, ang_bad=ang_bad, ang_good=ang_good, n=len(A))


def main():
    files = sorted(glob.glob(str(F.FUSED_DIR / "*.npz")))
    print(f"MMC-quality vs MMC-OMC disagreement over {len(files)} reps\n", flush=True)
    rows = []
    for i, f in enumerate(files):
        try:
            m = rep(np.load(f, allow_pickle=True))
        except Exception:
            m = None
        if m is not None:
            rows.append(m)
        if (i + 1) % 100 == 0 or i == len(files) - 1:
            print(f"  [{i+1}/{len(files)}] {len(rows)} reps", flush=True)

    cN = np.array([r["cN"] for r in rows if np.isfinite(r["cN"])])
    ab = np.array([r["ang_bad"] for r in rows if np.isfinite(r["ang_bad"]) and np.isfinite(r["ang_good"])])
    ag = np.array([r["ang_good"] for r in rows if np.isfinite(r["ang_bad"]) and np.isfinite(r["ang_good"])])
    print(f"\n  reps: {len(rows)}")
    print(f"  corr(angle, ncams): median {np.median(cN):+.2f}  "
          f"(negative = fewer cams -> worse disagreement)")
    print(f"  reps with corr < -0.3 (MMC-badness drives disagreement): {int((cN < -0.3).sum())}/{len(cN)}")
    print(f"  reps with corr > +0.3 (opposite):                        {int((cN > 0.3).sum())}/{len(cN)}")
    print(f"  angle on BAD-MMC frames:  median {np.median(ab):.0f} deg")
    print(f"  angle on GOOD-MMC frames: median {np.median(ag):.0f} deg")
    print(f"  reps where bad-MMC angle > good-MMC by >10deg: "
          f"{int((ab > ag + 10).sum())}/{len(ab)}  ({100*(ab>ag+10).mean():.0f}%)", flush=True)
    print(f"\n  worst reps (most negative corr = disagreement tied to MMC-badness):")
    for r in sorted([r for r in rows if np.isfinite(r['cN'])], key=lambda z: z['cN'])[:12]:
        print(f"    {r['video'][:40]:<42} corr={r['cN']:+.2f}  bad={r['ang_bad']:3.0f} good={r['ang_good']:3.0f} n={r['n']}", flush=True)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.parent.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].hist(cN, bins=30, range=(-1, 1), color="#4477aa", alpha=0.85)
    ax[0].axvline(0, color="k", lw=0.8); ax[0].axvline(np.median(cN), color="r", ls="--", lw=1)
    ax[0].set_xlabel("corr(disagreement angle, ncams) per rep"); ax[0].set_ylabel("reps")
    ax[0].set_title(f"disagreement vs MMC-quality\nmedian corr {np.median(cN):+.2f} "
                    f"({int((cN<-0.3).sum())} reps < -0.3)")
    ax[1].scatter(ag, ab, s=10, alpha=0.4, color="#cc6677")
    lim = max(np.percentile(ab, 98), np.percentile(ag, 98))
    ax[1].plot([0, lim], [0, lim], "k--", lw=0.8)
    ax[1].set_xlabel("median angle on GOOD-MMC frames")
    ax[1].set_ylabel("median angle on BAD-MMC frames")
    ax[1].set_title("above line = bad-MMC frames disagree more\n(the disagreement is a tracking problem)")
    fig.tight_layout(); fig.savefig(OUT, dpi=120)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
