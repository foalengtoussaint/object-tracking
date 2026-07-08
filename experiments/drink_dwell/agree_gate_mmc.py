"""If we drop the BAD-MMC frames (few cameras / high reprojection), what happens to the
MMC<->OMC agreement?

Under the session-R fit, recompute the good-frame% (velocity angle < 20deg) two ways:
  RAW    all moving frames
  GATED  only frames where the video track is TRUSTWORTHY (ncams >= MIN_CAMS and med_px <= MAX_PX)

If the disagreement is a tracking problem, gating should lift agreement a lot on the hard reps
and leave good reps ~unchanged. Also report how many frames survive the gate (coverage cost).
Cache-only.  -> slides/agree_gate_mmc.png
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F
from mocap import load_trial
from session_align import alignment_for
import agreement as AG            # THE shared sync/angle/quality math (no more copy-paste)

MIN_CAMS, MAX_PX, GOOD = AG.MIN_CAMS, AG.MAX_PX, AG.GOOD_ANG
OUT = Path(__file__).resolve().parent / "slides" / "agree_gate_mmc.png"


def rep(npz):
    v = str(npz["video"]); idx = F.align_index()
    if v not in idx:
        return None
    r = idx[v]; tr = load_trial(r["c3d"]); lag = r["lag"]
    fused = np.asarray(npz["fused"], float)
    al = alignment_for(v, "session", npz=npz, tr=tr)
    if al is None:
        return None
    R, t, _ = al
    v_s, mo_s = AG.sync_tracks(fused, tr.centroid(), tr.rate, lag)
    omc = mo_s @ R.T + t
    ang, mv = AG.velocity_angles(v_s, omc)         # shared
    q = AG.mmc_quality(v)
    if q is None:
        return None
    ncams, mpx = q
    n = len(ang)
    nc = AG.align_quality_to_grid(ncams, lag, n)
    mp = AG.align_quality_to_grid(np.nan_to_num(mpx, nan=99.0), lag, n)
    if mv.sum() < 10:
        return None
    goodmmc = (nc >= MIN_CAMS) & (mp <= MAX_PX)
    raw = float(np.mean(ang[mv] < GOOD))
    gate_mask = mv & goodmmc[:len(mv)]
    gated = float(np.mean(ang[gate_mask] < GOOD)) if gate_mask.sum() >= 5 else np.nan
    coverage = float(gate_mask.sum() / mv.sum())
    return dict(video=v, raw=raw, gated=gated, coverage=coverage)


def main():
    files = sorted(glob.glob(str(F.FUSED_DIR / "*.npz")))
    print(f"MMC-gated agreement over {len(files)} reps "
          f"(gate: ncams>={MIN_CAMS} & px<={MAX_PX})\n", flush=True)
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

    raw = np.array([r["raw"] for r in rows]) * 100
    g = np.array([r["gated"] for r in rows if np.isfinite(r["gated"])]) * 100
    cov = np.array([r["coverage"] for r in rows if np.isfinite(r["gated"])]) * 100
    # matched: reps with both
    both = [(r["raw"] * 100, r["gated"] * 100, r["coverage"] * 100) for r in rows if np.isfinite(r["gated"])]
    br, bg, bc = np.array([x[0] for x in both]), np.array([x[1] for x in both]), np.array([x[2] for x in both])
    print(f"\n  reps: {len(rows)}  (gated-scorable: {len(g)})")
    print(f"  RAW   all-moving good-frame%: median {np.median(raw):.0f}%  mean {raw.mean():.0f}%")
    print(f"  GATED good-MMC   good-frame%: median {np.median(g):.0f}%  mean {g.mean():.0f}%")
    print(f"  frame coverage after gate: median {np.median(cov):.0f}%  (how many frames survive)")
    print(f"  reps improved by gating (>5pts): {int((bg > br + 5).sum())}/{len(both)}")
    print(f"\n  reps most lifted by dropping bad-MMC frames:")
    for vr_, gr_, cv_ in sorted(both, key=lambda z: -(z[1] - z[0]))[:12]:
        # find the rep name
        pass
    order = sorted([r for r in rows if np.isfinite(r["gated"])], key=lambda z: -(z["gated"] - z["raw"]))
    for r in order[:12]:
        print(f"    {r['video'][:40]:<42} raw {r['raw']*100:3.0f}% -> gated {r['gated']*100:3.0f}%  "
              f"(coverage {r['coverage']*100:3.0f}%)", flush=True)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.parent.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].hist(br, bins=25, range=(0, 100), color="#cc6677", alpha=0.6, label=f"raw (med {np.median(br):.0f})")
    ax[0].hist(bg, bins=25, range=(0, 100), color="#228833", alpha=0.6, label=f"MMC-gated (med {np.median(bg):.0f})")
    ax[0].legend(); ax[0].set_xlabel("good-frame %"); ax[0].set_ylabel("reps")
    ax[0].set_title("agreement: all frames vs good-MMC-only frames")
    ax[1].scatter(br, bg, s=10, alpha=0.4, c=bc, cmap="viridis")
    ax[1].plot([0, 100], [0, 100], "k--", lw=0.8)
    ax[1].set_xlabel("raw good-frame %"); ax[1].set_ylabel("MMC-gated good-frame %")
    cb = fig.colorbar(ax[1].collections[0], ax=ax[1]); cb.set_label("frame coverage %")
    ax[1].set_title("above line = dropping bad-MMC frames helped")
    fig.tight_layout(); fig.savefig(OUT, dpi=120)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
