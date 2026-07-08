"""How well does the VIDEO cup track (MMC) agree with the MOCAP cup (OMC)?

For every pairable rep, align the mocap cup-centroid into W0 with the SAME shared
`features.mocap_to_w0` (fit on the RAW consensus track, hard-exclude), then measure the
per-frame 3D distance between the MMC cup and the OMC cup. We report three views because
they answer different questions:

  ALL      every overlapping frame  -> the honest overall agreement
  KEPT     frames the fit trusts (residual < ALIGN_EXCLUDE_MM) -> "when it's good, how good"
  DRINK    frames inside the truth dwell span -> the tilt regime (cup 4-marker centroid vs
           the video cup point diverge ~50mm when the cup tilts to the mouth)

Prints a per-rep table (flush) and the population percentiles, and writes a 3-panel
histogram to slides/mmc_omc_agree.png.  Cache-only: reads drink_study caches, no GPU.
"""
from __future__ import annotations
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F
from mocap import load_trial
from truth import dwell_truth
import agreement as AG            # shared sync math

OUT = Path(__file__).resolve().parent / "slides" / "mmc_omc_agree.png"


def rep_agreement(npz):
    """-> dict(video, all, kept, drink) each an array of per-frame MMC-OMC mm, or None."""
    video = str(npz["video"])
    idx = F.align_index()
    if video not in idx:
        return None
    r = idx[video]
    tr = load_trial(r["c3d"])
    fused = np.asarray(npz["fused"], float)
    raw = np.asarray(npz["cons"], float) if "cons" in npz else fused
    fit = F.mocap_to_w0(raw, tr.centroid(), tr.rate, r["lag"])   # fit on RAW consensus
    if fit is None:
        return None
    R, t, _ = fit
    mmc, mo = AG.sync_tracks(fused, tr.centroid(), tr.rate, r["lag"])   # shared
    omc = mo @ R.T + t
    d = np.linalg.norm(mmc - omc, axis=1)
    ok = np.isfinite(d)
    d_all = d[ok]
    if d_all.size < 5:
        return None
    kept = d[ok] < F.ALIGN_EXCLUDE_MM
    # drink-phase frames: map the truth dwell span onto this synced index (len == len(d))
    dw = dwell_truth(tr)
    drink_mask = np.zeros(len(d), bool)
    if dw.span:
        sp = dw.span_at(len(d))
        if sp:
            drink_mask[sp[0]:sp[1]] = True
    return dict(video=video,
                all=d_all,
                kept=d[ok][kept],
                drink=d[ok & drink_mask],
                good_frac=float(np.mean(d_all < F.ALIGN_EXCLUDE_MM)))


def main():
    files = sorted(glob.glob(str(F.FUSED_DIR / "*.npz")))
    print(f"MMC vs OMC cup agreement over {len(files)} candidate reps\n", flush=True)
    rows, all_d, kept_d, drink_d, good_fracs = [], [], [], [], []
    for i, f in enumerate(files):
        try:
            npz = np.load(f, allow_pickle=True)
            a = rep_agreement(npz)
        except Exception as e:
            print(f"  [{i+1}/{len(files)}] {Path(f).stem}: SKIP ({e})", flush=True)
            continue
        if a is None:
            continue
        med = float(np.median(a["all"]))
        p90 = float(np.percentile(a["all"], 90))
        kmed = float(np.median(a["kept"])) if a["kept"].size else float("nan")
        dmed = float(np.median(a["drink"])) if a["drink"].size else float("nan")
        rows.append((a["video"], med, p90, kmed, dmed, a["good_frac"], a["all"].size))
        all_d.append(a["all"]); kept_d.append(a["kept"]); drink_d.append(a["drink"])
        good_fracs.append(a["good_frac"])
        if (i + 1) % 50 == 0 or i == len(files) - 1:
            print(f"  [{i+1}/{len(files)}] paired so far: {len(rows)}", flush=True)

    if not rows:
        print("no pairable reps"); return
    all_d = np.concatenate(all_d)
    kept_d = np.concatenate([k for k in kept_d if k.size])
    drink_d = np.concatenate([d for d in drink_d if d.size])
    good_fracs = np.asarray(good_fracs)

    def pct(x):
        return {p: float(np.percentile(x, p)) for p in (50, 75, 90, 95)}

    # WORST reps by good-frame fraction (the ones whose MMC cup barely agrees with OMC)
    print(f"\n{'':<30}{'med':>7}{'p90':>7}{'kept-med':>10}{'drink-med':>11}{'good%':>7}{'N':>7}")
    for v, med, p90, km, dm, gf, n in sorted(rows, key=lambda z: z[5])[:12]:
        print(f"  {v:<28}{med:7.1f}{p90:7.1f}{km:10.1f}{dm:11.1f}{gf*100:6.0f}%{n:7d}", flush=True)
    print(f"\n  reps paired: {len(rows)}")
    print(f"  ALL   frames med/p75/p90/p95 mm: {pct(all_d)}")
    print(f"  KEPT  frames med/p75/p90/p95 mm: {pct(kept_d)}")
    print(f"  DRINK frames med/p75/p90/p95 mm: {pct(drink_d)}")
    print(f"  GOOD-FRAME FRAC per rep (<{F.ALIGN_EXCLUDE_MM:.0f}mm):"
          f" median {np.median(good_fracs)*100:.0f}%  mean {good_fracs.mean()*100:.0f}%"
          f"  |  reps >=80%: {int((good_fracs>=0.8).sum())}/{len(good_fracs)}"
          f"  reps <50%: {int((good_fracs<0.5).sum())}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.parent.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    for ax, (name, data, col) in zip(axes[:3], [
            ("ALL frames", all_d, "#4477aa"),
            ("KEPT frames (fit-trusted, <%.0fmm)" % F.ALIGN_EXCLUDE_MM, kept_d, "#228833"),
            ("DRINK-phase frames (tilt)", drink_d, "#cc6677")]):
        clip = np.clip(data, 0, 120)
        ax.hist(clip, bins=60, color=col, alpha=0.85)
        m = np.median(data)
        ax.axvline(m, color="k", ls="--", lw=1)
        ax.set_title(f"{name}\nmed {m:.1f}mm  N={data.size}", fontsize=10)
        ax.set_xlabel("MMC-OMC cup distance (mm)")
        ax.set_ylabel("frames")
    # PANEL 4: per-rep good-frame fraction — "how much of each rep does the video cup agree?"
    ax = axes[3]
    ax.hist(good_fracs * 100, bins=25, range=(0, 100), color="#aa3377", alpha=0.85)
    ax.axvline(np.median(good_fracs) * 100, color="k", ls="--", lw=1)
    ax.set_title(f"per-rep good-frame %  (<{F.ALIGN_EXCLUDE_MM:.0f}mm)\n"
                 f"median {np.median(good_fracs)*100:.0f}%  N={len(good_fracs)} reps", fontsize=10)
    ax.set_xlabel("% of frames MMC agrees with OMC")
    ax.set_ylabel("reps")
    fig.suptitle("Video (MMC) vs mocap (OMC) cup-track agreement — shared raw-fit alignment",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, dpi=110)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
