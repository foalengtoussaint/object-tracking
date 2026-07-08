"""Does the VIDEO cup MOVE in the same DIRECTION as the MOCAP cup?

Distance (mmc_omc_agree.py) conflates two errors: a translation offset (cup 4-marker
centroid sits ~5cm from the video cup point when the cup tilts) and a ROTATION error (a
bad Kabsch fit rotates one track relative to the other). The velocity DIRECTION isolates
rotation: a pure translation offset leaves the frame-to-frame velocity vector unchanged,
but a wrong fit rotation tilts it. So we compare the per-frame velocity ANGLE:

  angle(t) = arccos( v_mmc . v_omc / (|v_mmc| |v_omc|) )   in W0, after the shared raw-fit

measured only on MOVING frames (both speeds > SPEED_MM_S) because direction is noise when
the cup is nearly still. We report the population angle distribution, per-rep median angle,
and split by phase (drink vs. non-drink), plus the same restricted to the reps whose raw/
fused Kabsch fits FLIP (>30deg) to see if those are the reps whose motion direction is off.

Cache-only, no GPU.  -> slides/mmc_omc_angle.png
"""
from __future__ import annotations
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F
from mocap import load_trial
from truth import dwell_truth
import agreement as AG            # shared sync math

OUT = Path(__file__).resolve().parent / "slides" / "mmc_omc_angle.png"
SPEED_MM_S = AG.SPEED_MM_S   # only score frames where BOTH cups move faster than this (mm/s)
HZ = AG.HZ


def _synced_pair(cup_world, mocap_centroid, rate, lag, R, t):
    """thin wrapper on the shared sync_tracks, then transform the mocap side into W0."""
    v, mo = AG.sync_tracks(cup_world, mocap_centroid, rate, lag)
    return v, mo @ R.T + t


def _vel(xyz):
    """frame-to-frame velocity (mm/s) and speed, NaN-safe (NaN where either endpoint NaN)."""
    d = np.diff(xyz, axis=0) * HZ
    d = np.vstack([d, d[-1:]])           # pad to len
    sp = np.linalg.norm(d, axis=1)
    return d, sp


def rep_angles(npz):
    """-> dict(video, ang(deg on moving frames), drink_mask_on_moving, flip_deg) or None."""
    video = str(npz["video"])
    idx = F.align_index()
    if video not in idx:
        return None
    r = idx[video]
    tr = load_trial(r["c3d"])
    fused = np.asarray(npz["fused"], float)
    raw = np.asarray(npz["cons"], float) if "cons" in npz else fused
    fit = F.mocap_to_w0(raw, tr.centroid(), tr.rate, r["lag"])          # raw-fit
    if fit is None:
        return None
    R, t, _ = fit
    # how much does the fused-fit differ in rotation from the raw-fit? (the "flip" magnitude)
    fit_f = F.mocap_to_w0(fused, tr.centroid(), tr.rate, r["lag"])
    flip = np.nan
    if fit_f is not None:
        Rf = fit_f[0]
        c = (np.trace(R.T @ Rf) - 1) / 2
        flip = float(np.degrees(np.arccos(np.clip(c, -1, 1))))

    mmc, omc = _synced_pair(fused, tr.centroid(), tr.rate, r["lag"], R, t)
    vm, sm = _vel(mmc)
    vo, so = _vel(omc)
    moving = (sm > SPEED_MM_S) & (so > SPEED_MM_S) & np.isfinite(sm) & np.isfinite(so)
    if moving.sum() < 5:
        return None
    cosang = np.sum(vm * vo, axis=1) / (sm * so + 1e-9)
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    # drink-phase mask on the synced index
    dm = np.zeros(len(ang), bool)
    dw = dwell_truth(tr)
    if dw.span:
        sp = dw.span_at(len(ang))
        if sp:
            dm[sp[0]:sp[1]] = True
    return dict(video=video, ang=ang[moving], drink=dm[moving], flip=flip)


def main():
    files = sorted(glob.glob(str(F.FUSED_DIR / "*.npz")))
    print(f"MMC vs OMC cup velocity-DIRECTION angle over {len(files)} reps "
          f"(moving frames only, speed>{SPEED_MM_S:.0f}mm/s)\n", flush=True)
    rows, all_ang, drink_ang, nondrink_ang, flip_ang = [], [], [], [], []
    for i, f in enumerate(files):
        try:
            npz = np.load(f, allow_pickle=True)
            a = rep_angles(npz)
        except Exception as e:
            print(f"  [{i+1}/{len(files)}] {Path(f).stem}: SKIP ({e})", flush=True)
            continue
        if a is None:
            continue
        med = float(np.median(a["ang"]))
        good = float(np.mean(a["ang"] < 30.0))       # frac of moving frames within 30deg
        rows.append((a["video"], med, good, a["flip"], a["ang"].size))
        all_ang.append(a["ang"])
        if a["drink"].any():
            drink_ang.append(a["ang"][a["drink"]])
        if (~a["drink"]).any():
            nondrink_ang.append(a["ang"][~a["drink"]])
        if np.isfinite(a["flip"]) and a["flip"] > 30.0:
            flip_ang.append(a["ang"])
        if (i + 1) % 100 == 0 or i == len(files) - 1:
            print(f"  [{i+1}/{len(files)}] paired: {len(rows)}", flush=True)

    if not rows:
        print("no pairable reps"); return
    all_ang = np.concatenate(all_ang)
    drink_ang = np.concatenate(drink_ang) if drink_ang else np.array([])
    nondrink_ang = np.concatenate(nondrink_ang) if nondrink_ang else np.array([])
    flip_ang = np.concatenate(flip_ang) if flip_ang else np.array([])

    def pct(x):
        return {p: round(float(np.percentile(x, p)), 1) for p in (50, 75, 90)} if x.size else {}

    print(f"\n  WORST reps by median velocity-angle (deg):")
    print(f"  {'':<30}{'med_ang':>8}{'<30deg%':>9}{'flip':>7}{'N':>7}")
    for v, med, good, flip, n in sorted(rows, key=lambda z: -z[1])[:12]:
        print(f"  {v:<28}{med:8.1f}{good*100:8.0f}%{flip:7.0f}{n:7d}", flush=True)
    print(f"\n  reps: {len(rows)}")
    print(f"  ALL moving frames    angle med/p75/p90: {pct(all_ang)}")
    print(f"  DRINK-phase frames   angle med/p75/p90: {pct(drink_ang)}")
    print(f"  NON-drink frames     angle med/p75/p90: {pct(nondrink_ang)}")
    print(f"  FLIP reps (>30deg)   angle med/p75/p90: {pct(flip_ang)}  ({len(flip_ang)} frames)")
    med_per_rep = np.array([r[1] for r in rows])
    print(f"  per-rep median angle: median {np.median(med_per_rep):.1f}deg  "
          f"reps <15deg: {(med_per_rep<15).sum()}/{len(rows)}  reps >45deg: {(med_per_rep>45).sum()}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.parent.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (name, data, col) in zip(axes, [
            ("ALL moving frames", all_ang, "#4477aa"),
            ("DRINK vs NON-drink", None, None),
            ("per-rep median angle", med_per_rep, "#aa3377")]):
        if name.startswith("DRINK"):
            ax.hist(nondrink_ang, bins=45, range=(0, 180), color="#4477aa", alpha=0.6,
                    density=True, label=f"non-drink (med {np.median(nondrink_ang):.0f})")
            ax.hist(drink_ang, bins=45, range=(0, 180), color="#cc6677", alpha=0.6,
                    density=True, label=f"drink (med {np.median(drink_ang):.0f})")
            ax.legend(fontsize=9); ax.set_xlabel("velocity angle (deg)")
            ax.set_title("MMC vs OMC motion direction by phase", fontsize=10)
        elif name.startswith("per-rep"):
            ax.hist(data, bins=40, range=(0, 90), color=col, alpha=0.85)
            ax.axvline(np.median(data), color="k", ls="--", lw=1)
            ax.set_title(f"{name}\nmed {np.median(data):.0f}deg  N={len(data)} reps", fontsize=10)
            ax.set_xlabel("per-rep median angle (deg)"); ax.set_ylabel("reps")
        else:
            ax.hist(data, bins=45, range=(0, 180), color=col, alpha=0.85)
            ax.axvline(np.median(data), color="k", ls="--", lw=1)
            ax.set_title(f"{name}\nmed {np.median(data):.0f}deg  N={data.size}", fontsize=10)
            ax.set_xlabel("velocity angle (deg)"); ax.set_ylabel("frames")
    fig.suptitle("MMC vs OMC cup velocity-DIRECTION agreement (moving frames, shared raw-fit)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, dpi=110)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
