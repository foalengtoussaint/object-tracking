"""VELOCITY-driven mocap->W0 rotation fit, vs the position-Kabsch.

The position Kabsch fits R,t from cup CENTROIDS. The 4-marker mocap centroid can't constrain
spin about the cup's near-vertical axis (rotational-symmetry degeneracy) and is offset from the
video cup point when the cup tilts. The exclude-threshold sweep confirmed tightening the fit
does NOT reduce the residual drink-phase direction error -> that error is rotation ambiguity, not
tilt pollution. VELOCITY direction carries the missing rotation info and is translation-invariant
(the tilt offset is a constant, differentiates away), so a velocity-Procrustes can pin the axis
the centroid leaves free.

fit_velocity(mmc_vel, omc_vel):  R = argmin_R  sum |v_mmc - R v_omc|^2   (rotation-only, SVD),
on GOOD frames only (both cups moving > SPEED_MM_S AND position-residual < POS_GATE_MM so we
don't align direction on spatially-divergent junk). Translation recovered from the position
centroids given R.

CLI compares, per rep, the velocity-fit vs the position-fit on the drink-phase angle + the
rotation difference between the two fits. Cache-only.  -> slides/velfit_compare.png
"""
from __future__ import annotations
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F
from mocap import load_trial, resample as resample3d, VIDEO_FPS, kabsch
from truth import dwell_truth

OUT = Path(__file__).resolve().parent / "slides" / "velfit_compare.png"
SPEED_MM_S = 80.0        # a frame's velocity direction is only trustworthy above this speed
POS_GATE_MM = 15.0       # only fit direction where the two cups are also spatially close
HZ = 60.0


def fit_source(npz):
    """The MMC track to FIT the alignment on: KF-RTS smoothed BUT only on frames that had a
    real detection (drop the coasted/gap-filled frames the KF invents). This gives the KF's
    per-frame noise-smoothing on genuine measurements, without the overshoot/coast frames that
    pull the rotation (why plain `fused` was bad) and without raw's per-frame jitter. NaN where
    there was no detection so downstream sync/fit skips those frames.  Returns (T,3)."""
    kf = np.asarray(npz["kf"], float).copy()
    if "valid" in npz:
        valid = np.asarray(npz["valid"], bool)
    elif "cons" in npz:
        valid = np.isfinite(np.asarray(npz["cons"], float)).all(1)
    else:
        return kf
    kf[~valid] = np.nan
    return kf


def _rot_between(R0, R1):
    c = (np.trace(R0.T @ R1) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def _procrustes_rot(A, B):
    """R minimising sum |A - R B|^2 over rows (rotation only, no translation, no scale)."""
    H = B.T @ A
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def _synced(cup_world, mocap_centroid, rate, lag):
    """MMC cup + OMC centroid (lab frame, NOT yet in W0), synced at lag."""
    vr = resample3d(cup_world, VIDEO_FPS)
    mr = resample3d(mocap_centroid, rate)
    if lag >= 0:
        v = vr[lag:]; mo = mr[:len(v)]
    else:
        mo = mr[-lag:]; v = vr[:len(mo)]
    L = min(len(v), len(mo))
    return v[:L], mo[:L]


def _vel(xyz):
    d = np.diff(xyz, axis=0) * HZ
    return np.vstack([d, d[-1:]]), None


def velocity_fit(mmc, omc_lab, pos_R):
    """R,t: mocap-lab -> W0 via VELOCITY Procrustes on good frames. pos_R = the position fit's
    rotation, used only to pick which frames are spatially close (position gate). Returns
    (R, t, n_good) or None."""
    vm = np.diff(mmc, axis=0)
    vo = np.diff(omc_lab, axis=0)
    sm = np.linalg.norm(vm, axis=1) * HZ
    so = np.linalg.norm(vo, axis=1) * HZ
    # position residual under the POSITION fit (to gate on spatial closeness)
    t_pos = mmc.mean(0) - omc_lab.mean(0) @ pos_R.T
    resid = np.linalg.norm(mmc - (omc_lab @ pos_R.T + t_pos), axis=1)[:-1]
    good = (sm > SPEED_MM_S) & (so > SPEED_MM_S) & (resid < POS_GATE_MM) \
        & np.isfinite(sm) & np.isfinite(so)
    if good.sum() < 8:
        return None
    R = _procrustes_rot(vm[good], vo[good])
    t = mmc.mean(0) - omc_lab.mean(0) @ R.T          # translation from position centroids
    return R, t, int(good.sum())


def _angle(mmc, omc_w0, drink):
    vm = np.diff(mmc, axis=0) * HZ; vo = np.diff(omc_w0, axis=0) * HZ
    sm = np.linalg.norm(vm, axis=1); so = np.linalg.norm(vo, axis=1)
    moving = (sm > SPEED_MM_S) & (so > SPEED_MM_S) & np.isfinite(sm) & np.isfinite(so)
    if moving.sum() < 5:
        return np.nan, np.nan
    cos = np.sum(vm * vo, axis=1) / (sm * so + 1e-9)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    dm = drink[:len(ang)] & moving
    return (float(np.median(ang[moving])),
            float(np.median(ang[dm])) if dm.any() else np.nan)


def rep_compare(npz):
    video = str(npz["video"])
    idx = F.align_index()
    if video not in idx:
        return None
    r = idx[video]
    tr = load_trial(r["c3d"])
    fused = np.asarray(npz["fused"], float)
    src = fit_source(npz)    # KF-RTS masked to detected frames
    pfit = F.mocap_to_w0(src, tr.centroid(), tr.rate, r["lag"])   # POSITION fit
    if pfit is None:
        return None
    pR, pt, _ = pfit
    mmc_r, omc_lab = _synced(src, tr.centroid(), tr.rate, r["lag"])
    vf = velocity_fit(mmc_r, omc_lab, pR)
    if vf is None:
        return None
    vR, vt, ngood = vf
    rotdiff = _rot_between(pR, vR)
    # evaluate BOTH fits' drink-phase angle on the FUSED track
    mmc_f, omc_labf = _synced(fused, tr.centroid(), tr.rate, r["lag"])
    dw = dwell_truth(tr)
    drink = np.zeros(len(mmc_f), bool)
    sp = dw.span_at(len(mmc_f)) if dw.span else None
    if sp:
        drink[sp[0]:sp[1]] = True
    _, pos_drink = _angle(mmc_f, omc_labf @ pR.T + pt, drink)
    _, vel_drink = _angle(mmc_f, omc_labf @ vR.T + vt, drink)
    return dict(video=video, rotdiff=rotdiff, pos_drink=pos_drink,
                vel_drink=vel_drink, ngood=ngood)


def main():
    files = sorted(glob.glob(str(F.FUSED_DIR / "*.npz")))
    print(f"velocity-fit vs position-fit over {len(files)} reps\n", flush=True)
    rows = []
    for i, f in enumerate(files):
        try:
            m = rep_compare(np.load(f, allow_pickle=True))
        except Exception:
            m = None
        if m is not None:
            rows.append(m)
        if (i + 1) % 150 == 0 or i == len(files) - 1:
            print(f"  [{i+1}/{len(files)}] paired {len(rows)}", flush=True)

    pd = np.array([r["pos_drink"] for r in rows if np.isfinite(r["pos_drink"]) and np.isfinite(r["vel_drink"])])
    vd = np.array([r["vel_drink"] for r in rows if np.isfinite(r["pos_drink"]) and np.isfinite(r["vel_drink"])])
    rot = np.array([r["rotdiff"] for r in rows])
    print(f"\n  reps compared: {len(pd)}")
    print(f"  DRINK-phase angle  position-fit: median {np.median(pd):.1f}deg")
    print(f"  DRINK-phase angle  velocity-fit: median {np.median(vd):.1f}deg")
    improved = int((vd < pd - 2).sum()); worsened = int((vd > pd + 2).sum())
    print(f"  velocity-fit BETTER (>2deg): {improved}/{len(pd)}   WORSE: {worsened}")
    print(f"  rotation diff pos-vs-vel fit: median {np.median(rot):.1f}deg  "
          f"p90 {np.percentile(rot,90):.1f}  reps>30deg {(rot>30).sum()}")
    print("\n  reps where velocity-fit helps most (drink-angle drop):")
    order = sorted([r for r in rows if np.isfinite(r["pos_drink"]) and np.isfinite(r["vel_drink"])],
                   key=lambda z: z["vel_drink"] - z["pos_drink"])[:10]
    print(f"  {'':<34}{'pos':>6}{'vel':>7}{'rotdiff':>9}{'ngood':>7}")
    for r in order:
        print(f"  {r['video'][:32]:<34}{r['pos_drink']:6.0f}{r['vel_drink']:7.0f}"
              f"{r['rotdiff']:9.0f}{r['ngood']:7d}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.parent.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].scatter(pd, vd, s=10, alpha=0.5, color="#4477aa")
    lim = max(pd.max(), vd.max()) * 1.05
    ax[0].plot([0, lim], [0, lim], "k--", lw=0.8)
    ax[0].set_xlabel("position-fit drink angle (deg)")
    ax[0].set_ylabel("velocity-fit drink angle (deg)")
    ax[0].set_title(f"per-rep drink-phase angle\nbelow line = velocity-fit better "
                    f"({improved} reps)", fontsize=10)
    ax[1].hist(rot, bins=40, range=(0, 180), color="#aa3377", alpha=0.85)
    ax[1].axvline(np.median(rot), color="k", ls="--", lw=1)
    ax[1].set_xlabel("rotation diff: position-fit vs velocity-fit (deg)")
    ax[1].set_ylabel("reps")
    ax[1].set_title(f"how far the two fits' rotations differ\nmed {np.median(rot):.0f}deg",
                    fontsize=10)
    fig.tight_layout(); fig.savefig(OUT, dpi=110)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
