"""Align on the 3D VELOCITY VECTORS (not speed magnitude), and use the achieved agreement as a
CORRESPONDENCE / exclusion score.

Why: speed magnitude |v| throws away direction, so two DIFFERENT motions with the same tempo
correlate ~0.9 (P16_105728: speeds overlay, but the cup moves UP in video / DOWN in mocap at the
same instant -> they don't correspond). The velocity VECTOR keeps direction. But the two streams
are in different frames, so we must fit a ROTATION to compare vectors -> joint (R, lag) fit:

  for each candidate lag: R = argmin_R sum_good |v_vid - R v_omc|^2   (velocity Procrustes),
  score = fraction of DRINK moving frames with angle(v_vid, R v_omc) < GOOD_ANG.
  pick the lag with the best score.  That best score is BOTH the alignment quality AND the
  correspondence test: a genuine rep reaches high good-frame%; a non-corresponding / mis-paired
  rep (like P16_105728) stays ~0 no matter the rotation or lag.

Compares, per rep, this best-achievable vector-agreement against the current position-fit's
drink good-frame%, and flags reps that CANNOT be aligned (correspondence failures) for exclusion.
Cache-only.  -> prints table + slides/vecalign_corr.png
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
from velfit import fit_source, _procrustes_rot

HZ = 60.0
SPEED_MM_S = 80.0
GOOD_ANG = 20.0
LAG_SEARCH = 8          # +/- frames around the stored lag
OUT = Path(__file__).resolve().parent / "slides" / "vecalign_corr.png"


def _vel60(track):
    v = resample3d(track, VIDEO_FPS)
    d = np.diff(v, axis=0) * HZ
    s = np.linalg.norm(d, axis=1)
    return d, s          # (T-1,3), (T-1,)


def _omc_vel_at_lag(omc60, lag, n):
    """OMC velocity resampled to video frames, shifted by lag (omc index = k - lag)."""
    idx = np.arange(n) - lag
    ok = (idx >= 1) & (idx < len(omc60))
    d = np.full((n, 3), np.nan)
    d[ok] = (omc60[idx[ok]] - omc60[idx[ok] - 1]) * HZ
    return d[:-1]        # match _vel60 length


def best_vector_alignment(npz):
    video = str(npz["video"])
    idx = F.align_index()
    if video not in idx:
        return None
    r = idx[video]; lag0 = r["lag"]
    tr = load_trial(r["c3d"])
    fused = np.asarray(npz["fused"], float)
    cent, rate = tr.centroid(), tr.rate
    omc60 = resample3d(cent, rate)
    vd, vs = _vel60(fused)                       # video velocity (W0), (n-1,3)
    n = len(vd) + 1
    dw = dwell_truth(tr)
    drink = np.zeros(n - 1, bool)
    sp = dw.span_at(n) if dw.span else None
    if sp:
        drink[sp[0]:min(sp[1], n - 1)] = True

    def score_lag(lag):
        od = _omc_vel_at_lag(omc60, lag, n)      # (n-1,3)
        os = np.linalg.norm(od, axis=1)
        moving = (vs > SPEED_MM_S) & (os > SPEED_MM_S) & np.isfinite(vs) & np.isfinite(os)
        fit_frames = moving & drink
        if fit_frames.sum() < 6:
            fit_frames = moving
        if fit_frames.sum() < 6:
            return np.nan, None
        R = _procrustes_rot(vd[fit_frames], od[fit_frames])   # v_vid ~ R v_omc
        rv = od @ R.T
        rs = np.linalg.norm(rv, axis=1)
        cos = np.sum(vd * rv, axis=1) / (vs * rs + 1e-9)
        ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        ev = moving & drink
        if ev.sum() < 4:
            ev = moving
        return float(np.mean(ang[ev] < GOOD_ANG)), lag

    scores = [(score_lag(L)[0], L) for L in range(lag0 - LAG_SEARCH, lag0 + LAG_SEARCH + 1)]
    scores = [(g, L) for g, L in scores if np.isfinite(g)]
    if not scores:
        return None
    best_g, best_lag = max(scores, key=lambda z: z[0])
    at_stored = dict((L, g) for g, L in scores).get(lag0, np.nan)
    return dict(video=video, best_good=best_g, best_lag=best_lag,
                stored_good=at_stored, lag0=lag0)


def _pos_drink_good(npz):
    """the CURRENT position-fit's drink good-frame% (for comparison)."""
    video = str(npz["video"]); idx = F.align_index()
    r = idx[video]; tr = load_trial(r["c3d"])
    fused = np.asarray(npz["fused"], float); src = fit_source(npz)
    fit = F.mocap_to_w0(src, tr.centroid(), tr.rate, r["lag"])
    if fit is None:
        return np.nan
    R, t, _ = fit
    vr = resample3d(fused, VIDEO_FPS); mr = resample3d(tr.centroid(), tr.rate) @ R.T + t
    lag = r["lag"]
    if lag >= 0:
        v = vr[lag:]; mo = mr[:len(v)]
    else:
        mo = mr[-lag:]; v = vr[:len(mo)]
    L = min(len(v), len(mo)); v, mo = v[:L], mo[:L]
    vm = np.diff(v, axis=0) * HZ; vo = np.diff(mo, axis=0) * HZ
    sm = np.linalg.norm(vm, axis=1); so = np.linalg.norm(vo, axis=1)
    moving = (sm > SPEED_MM_S) & (so > SPEED_MM_S) & np.isfinite(sm) & np.isfinite(so)
    dw = dwell_truth(tr); drink = np.zeros(len(vm), bool)
    sp = dw.span_at(len(vm) + 1) if dw.span else None
    if sp:
        drink[sp[0]:min(sp[1], len(vm))] = True
    ev = moving & drink
    if ev.sum() < 4:
        ev = moving
    if ev.sum() < 2:
        return np.nan
    cos = np.sum(vm[ev] * vo[ev], axis=1) / (sm[ev] * so[ev] + 1e-9)
    return float(np.mean(np.degrees(np.arccos(np.clip(cos, -1, 1))) < GOOD_ANG))


def main():
    files = sorted(glob.glob(str(F.FUSED_DIR / "*.npz")))
    print(f"vector-alignment correspondence over {len(files)} reps "
          f"(good-frame%<{GOOD_ANG:.0f}deg, drink frames)\n", flush=True)
    rows = []
    for i, f in enumerate(files):
        try:
            npz = np.load(f, allow_pickle=True)
            m = best_vector_alignment(npz)
            if m is not None:
                m["pos_good"] = _pos_drink_good(npz)
                rows.append(m)
        except Exception:
            pass
        if (i + 1) % 100 == 0 or i == len(files) - 1:
            print(f"  [{i+1}/{len(files)}] paired {len(rows)}", flush=True)

    bg = np.array([r["best_good"] for r in rows]) * 100
    pg = np.array([r["pos_good"] for r in rows if np.isfinite(r["pos_good"])]) * 100
    print(f"\n  reps: {len(rows)}")
    print(f"  BEST vector-align drink good-frame%: median {np.median(bg):.0f}%  mean {bg.mean():.0f}%")
    print(f"  position-fit      drink good-frame%: median {np.median(pg):.0f}%  mean {pg.mean():.0f}%")
    print(f"  NON-CORRESPONDING reps (best good < 25% even after vector R+lag): "
          f"{int((bg < 25).sum())}/{len(bg)}")
    print(f"  strong reps (best good >= 60%): {int((bg >= 60).sum())}/{len(bg)}", flush=True)
    print("\n  WORST correspondence (candidate exclusions):")
    print(f"  {'':<40}{'best%':>6}{'@lag':>6}{'stored%':>8}{'posfit%':>8}")
    for r in sorted(rows, key=lambda z: z["best_good"])[:15]:
        print(f"  {r['video'][:38]:<40}{r['best_good']*100:6.0f}{r['best_lag']:6d}"
              f"{r['stored_good']*100:8.0f}"
              f"{(r['pos_good']*100 if np.isfinite(r['pos_good']) else -1):8.0f}", flush=True)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.parent.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].hist(bg, bins=25, range=(0, 100), color="#4477aa", alpha=0.85)
    ax[0].axvline(25, color="r", ls="--", lw=1, label="exclusion (25%)")
    ax[0].set_xlabel("best vector-align drink good-frame %"); ax[0].set_ylabel("reps")
    ax[0].set_title(f"correspondence score\n{int((bg<25).sum())} non-corresponding (<25%)")
    ax[0].legend(fontsize=8)
    both = [(r["pos_good"] * 100, r["best_good"] * 100) for r in rows if np.isfinite(r["pos_good"])]
    px, bx = zip(*both)
    ax[1].scatter(px, bx, s=10, alpha=0.5, color="#aa3377")
    ax[1].plot([0, 100], [0, 100], "k--", lw=0.8)
    ax[1].set_xlabel("position-fit good-frame %"); ax[1].set_ylabel("vector-align good-frame %")
    ax[1].set_title("vector alignment vs position fit\n(above line = vector better)")
    fig.tight_layout(); fig.savefig(OUT, dpi=120)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
