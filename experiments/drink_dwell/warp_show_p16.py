"""Show the P16_105728 rep the user keeps pointing at: video vs mocap cup SPEED on one time
axis (are they on the same time scale?), + the velocity-angle timeline, under LAG-ONLY vs DTW.
-> slides/warp_p16_105728.png
"""
import glob, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, ".")
import features as F
from mocap import load_trial, resample as resample3d, VIDEO_FPS, kabsch
from truth import dwell_truth
from velfit import fit_source

HZ = 60.0; SPEED = 80.0; GOOD = 20.0
OUT = Path("slides/warp_p16_105728.png")

f = [x for x in sorted(glob.glob(str(F.FUSED_DIR / "*.npz")))
     if "P16" in x and "20240306" in x and "105728" in x][0]
d = np.load(f, allow_pickle=True); v = str(d["video"]); r = F.align_index()[v]
tr = load_trial(r["c3d"])
fused = np.asarray(d["fused"], float); src = fit_source(d)
vid60 = resample3d(fused, VIDEO_FPS); omc60 = resample3d(tr.centroid(), tr.rate)
n = len(vid60)
sv = np.nan_to_num(np.r_[0, np.linalg.norm(np.diff(vid60, axis=0), axis=1)] * 60)
so = np.nan_to_num(np.r_[0, np.linalg.norm(np.diff(omc60, axis=0), axis=1)] * 60)
dw = dwell_truth(tr); drink = np.zeros(n, bool)
sp = dw.span_at(n) if dw.span else None
if sp: drink[sp[0]:sp[1]] = True


def angle_series(omc_w0):
    vm = np.diff(vid60, axis=0) * HZ; vo = np.diff(omc_w0, axis=0) * HZ
    sm = np.linalg.norm(vm, axis=1); soo = np.linalg.norm(vo, axis=1)
    mv = (sm > SPEED) & (soo > SPEED) & np.isfinite(sm) & np.isfinite(soo)
    cos = np.sum(vm * vo, 1) / (sm * soo + 1e-9)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1))); ang[~mv] = np.nan
    idx = np.where(mv & drink[:len(mv)])[0]
    gf = float(np.mean(ang[idx] < GOOD)) if len(idx) else np.nan
    return np.r_[ang, np.nan], gf     # pad angle to n for plotting


def heading_series(omc_w0):
    """Per-frame velocity DIRECTION of each stream as azimuth+elevation (deg), on moving frames.
    If the two streams CORRESPOND, video and mocap heading curves overlay; if not, they diverge."""
    vm = np.diff(vid60, axis=0) * HZ; vo = np.diff(omc_w0, axis=0) * HZ
    sm = np.linalg.norm(vm, axis=1); soo = np.linalg.norm(vo, axis=1)
    mv = (sm > SPEED) & (soo > SPEED) & np.isfinite(sm) & np.isfinite(soo)

    def az_el(v, s):
        az = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
        el = np.degrees(np.arcsin(np.clip(v[:, 2] / (s + 1e-9), -1, 1)))
        az[~mv] = np.nan; el[~mv] = np.nan
        return np.r_[az, np.nan], np.r_[el, np.nan]
    vaz, vel = az_el(vm, sm)
    oaz, oel = az_el(vo, soo)
    return vaz, vel, oaz, oel


def fit_eval(omc_path):
    lo = np.floor(omc_path).astype(int); fr = omc_path - lo
    ok = (lo >= 0) & (lo < len(omc60) - 1)
    om = np.full((n, 3), np.nan)
    om[ok] = (1 - fr[ok])[:, None] * omc60[lo[ok]] + fr[ok][:, None] * omc60[lo[ok] + 1]
    okk = ~(np.isnan(vid60).any(1) | np.isnan(om).any(1))
    R, t, _ = kabsch(om[okk], vid60[okk], robust=True)
    for _ in range(4):
        rr = np.linalg.norm(vid60[okk] - (om[okk] @ R.T + t), axis=1); keep = rr < 15
        if keep.sum() < 10: break
        R, t, _ = kabsch(om[okk][keep], vid60[okk][keep], robust=False)
    return om @ R.T + t


# --- POSITION fit (Kabsch on centroids, current method) ---
omc_pos = fit_eval(np.arange(n) - r["lag"])
ang_pos, gf_pos = angle_series(omc_pos)

# --- VELOCITY-VECTOR fit (Procrustes on 3D velocity vectors) ---
from velfit import _procrustes_rot
from mocap import kabsch as _kab
best = (-1, None, None)   # (good, R, t)
vd_all = np.diff(vid60, axis=0) * HZ
for lag in range(r["lag"] - 8, r["lag"] + 9):
    idxo = np.arange(n) - lag
    ok = (idxo >= 1) & (idxo < len(omc60))
    od = np.full((n, 3), np.nan)
    od[ok] = (omc60[idxo[ok]] - omc60[idxo[ok] - 1]) * HZ
    od = od[:-1]
    vs2 = np.linalg.norm(vd_all, axis=1); os2 = np.linalg.norm(od, axis=1)
    mvf = (vs2 > SPEED) & (os2 > SPEED) & np.isfinite(vs2) & np.isfinite(os2)
    ff = mvf & drink[:len(mvf)]
    if ff.sum() < 6:
        ff = mvf
    if ff.sum() < 6:
        continue
    R = _procrustes_rot(vd_all[ff], od[ff])
    rv = od @ R.T; rs = np.linalg.norm(rv, axis=1)
    cos = np.sum(vd_all * rv, 1) / (vs2 * rs + 1e-9)
    ev = mvf & drink[:len(mvf)]
    if ev.sum() < 4: ev = mvf
    g = float(np.mean(np.degrees(np.arccos(np.clip(cos[ev], -1, 1))) < GOOD))
    if g > best[0]:
        best = (g, R, lag)
gf_vec, Rv, vlag = best
# recover a translation for plotting the vector-fit heading: rotate OMC velocity by Rv, at vlag
idxo = np.arange(n) - vlag
ok = (idxo >= 1) & (idxo < len(omc60))
od_full = np.full((n, 3), np.nan)
od_full[ok] = (omc60[idxo[ok]] - omc60[idxo[ok] - 1]) * HZ
# build an "omc_w0-like" velocity for heading_series: integrate not needed, pass rotated velocity
omc_vec_vel = od_full @ Rv.T                      # (n,3) rotated OMC velocity on video clock
# angle series for the vector fit (compare video velocity to rotated OMC velocity directly)
vm_p = np.diff(vid60, axis=0) * HZ
ang_vec = np.full(n, np.nan)
sm_p = np.linalg.norm(vm_p, axis=1); so_p = np.linalg.norm(omc_vec_vel[:-1], axis=1)
mvp = (sm_p > SPEED) & (so_p > SPEED) & np.isfinite(sm_p) & np.isfinite(so_p)
cosp = np.sum(vm_p * omc_vec_vel[:-1], 1) / (sm_p * so_p + 1e-9)
ap = np.degrees(np.arccos(np.clip(cosp, -1, 1))); ap[~mvp] = np.nan
ang_vec[:-1] = ap

# headings: POSITION fit vs VELOCITY-VECTOR fit
vaz, vel_, oaz_pos, oel_pos = heading_series(omc_pos)   # video + mocap(pos-fit)

def _azel(v):
    s = np.linalg.norm(v, axis=1)
    az = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
    el = np.degrees(np.arcsin(np.clip(v[:, 2] / (s + 1e-9), -1, 1)))
    bad = ~mvp
    az[bad] = np.nan; el[bad] = np.nan
    return np.r_[az, np.nan], np.r_[el, np.nan]
oaz_vec, oel_vec = _azel(omc_vec_vel[:-1])

# --- SESSION-R fit: robust session rotation, per-trial translation ---
from session_align import session_rotation
sr = session_rotation(v)
if sr is not None:
    R_sess, ntr, ninl, dev = sr
    mc = resample3d(tr.centroid(), tr.rate)
    rot = mc @ R_sess.T
    # per-trial translation given session R, on synced spatially-close frames
    lag = r["lag"]
    vr = vid60; mrr = rot
    if lag >= 0:
        vv = vr[lag:]; mm = mrr[:len(vv)]
    else:
        mm = mrr[-lag:]; vv = vr[:len(mm)]
    L = min(len(vv), len(mm)); vv, mm = vv[:L], mm[:L]
    ok = ~(np.isnan(vv).any(1) | np.isnan(mm).any(1))
    t_sess = np.nanmedian((vv - mm)[ok], axis=0) if ok.any() else np.zeros(3)
    omc_sess = fit_eval.__wrapped__ if False else None
    # build session-fit omc on video clock (same synced convention as fit_eval path)
    omc_full = resample3d(tr.centroid(), tr.rate) @ R_sess.T + t_sess
    idxo = np.arange(n) - lag
    ok2 = (idxo >= 0) & (idxo < len(omc_full))
    omc_sess_w0 = np.full((n, 3), np.nan)
    omc_sess_w0[ok2] = omc_full[idxo[ok2]]
    ang_sess, gf_sess = angle_series(omc_sess_w0)
    _, _, oaz_sess, oel_sess = heading_series(omc_sess_w0)
else:
    ang_sess = np.full(n, np.nan); gf_sess = np.nan
    oaz_sess = np.full(n, np.nan); oel_sess = np.full(n, np.nan)
    ninl = ntr = 0; dev = np.nan

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
t = np.arange(n) / 60.0
s = np.where(drink)[0]
fig, ax = plt.subplots(6, 1, figsize=(12, 15), sharex=True)
def _band(a):
    if drink.any(): a.axvspan(t[s[0]], t[s[-1]], color="#88bbdd", alpha=0.3)
# 1: speeds
ax[0].plot(t, sv, color="0.35", lw=1.4, label="VIDEO cup speed")
ax[0].plot(t, so, color="#cc3311", lw=1.4, label="MOCAP cup speed (stored lag)")
_band(ax[0]); ax[0].legend(); ax[0].set_ylabel("speed mm/s")
ax[0].set_title(f"{v}\nspeeds overlay (time is fine) — question is whether DIRECTION corresponds")
# 2: POSITION fit angle
ax[1].plot(t, ang_pos, color="#cc3311", lw=1.1); _band(ax[1])
ax[1].axhline(GOOD, color="k", ls=":", lw=0.7); ax[1].set_ylim(0, 180); ax[1].set_ylabel("angle deg")
ax[1].set_title(f"POSITION-fit angle (current)   drink good = {gf_pos*100:.0f}%")
# 3: SESSION-R fit angle  (the new idea)
ax[2].plot(t, ang_sess, color="#3366cc", lw=1.1); _band(ax[2])
ax[2].axhline(GOOD, color="k", ls=":", lw=0.7); ax[2].set_ylim(0, 180); ax[2].set_ylabel("angle deg")
ax[2].set_title(f"SESSION-R fit angle   drink good = {gf_sess*100:.0f}%  "
                f"({ninl}/{ntr} inlier trials, self-dev {dev:.0f}deg)")
# 4: VELOCITY-VECTOR fit angle
ax[3].plot(t, ang_vec, color="#227722", lw=1.1); _band(ax[3])
ax[3].axhline(GOOD, color="k", ls=":", lw=0.7); ax[3].set_ylim(0, 180); ax[3].set_ylabel("angle deg")
ax[3].set_title(f"VELOCITY-VECTOR fit angle   drink good = {gf_vec*100:.0f}%  (upper bound)")
# 5: azimuth — video vs all three mocap fits
ax[4].plot(t, vaz, color="0.2", lw=1.5, label="VIDEO")
ax[4].plot(t, oaz_pos, color="#cc3311", lw=1.0, ls="--", label="position fit")
ax[4].plot(t, oaz_sess, color="#3366cc", lw=1.3, label="session-R fit")
ax[4].plot(t, oaz_vec, color="#227722", lw=1.0, ls=":", label="velocity fit")
_band(ax[4]); ax[4].legend(fontsize=8); ax[4].set_ylabel("azimuth deg")
ax[4].set_title("velocity DIRECTION — azimuth (a fit CORRESPONDS if it overlays black VIDEO)")
# 6: elevation
ax[5].plot(t, vel_, color="0.2", lw=1.5, label="VIDEO")
ax[5].plot(t, oel_pos, color="#cc3311", lw=1.0, ls="--", label="position fit")
ax[5].plot(t, oel_sess, color="#3366cc", lw=1.3, label="session-R fit")
ax[5].plot(t, oel_vec, color="#227722", lw=1.0, ls=":", label="velocity fit")
_band(ax[5]); ax[5].legend(fontsize=8); ax[5].set_ylabel("elevation deg"); ax[5].set_xlabel("time (s)")
ax[5].set_title("velocity DIRECTION — elevation (up/down)")
fig.tight_layout(); OUT.parent.mkdir(exist_ok=True); fig.savefig(OUT, dpi=120)
print("wrote", OUT, f"  position {gf_pos*100:.0f}%  session-R {gf_sess*100:.0f}%  "
      f"velocity {gf_vec*100:.0f}%")
