"""Consolidate the 3 trajectory-weirdness detectors into one flagged-trials list.

  M1 shape    : cohort shape outlier (PCA-Mahalanobis on 50x3 resampled path  OR
                DTW robust-z). flagged if md>chi2_97.5 or dtw_z>3.5
  M2 up-down  : not a clean single up-and-down (n_peaks!=1 or detour>=1.6)
  M3 lateral  : not planar (PC3 left/right extent>=120mm or PC3/PC1>=0.25)
  (+ drink_fail: cup-only segmentation found no drink-dwell — supplementary)

    python experiments/drink_study/flag_trials.py
"""
import json, glob
from pathlib import Path
import numpy as np
from scipy.stats import chi2
from sklearn.decomposition import PCA

CACHE = Path("experiments/drink_study/cache")
NRES = 50


def resample(t, n):
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(t, axis=0), axis=1))]
    if seg[-1] == 0:
        return np.repeat(t[:1], n, 0)
    u = seg / seg[-1]; g = np.linspace(0, 1, n)
    return np.stack([np.interp(g, u, t[:, k]) for k in range(3)], 1)


def rz(x):
    x = np.asarray(x, float); m = np.median(x); d = np.median(np.abs(x - m)) or 1e-9
    return 0.6745 * (x - m) / d


reps = []
for f in sorted(glob.glob(str(CACHE / "track3d" / "P*__pscale_4.json"))):
    if "_summary" in f:
        continue
    d = json.load(open(f))
    rts = np.array([fr["rts"] if fr["rts"] else [np.nan] * 3 for fr in d["frames"]], float)
    vmask = np.isfinite(rts).all(1)
    if vmask.sum() < 10:
        continue
    idx = np.flatnonzero(vmask)
    for a in range(3):
        rts[:, a] = np.interp(np.arange(len(rts)), idx, rts[idx, a])
    C = rts - rts.mean(0)
    ext = (C @ np.linalg.svd(C, full_matrices=False)[2].T)
    ext = ext.max(0) - ext.min(0)
    lat_mm = float(ext[2]); lat_frac = float(ext[2] / ext[0]) if ext[0] > 0 else 0.0
    reps.append({"name": Path(f).stem, "rs": resample(rts - rts[0], NRES),
                 "lat_mm": lat_mm, "planar": lat_mm < 120 and lat_frac < 0.25})

names = [r["name"] for r in reps]; R = len(reps)
X = np.stack([r["rs"].ravel() for r in reps])
pcs = PCA(2).fit_transform(X)
mu = pcs.mean(0); inv = np.linalg.inv(np.cov(pcs.T))
md = np.array([float(np.sqrt((p - mu) @ inv @ (p - mu))) for p in pcs])
mdcut = float(np.sqrt(chi2.ppf(0.975, 2)))
z = np.load(CACHE / "seg_dtw.npy", allow_pickle=True).item()
dz = rz(z["Dm"].sum(1) / (R - 1)) if z["names"] == names else np.zeros(R)

drink = {r["rep"]: r["drink"] for r in json.load(open(CACHE / "cup_only_all.json"))}
clean = {r["name"]: r for r in json.load(open(CACHE / "trajectory_cleanliness.json"))}

rows = []
for i, r in enumerate(reps):
    nm = r["name"]; mdv = float(md[i]); dzv = float(dz[i])
    m1 = bool(mdv > mdcut or dzv > 3.5)
    c = clean.get(nm, {}); m2 = bool(not c.get("clean", True))
    m3 = bool(not r["planar"])
    df = bool(not drink.get(nm, True))
    if m1 or m2 or m3 or df:
        rows.append(dict(trial=nm.replace("__pscale_4", ""), shape=m1, updown=m2, lateral=m3,
                         drink_fail=df, n_methods=int(m1) + int(m2) + int(m3),
                         pca_md=mdv, dtw_z=dzv, detour=c.get("detour"), n_peaks=c.get("npk"),
                         lat_mm=r["lat_mm"]))
rows.sort(key=lambda r: (-r["n_methods"], -int(r["drink_fail"]), -r["pca_md"]))

print(f"{R} reps total | {len(rows)} flagged by >=1 method\n")
print(f"{'trial':40s} shp updn lat drkX | nM   md  dtwz det pk latmm")
for r in rows:
    g = lambda b: " X " if b else " . "
    det = r["detour"]; det = f"{det:.2f}" if det is not None else "  - "
    print(f"{r['trial']:40s}{g(r['shape'])}{g(r['updown'])}{g(r['lateral'])}{g(r['drink_fail'])}| "
          f"{r['n_methods']}  {r['pca_md']:4.1f} {r['dtw_z']:5.1f} {det} {r['n_peaks']:2d} {r['lat_mm']:5.0f}")

print(f"\nFlagged by ALL 3 trajectory methods: {[r['trial'] for r in rows if r['n_methods']==3] or 'none'}")
print(f"Flagged by >=2 trajectory methods : {[r['trial'] for r in rows if r['n_methods']>=2] or 'none'}")
print(f"per-method totals: shape={sum(r['shape'] for r in rows)}  up-down={sum(r['updown'] for r in rows)}  "
      f"lateral={sum(r['lateral'] for r in rows)}  drink-fail={sum(r['drink_fail'] for r in rows)}")
json.dump(rows, open(CACHE / "flagged_trials.json", "w"), indent=1, default=float)
print(f"\nwrote {CACHE/'flagged_trials.json'}")
