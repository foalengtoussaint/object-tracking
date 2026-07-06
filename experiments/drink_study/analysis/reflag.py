"""Re-run trajectory flagging on a rebuilt track set (track3d_<tag>) and compare
to the original pscale_4 flagged trials.

Metrics per rep (same as flag_trials, shape=PCA-Mahalanobis only — DTW over-flags
whole-participant clusters so it's unreliable per-trial):
  drink_fail : cup-only segmentation found no drink-dwell
  updown     : n_peaks!=1 (5% prominence) or detour>=1.6
  lateral    : PC3 (left/right) extent >= 120mm
  shape      : PCA-Mahalanobis on 50x3 resampled path > chi2 97.5%

    python experiments/drink_study/reflag.py --tag clean3d_fill
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import argparse, json, glob, sys
from pathlib import Path
import numpy as np
from scipy.stats import chi2
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
sys.path.insert(0, str(Path(__file__).resolve().parent))
import segment_cup_only as sc

FPS = 60.0

# the original pscale_4 flagged trials (drinking_right)
ORIG = ["P10_P10_drinking_right_20240202_153048", "P10_P10_drinking_right_20240202_153157",
        "P10_P10_drinking_right_20240202_153258", "P10_P10_drinking_right_20240202_153316",
        "P10_P10_drinking_right_20240202_153332", "P10_P10_drinking_right_20240202_153348",
        "P24_P24_drinking_right_20240724_105712", "P24_P24_drinking_right_20240724_105744",
        "P20_P20_drinking_right_20240505_175923", "P14_P14_drinking_right_20240221_092230"]


def resample(t, n=50):
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(t, axis=0), axis=1))]
    if seg[-1] == 0:
        return np.repeat(t[:1], n, 0)
    u = seg / seg[-1]
    return np.stack([np.interp(np.linspace(0, 1, n), u, t[:, k]) for k in range(3)], 1)


def load_xyz(d):
    fr = d["frames"]
    xyz = np.array([f["rts"] or f["kf"] or f["consensus"] or [np.nan] * 3 for f in fr], float)
    v = np.isfinite(xyz).all(1); idx = np.flatnonzero(v)
    if len(idx) < 10:
        return None
    for a in range(3):
        xyz[:, a] = np.interp(np.arange(len(xyz)), idx, xyz[idx, a])
    return xyz


def metrics(xyz):
    r = sc.segment_cup_only(xyz)
    drink = len(r["drink_runs"]) > 0
    rest = np.median(xyz[:30], 0); disp = np.linalg.norm(xyz - rest, axis=1)
    peak = disp.max(); arc = np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()
    detour = arc / (2 * peak) if peak > 50 else np.nan
    ds = np.array([np.median(disp[max(0, i - 4):i + 5]) for i in range(len(disp))])
    rng = ds.max() - ds.min()
    npk = len(find_peaks(ds, prominence=0.05 * rng, distance=int(0.4 * FPS))[0]) if rng > 0 else 0
    C = xyz - xyz.mean(0); ext = (C @ np.linalg.svd(C, full_matrices=False)[2].T); ext = ext.max(0) - ext.min(0)
    lat = float(ext[2])
    return dict(drink=bool(drink), detour=float(detour), npk=int(npk), lat=lat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    tdir = Path(f"experiments/drink_study/cache/track3d_{args.tag}")
    files = sorted(f for f in glob.glob(str(tdir / "*.json")) if "_summary" not in f)

    reps = []
    for f in files:
        d = json.loads(Path(f).read_text())
        xyz = load_xyz(d)
        if xyz is None:
            continue
        name = Path(f).stem.replace(f"__{args.tag}", "")
        m = metrics(xyz)
        reps.append(dict(name=name, rs=resample(xyz - xyz[np.isfinite(xyz).all(1)][0]), **m))
    R = len(reps)
    # PCA shape
    X = np.stack([r["rs"].ravel() for r in reps])
    pcs = PCA(2).fit_transform(X); mu = pcs.mean(0); inv = np.linalg.inv(np.cov(pcs.T))
    md = np.array([float(np.sqrt((p - mu) @ inv @ (p - mu))) for p in pcs])
    mdcut = float(np.sqrt(chi2.ppf(0.975, 2)))
    for i, r in enumerate(reps):
        r["shape"] = bool(md[i] > mdcut)
        r["updown"] = bool(r["npk"] != 1 or (np.isfinite(r["detour"]) and r["detour"] >= 1.6))
        r["lateral"] = bool(r["lat"] >= 120)
        r["drink_fail"] = bool(not r["drink"])
        r["geo_flags"] = int(r["updown"]) + int(r["lateral"])

    geo = [r for r in reps if r["updown"] or r["lateral"] or r["drink_fail"]]
    print(f"[{args.tag}] {R} reps | flagged (geo+drink): {len(geo)} | "
          f"updown {sum(r['updown'] for r in reps)} | lateral {sum(r['lateral'] for r in reps)} | "
          f"drink_fail {sum(r['drink_fail'] for r in reps)} | shape(PCA) {sum(r['shape'] for r in reps)}")
    print()
    by = {r["name"]: r for r in reps}
    print(f"  ORIGINAL 10 pscale_4-flagged trials -> status on {args.tag}:")
    for nm in ORIG:
        r = by.get(nm)
        if not r:
            print(f"    {nm:42s} (not in set)"); continue
        fl = [m for m in ("updown", "lateral", "drink_fail") if r[m]] or ["CLEAN"]
        print(f"    {nm:42s} {','.join(fl):24s} detour={r['detour']:.2f} npk={r['npk']} lat={r['lat']:.0f}")
    json.dump([{k: r[k] for k in ("name", "drink", "updown", "lateral", "shape", "detour", "npk", "lat")}
               for r in reps], open(f"experiments/drink_study/cache/flagged_{args.tag}.json", "w"), indent=1, default=float)
    print(f"\n  wrote cache/flagged_{args.tag}.json")


if __name__ == "__main__":
    main()
