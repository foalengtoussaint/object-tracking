"""Is the KF actually helping? Compare three reconstructions of the cup track,
all from the SAME cached per-frame data (no GPU, no re-run):

  RTS         : consensus -> causal KF -> RTS smooth   (current pipeline output)
  CONSENSUS   : raw gated triangulation only, linear-interpolated over gaps
                (anchored to real triangulations -> cannot ballistically diverge)

For each rep we measure how far the KF strayed from the consensus, then run the
cup-only segmentation + trajectory-cleanliness metrics on BOTH and see which is
cleaner. If consensus-only is as good or better, the KF is adding divergence
bugs more than value.

    python experiments/drink_study/test_no_kf.py
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import json, glob
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import segment_cup_only as sc
from scipy.signal import find_peaks

TRACK = Path("experiments/drink_study/cache/track3d")
FPS = 60.0


def fill(track):
    """Linear-interp a (T,3) array with NaN gaps; anchored, no extrapolation drift."""
    T = len(track); out = track.copy()
    v = np.isfinite(track).all(1); idx = np.flatnonzero(v)
    if len(idx) < 2:
        return out, v
    for a in range(3):
        out[:, a] = np.interp(np.arange(T), idx, track[idx, a])
    return out, v


def metrics(xyz):
    """drink-dwell?, detour, n_peaks(5%), lat_mm — on a (T,3) track."""
    r = sc.segment_cup_only(xyz)
    drink = len(r["drink_runs"]) > 0
    rest = np.median(xyz[:30], 0)
    disp = np.linalg.norm(xyz - rest, axis=1)
    peak = disp.max()
    arc = np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()
    detour = arc / (2 * peak) if peak > 50 else np.nan
    ds = np.array([np.median(disp[max(0, i - 4):i + 5]) for i in range(len(disp))])
    rng = ds.max() - ds.min()
    npk = len(find_peaks(ds, prominence=0.05 * rng, distance=int(0.4 * FPS))[0]) if rng > 0 else 0
    # lateral (PC3 extent)
    C = xyz - xyz.mean(0)
    ext = (C @ np.linalg.svd(C, full_matrices=False)[2].T); ext = ext.max(0) - ext.min(0)
    lat = float(ext[2])
    clean = (npk == 1) and (detour < 1.6) and (lat < 120)
    return dict(drink=drink, detour=float(detour), npk=npk, lat=lat, clean=bool(clean))


def main():
    files = sorted(f for f in glob.glob(str(TRACK / "P*__pscale_4.json")) if "_summary" not in f)
    rows = []
    for f in files:
        d = json.loads(Path(f).read_text()); fr = d["frames"]
        cons = np.array([x["consensus"] if x["consensus"] else [np.nan] * 3 for x in fr], float)
        rts = np.array([x["rts"] if x["rts"] else [np.nan] * 3 for x in fr], float)
        kf = np.array([x["kf"] if x["kf"] else [np.nan] * 3 for x in fr], float)
        cons_f, cmask = fill(cons)
        rts_f, _ = fill(rts)
        if cmask.sum() < 10:
            continue
        # KF stray from consensus (where both exist)
        both = cmask & np.isfinite(kf).all(1)
        stray = np.linalg.norm(kf[both] - cons[both], axis=1) if both.sum() else np.array([0.0])
        m_rts = metrics(rts_f); m_con = metrics(cons_f)
        rows.append(dict(name=Path(f).stem.replace("__pscale_4", ""), p=d["participant"],
                         cons_cov=float(cmask.mean()), kf_stray_max=float(stray.max()),
                         kf_stray_med=float(np.median(stray)),
                         rts=m_rts, con=m_con))
    R = len(rows)
    big = [r for r in rows if r["kf_stray_max"] > 200]
    print(f"{R} reps")
    print(f"KF strays >200mm from consensus in {len(big)}/{R} reps "
          f"(>500mm in {sum(r['kf_stray_max']>500 for r in rows)})")
    print(f"drink-dwell:  RTS {sum(r['rts']['drink'] for r in rows)}/{R}   "
          f"CONSENSUS {sum(r['con']['drink'] for r in rows)}/{R}")
    print(f"clean traj :  RTS {sum(r['rts']['clean'] for r in rows)}/{R}   "
          f"CONSENSUS {sum(r['con']['clean'] for r in rows)}/{R}")
    print(f"median detour: RTS {np.median([r['rts']['detour'] for r in rows if np.isfinite(r['rts']['detour'])]):.3f}  "
          f"CONSENSUS {np.median([r['con']['detour'] for r in rows if np.isfinite(r['con']['detour'])]):.3f}")
    print()
    print("reps where KF strays >200mm but CONSENSUS is clean (KF-only failures):")
    fixed = [r for r in rows if r["kf_stray_max"] > 200 and r["con"]["clean"] and not r["rts"]["clean"]]
    for r in sorted(fixed, key=lambda r: -r["kf_stray_max"]):
        print(f"  {r['name']:42s} kf_stray_max={r['kf_stray_max']:5.0f}mm  "
              f"RTS(clean={r['rts']['clean']},drink={r['rts']['drink']},detour={r['rts']['detour']:.2f})  "
              f"-> CON(clean={r['con']['clean']},drink={r['con']['drink']})")
    print(f"\n{len(fixed)} reps recover (clean on consensus, broken on RTS)")
    json.dump(rows, open("experiments/drink_study/cache/test_no_kf.json", "w"), default=float, indent=1)


if __name__ == "__main__":
    main()
