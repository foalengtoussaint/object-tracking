"""Cup-derived Murphy endpoint measures + combine with our trajectory geometry.

The fast Murphy model (iMOVE imove_extensions/murphy_measures.py) computes its
endpoint-kinematic measures from the dominant HAND site speed over the drink-task
phases. The cup IS the end-effector (hand holds it), so we can compute the pose-free
subset from our 3D cup track:
  total_movement_time, peak_velocity, time_to_peak (abs/%), number_of_movement_units.

We reuse Murphy's EXACT velocity/movement-unit logic (4Hz butter, median prefilter,
MU amplitude 60 mm/s, 150ms gap) via the installed module, mapped to our cup-only
phases (no 'reaching' — use forward_transport as the cup's transport-to-mouth onset).

Then we combine with our geometry (detour, lateral, n_peaks, drink dwell) into one
per-rep table and look at how the kinematic smoothness (movement units) relates to
the geometric flags.

    python experiments/drink_study/cup_murphy.py --tag clean3d_refill
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import argparse, glob, importlib.util, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import segment_cup_only as sc

# import the real Murphy module by path (cross-project, not on sys.path)
MURPHY = "/home/imove/Documents/iMOVE/DEV/imove_extensions/imove_extensions/murphy_measures.py"
spec = importlib.util.spec_from_file_location("murphy_measures", MURPHY)
mm = importlib.util.module_from_spec(spec)
sys.modules["murphy_measures"] = mm          # needed so @dataclass can resolve the module
spec.loader.exec_module(mm)

FPS = 60.0


def load_xyz(d):
    fr = d["frames"]
    xyz = np.array([f["rts"] or f["kf"] or f["consensus"] or [np.nan] * 3 for f in fr], float)
    v = np.isfinite(xyz).all(1); idx = np.flatnonzero(v)
    if len(idx) < 10:
        return None
    for a in range(3):
        xyz[:, a] = np.interp(np.arange(len(xyz)), idx, xyz[idx, a])
    return xyz


def cup_measures(xyz):
    seg = sc.segment_cup_only(xyz)
    iv = seg["intervals"]
    # Murphy hand-speed pipeline on the cup: smooth xyz (4Hz butter + median3) then diff
    hand_xyz = mm._smoothed_xyz(xyz, fs=FPS, cutoff=mm.DEFAULT_LOWPASS_HZ, order=mm.DEFAULT_BUTTER_ORDER)
    speed = mm._hand_speed_mmps(hand_xyz, fps=FPS)

    fwd = mm._phase_slice(iv, "forward_transport")
    drink = mm._phase_slice(iv, "drinking")
    rest_post = mm._phase_slice(iv, "rest_post")
    # movement window: cup lift (fwd start) -> back at rest (rest_post start)
    mstart = fwd[0] if fwd else 0
    mend = rest_post[0] if rest_post else len(xyz)
    movement_time = (mend - mstart) / FPS

    # peak velocity + timing over the forward transport (cup reach to mouth)
    if fwd and fwd[1] > fwd[0]:
        s = speed[fwd[0]:fwd[1]]
        pv = float(np.max(s)); ttp = int(np.argmax(s)) / FPS
        ttp_pct = int(np.argmax(s)) / max(fwd[1] - fwd[0], 1) * 100
    else:
        pv = ttp = ttp_pct = float("nan")

    # movement units over transport phases (exclude drinking + rests), Murphy rule
    gap = max(int(mm.DEFAULT_MU_TIME_GAP_S * FPS), 1)
    nmu = 0
    for ph in ("forward_transport", "back_transport"):
        sl = mm._phase_slice(iv, ph)
        if sl:
            nmu += mm._count_movement_units(speed[sl[0]:sl[1]], mm.DEFAULT_MU_AMPLITUDE_MMPS, gap)

    # our geometry
    rest = np.median(xyz[:30], 0); disp = np.linalg.norm(xyz - rest, axis=1)
    peak = disp.max(); arc = np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()
    detour = arc / (2 * peak) if peak > 50 else np.nan
    C = xyz - xyz.mean(0); ext = (C @ np.linalg.svd(C, full_matrices=False)[2].T); ext = ext.max(0) - ext.min(0)
    lat = float(ext[2])
    drink_dur = (drink[1] - drink[0]) / FPS if drink else 0.0

    return dict(movement_time=movement_time, peak_velocity=pv, time_to_peak=ttp,
                ttp_pct=ttp_pct, n_movement_units=nmu, drink_dur=drink_dur,
                detour=float(detour), lat_mm=lat, drink=bool(seg["drink_runs"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="clean3d_refill")
    args = ap.parse_args()
    files = sorted(f for f in glob.glob(f"experiments/drink_study/cache/track3d_{args.tag}/*.json")
                   if "_summary" not in f)
    rows = []
    for f in files:
        xyz = load_xyz(json.loads(Path(f).read_text()))
        if xyz is None:
            continue
        name = Path(f).stem.replace(f"__{args.tag}", "")
        rows.append(dict(name=name, **cup_measures(xyz)))
    R = len(rows)
    arr = lambda k: np.array([r[k] for r in rows if np.isfinite(r[k])], float)
    print(f"[{args.tag}] {R} reps — cup-derived Murphy endpoint measures (median [IQR]):")
    for k, unit in [("movement_time", "s"), ("peak_velocity", "mm/s"), ("time_to_peak", "s"),
                    ("n_movement_units", ""), ("drink_dur", "s")]:
        a = arr(k); q1, q2, q3 = np.percentile(a, [25, 50, 75])
        print(f"   {k:18s} {q2:7.2f} [{q1:.2f}, {q3:.2f}] {unit}")
    # combine: does smoothness (movement units) relate to our geometry?
    mu = arr("n_movement_units"); det = np.array([r["detour"] for r in rows])
    print(f"\n   #movement-units vs detour corr: {np.corrcoef([r['n_movement_units'] for r in rows], det)[0,1]:.2f}")
    jerky = sorted(rows, key=lambda r: -r["n_movement_units"])[:8]
    print("   jerkiest reps (most movement units = least smooth):")
    for r in jerky:
        print(f"     {r['name']:42s} MU={r['n_movement_units']:2d} detour={r['detour']:.2f} "
              f"lat={r['lat_mm']:.0f} mvt_t={r['movement_time']:.1f}s drink={r['drink']}")
    json.dump(rows, open(f"experiments/drink_study/cache/cup_murphy_{args.tag}.json", "w"), default=float, indent=1)
    print(f"\n   wrote cache/cup_murphy_{args.tag}.json")


if __name__ == "__main__":
    main()
