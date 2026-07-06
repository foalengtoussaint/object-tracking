"""Plot all four track variants vs mocap-truth, straight from the lopo_fused cache.

NO GPU / NO retraining -- reloads cache/lopo_fused/<video>.npz (kf/hard/hard_kf/fused/
true tracks + valid + phases). Shows concretely why the LOPO numbers come out as they do:
  - blow-up reps: hard (raw anchor) diverges; hard_kf (KF anchor) & fused stay sane
  - easy reps   : hard_kf is slightly blunter than hard

    python experiments/drink_study/viz_variants.py --reps SUBSTR [SUBSTR ...]
Run from repo root. Writes cache/variants.png
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import sys, argparse, glob
from pathlib import Path
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "drink_study"))
import segment_cup_only as S

from _paths import CACHE as _C
CACHE = _C / "lopo_fused"
HZ = 60.0
COLS = {"true": "#111", "kf": "#1f77b4", "hard": "#d62728",
        "hard_kf": "#2ca02c", "fused": "#9467bd"}
LABELS = {"true": "TRUE (mocap)", "kf": "plain KF+RTS", "hard": "hard (raw anchor)",
          "hard_kf": "hard_kf (KF anchor)", "fused": "fused KF"}


def gaps_of(valid):
    out = []; i = 0; n = len(valid)
    while i < n:
        if not valid[i]:
            j = i
            while j < n and not valid[j]:
                j += 1
            out.append((i, j - 1)); i = j
        else:
            i += 1
    return out


def load(sub):
    hits = [f for f in glob.glob(str(CACHE / "*.npz")) if sub in Path(f).stem]
    return np.load(hits[0], allow_pickle=True) if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="*", default=[
        "P14_P14_drinking_left_20240221_092404",     # hard blows up, hard_kf/fused fine
        "P14_P14_drinking_right_20240221_092101",     # hard blows up
        "P13_P13_drinking_left_20240216_161925"])     # easy: hard_kf blunter
    args = ap.parse_args()
    ds = [(s, load(s)) for s in args.reps]
    ds = [(s, d) for s, d in ds if d is not None]
    n = len(ds)
    fig, axes = plt.subplots(1, n, figsize=(6.4 * n, 5.2), squeeze=False)
    for col, (sub, d) in enumerate(ds):
        true = d["true"]; valid = d["valid"]; rest = true[:5].mean(0)
        # reach axis = principal direction of the true track from rest
        Cn = true - rest; u = np.linalg.svd(Cn, full_matrices=False)[2][0]
        if (Cn @ u).max() < -(Cn @ u).min():
            u = -u
        reach = lambda xyz: (xyz - rest) @ u
        runs = S.segment_cup_only(true, fps=HZ)["drink_runs"]
        d0, d1 = (runs[0][0], runs[-1][1]) if runs else (0, len(true))
        a = max(0, d0 - 15); b = min(len(true), d1 + 15)
        t = (np.arange(a, b) - d0) / HZ * 1000
        ax = axes[0][col]
        for s, e in gaps_of(valid):
            if e >= a and s <= b:
                ax.axvspan((max(s, a) - d0) / HZ * 1000, (min(e, b) - d0) / HZ * 1000,
                           color="#999", alpha=0.13, lw=0)
        for key in ("true", "kf", "hard", "hard_kf", "fused"):
            lw = 2.6 if key == "true" else 1.7
            ax.plot(t, reach(d[key])[a:b], color=COLS[key], lw=lw, label=LABELS[key])
        # detections (consensus) as dots
        cons = d["cons"]; cv = np.isfinite(cons).all(1)
        dr = np.full(len(cons), np.nan); dr[cv] = (cons[cv] - rest) @ u
        ax.scatter(t, dr[a:b], s=14, c="#8b008b", zorder=6, marker="o", label="detections")
        errs = {k: float(d[f"drinkerr_{k}"]) for k in ("kf", "hard", "hard_kf", "fused")}
        ax.set_title(f"{sub[:26]}\ndrink err: kf {errs['kf']:.0f}  hard {errs['hard']:.0f}  "
                     f"hard_kf {errs['hard_kf']:.0f}  fused {errs['fused']:.0f} mm", fontsize=9)
        ax.set_xlabel("ms rel. drink start  (grey = occlusion gap)")
        if col == 0:
            ax.set_ylabel("reach-axis position (mm)")
        ax.grid(alpha=0.25)
        if col == 0:
            ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Why the LOPO numbers: HARD (red, raw anchor) diverges on bad-detection gaps; "
                 "HARD_KF (green, KF anchor) & FUSED (purple) stay anchored to the denoised state",
                 fontsize=11, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = _C / "variants.png"
    fig.savefig(out, dpi=118)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
