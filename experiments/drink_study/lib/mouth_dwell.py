"""Mouth-based drink-dwell TRUTH from the head+cup mocap (cache/qtm_c3d_cleaned).

The old dwell "truth" (segment_cup_only / geo_span on the mocap cup track) is a SPEED
PROXY: with no mouth marker it can only call the cup "nearly-still near its lift peak",
so it clips to the motionless core of the sip and misses the onset/offset where the cup
is already at the lips but still moving. That is exactly why the learned hybrid segmenter
"regressed" on the DISAGREE reps — it was bracketing a truth that was itself too narrow.

The cleaned C3Ds add a rigid 5-marker HEAD cluster (737/772 trials), so we can now define
drinking GEOMETRICALLY, the van Andel way: dwell = the cup is AT THE MOUTH. Concretely, the
cup-to-mouth distance drops from ~0.6 m at rest to <0.1 m at the apex; the dwell is the
contiguous interval where that distance stays within 15% of its steady-state range above
the minimum. No speed, no velocity gate — a true cup-at-mouth event.

    from mouth_dwell import dwell_truth
    dw = dwell_truth("P02_0012")          # -> span (mocap idx), distance signal, threshold
    dw.span_at(n_frames=377)              # resampled onto a 60Hz video-track of length 377

    python mouth_dwell.py P02_0012        # print + optional plot for one trial
    python mouth_dwell.py --all cache/mouth_dwell.json   # cache the truth for every trial
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _paths import QTM_C3D_HEAD
from qtm_c3d import load_trial

STEADY_PCTL = 90.0     # (legacy) "rest" cup-to-mouth distance = 90th pctl
DWELL_FRAC = 0.15      # van Andel: dwell = within 15% of steady-state range above the min
MIN_DWELL_S = 0.20     # ignore sub-200ms dips (not a sip)
REST_CROSS = 0.70      # reach edge = where cup-head dist returns to 70% of the way to max
SMOOTH_N = 7           # boxcar frames to smooth the cup-head distance


@dataclass
class Dwell:
    name: str
    rate: float                 # mocap Hz (100/120)
    dist: np.ndarray            # (T,) cup-to-mouth mm (NaN where head/cup missing)
    thr: float                  # dwell distance threshold (mm)
    span: tuple[int, int] | None  # (start, end) mocap-frame indices, end exclusive

    @property
    def dur_s(self) -> float:
        return (self.span[1] - self.span[0]) / self.rate if self.span else 0.0

    def mask(self, n: int | None = None) -> np.ndarray:
        """0/1 dwell mask at the mocap rate (or resampled to n frames if given)."""
        a = np.zeros(len(self.dist))
        if self.span:
            a[self.span[0]:self.span[1]] = 1.0
        if n is None or n == len(a):
            return a
        x0 = np.linspace(0, 1, len(a)); x1 = np.linspace(0, 1, n)
        return (np.interp(x1, x0, a) >= 0.5).astype(float)

    def span_at(self, n_frames: int) -> tuple[int, int] | None:
        """The dwell span mapped onto an n_frames-long (e.g. 60Hz video-track) index."""
        m = self.mask(n_frames)
        on = np.where(m > 0.5)[0]
        return (int(on[0]), int(on[-1] + 1)) if len(on) else None


def _longest_run(below: np.ndarray, min_len: int) -> tuple[int, int] | None:
    best = None
    for k, g in itertools.groupby(enumerate(below.astype(int)), key=lambda x: x[1]):
        if not k:
            continue
        gg = list(g); s, e = gg[0][0], gg[-1][0] + 1
        if e - s >= min_len and (best is None or (e - s) > (best[1] - best[0])):
            best = (s, e)
    return best


LEAVE_FRAC = 0.40      # a hump only ENDS the dwell if the cup retreats past this frac of range


def _bridged_span(dist, thr, leave_lvl, min_len):
    """Dwell span that BRIDGES a multi-apex drink. A drink can dip to the mouth, rise over a
    small HUMP (cup momentarily lifts a little between swallows / re-approaches), then dip
    again — one continuous drink. `_longest_run` wrongly chops that at the hump and keeps
    only one apex. Here: find the below-`thr` runs, then MERGE adjacent runs whose separating
    hump never rises above `leave_lvl` (the cup never actually left the mouth). The dwell is
    the first below-thr crossing to the last, across bridged humps. Only a retreat past
    leave_lvl (toward rest) is a real end. Returns (s,e) of the longest bridged group."""
    below = np.isfinite(dist) & (dist < thr)
    runs = []
    for k, g in itertools.groupby(enumerate(below.astype(int)), key=lambda x: x[1]):
        if k:
            gg = list(g); runs.append([gg[0][0], gg[-1][0] + 1])
    if not runs:
        return None
    # merge run i,i+1 if the cup stays below leave_lvl throughout the gap between them
    merged = [runs[0]]
    for s, e in runs[1:]:
        gap = dist[merged[-1][1]:s]
        stayed_near = gap.size == 0 or np.nanmax(gap) < leave_lvl
        if stayed_near:
            merged[-1][1] = e                               # bridge: extend across the hump
        else:
            merged.append([s, e])
    merged = [(s, e) for s, e in merged if e - s >= min_len]
    if not merged:
        return None
    return max(merged, key=lambda r: r[1] - r[0])


def _reach_rest(ds, apex, cross_frac=REST_CROSS):
    """(d_rest, l, r): cup-at-rest distance = avg of the cup->head distance where the reach
    STARTS (just before the cup rises) and ENDS (just after it settles), found by walking
    outward from the apex until the distance climbs back to `cross_frac` of the way to its
    max (i.e. the cup has returned near its resting distance from the head)."""
    dmax = np.nanmax(ds)
    cross = ds[apex] + cross_frac * (dmax - ds[apex])
    l = apex
    while l > 0 and ds[l] < cross:
        l -= 1
    r = apex
    while r < len(ds) - 1 and ds[r] < cross:
        r += 1
    return 0.5 * (ds[l] + ds[r]), l, r


def dwell_truth(name_or_trial, root: Path = QTM_C3D_HEAD,
                dwell_frac: float = DWELL_FRAC, min_dwell_s: float = MIN_DWELL_S,
                smooth_n: int = SMOOTH_N) -> Dwell:
    """Drink-dwell truth from the CUP -> HEAD-CENTROID distance (single head point, so it
    transfers to the future 1-landmark biomech head model — NO mouth proxy). The dwell is
    the interval where the cup is within `dwell_frac` (15%) of the way from its APEX (closest
    approach, the drinking floor) to its REST distance, where rest = the cup->head distance
    at the reach's start/end (see _reach_rest). Longest such run. `dist` in the returned
    Dwell is the (smoothed) cup->head distance."""
    tr = name_or_trial if hasattr(name_or_trial, "markers") else load_trial(name_or_trial, root=root)
    if not tr.has_head():
        return Dwell(tr.name, tr.rate, np.full(tr.n_frames, np.nan), float("nan"), None)
    from scipy.ndimage import uniform_filter1d
    d = tr.cup_to_head()
    if np.isfinite(d).sum() < 3:
        return Dwell(tr.name, tr.rate, d, float("nan"), None)
    ds = uniform_filter1d(np.nan_to_num(d, nan=np.nanmedian(d)), smooth_n)
    apex = int(np.nanargmin(ds)); d_apex = float(ds[apex])
    d_rest, _, _ = _reach_rest(ds, apex)
    thr = d_apex + dwell_frac * (d_rest - d_apex)           # 15% of apex->rest range
    span = _longest_run(np.isfinite(d) & (ds < thr), int(min_dwell_s * tr.rate))
    return Dwell(tr.name, tr.rate, ds, thr, span)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trial", nargs="?", help="C3D stem, e.g. P02_0012")
    ap.add_argument("--root", type=Path, default=QTM_C3D_HEAD)
    ap.add_argument("--all", metavar="OUT.json", help="cache dwell truth for every trial")
    ap.add_argument("--plot", metavar="OUT.png", help="save the distance+dwell plot")
    args = ap.parse_args()

    if args.all:
        import glob
        out = {}
        stems = sorted(Path(p).stem for p in glob.glob(str(Path(args.root) / "*.c3d")))
        print(f"computing mouth dwell for {len(stems)} trials ...", flush=True)
        nohead = 0
        for i, st in enumerate(stems):
            dw = dwell_truth(st, root=args.root)
            if dw.span is None and not np.isfinite(dw.thr):
                nohead += 1
            out[st] = dict(rate=dw.rate, thr=None if not np.isfinite(dw.thr) else round(dw.thr, 1),
                           span=list(dw.span) if dw.span else None, dur_s=round(dw.dur_s, 3))
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(stems)}  (no-head so far: {nohead})", flush=True)
        Path(args.all).write_text(json.dumps(out, indent=2))
        durs = [v["dur_s"] for v in out.values() if v["span"]]
        print(f"wrote {args.all}: {len(durs)} trials with a dwell, {nohead} without head markers; "
              f"dwell dur median {np.median(durs):.2f}s  IQR "
              f"[{np.percentile(durs,25):.2f},{np.percentile(durs,75):.2f}]s", flush=True)
        return

    if not args.trial:
        ap.error("give a trial stem or --all")
    dw = dwell_truth(args.trial, root=args.root)
    d = dw.dist; fin = np.isfinite(d)
    print(f"{dw.name}  rate={dw.rate:.0f}Hz  T={len(d)}")
    print(f"  cup-to-mouth: min={np.nanmin(d):.0f}  rest~={np.nanpercentile(d,90):.0f}  "
          f"max={np.nanmax(d):.0f} mm")
    print(f"  thr={dw.thr:.0f}mm  dwell span={dw.span}  dur={dw.dur_s:.2f}s")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = np.arange(len(d)) / dw.rate
        fig, ax = plt.subplots(figsize=(10, 3.2))
        ax.plot(t[fin], d[fin], lw=1.2, color="#2a6")
        ax.axhline(dw.thr, ls="--", color="#888", lw=1, label=f"thr {dw.thr:.0f}mm")
        if dw.span:
            ax.axvspan(dw.span[0] / dw.rate, dw.span[1] / dw.rate, color="#f80", alpha=0.25,
                       label=f"dwell {dw.dur_s:.2f}s")
        ax.set_xlabel("s"); ax.set_ylabel("cup→mouth mm"); ax.set_title(dw.name); ax.legend()
        fig.tight_layout(); fig.savefig(args.plot, dpi=130)
        print(f"  wrote {args.plot}")


if __name__ == "__main__":
    main()
