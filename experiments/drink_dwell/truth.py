"""STAGE: the drink-dwell TRUTH, from the mocap cup→head distance alone.

van Andel definition, made of visible steps you can check one at a time:
  1. smooth the cup→head distance (7-frame boxcar)
  2. apex   = closest approach (the drinking floor)
  3. rest   = cup→head distance at the reach's start/end (walk out from apex to 70% of max)
  4. thr    = apex + 15% of (rest − apex)
  5. dwell  = the longest contiguous run below thr (≥200ms)

No mouth proxy, no bridging: one head point → transfers to the future 1-landmark biomech
model. `dwell_truth(trial)` returns everything so each step is inspectable.
"""
from __future__ import annotations
from dataclasses import dataclass
import itertools
import numpy as np

DWELL_FRAC = 0.15      # dwell = within 15% of the apex→rest range
MIN_DWELL_S = 0.20     # ignore sub-200ms dips
REST_CROSS = 0.70      # reach edge = cup→head returns to 70% of the way to max
SMOOTH_N = 7           # boxcar frames


@dataclass
class Dwell:
    name: str
    rate: float
    dist: np.ndarray                  # (T,) smoothed cup→head mm
    thr: float
    span: tuple[int, int] | None      # (start, end) mocap-frame idx, end exclusive

    @property
    def dur_s(self) -> float:
        return (self.span[1] - self.span[0]) / self.rate if self.span else 0.0

    def span_at(self, n_frames: int) -> tuple[int, int] | None:
        """The dwell mapped onto an n_frames-long (e.g. 60Hz track) index."""
        a = np.zeros(len(self.dist))
        if self.span:
            a[self.span[0]:self.span[1]] = 1.0
        x0 = np.linspace(0, 1, len(a)); x1 = np.linspace(0, 1, n_frames)
        m = np.interp(x1, x0, a) >= 0.5
        on = np.where(m)[0]
        return (int(on[0]), int(on[-1] + 1)) if len(on) else None


def _reach_rest(ds, apex, cross_frac=REST_CROSS):
    """Rest cup→head distance = avg of where the reach starts / ends, found by walking out
    from the apex until the distance climbs back to `cross_frac` of the way to its max."""
    dmax = np.nanmax(ds)
    cross = ds[apex] + cross_frac * (dmax - ds[apex])
    l = apex
    while l > 0 and ds[l] < cross:
        l -= 1
    r = apex
    while r < len(ds) - 1 and ds[r] < cross:
        r += 1
    return 0.5 * (ds[l] + ds[r])


def _longest_run(below: np.ndarray, min_len: int):
    best = None
    for k, g in itertools.groupby(enumerate(below.astype(int)), key=lambda x: x[1]):
        if not k:
            continue
        gg = list(g); s, e = gg[0][0], gg[-1][0] + 1
        if e - s >= min_len and (best is None or (e - s) > (best[1] - best[0])):
            best = (s, e)
    return best


def dwell_truth(trial, dwell_frac=DWELL_FRAC, min_dwell_s=MIN_DWELL_S, smooth_n=SMOOTH_N) -> Dwell:
    """Drink-dwell from the cup→head distance (see module docstring for the 5 steps)."""
    from scipy.ndimage import uniform_filter1d
    if not trial.has_head():
        return Dwell(trial.name, trial.rate, np.full(trial.n_frames, np.nan), float("nan"), None)
    d = trial.cup_to_head()
    if np.isfinite(d).sum() < 3:
        return Dwell(trial.name, trial.rate, d, float("nan"), None)
    ds = uniform_filter1d(np.nan_to_num(d, nan=np.nanmedian(d)), smooth_n)   # 1. smooth
    apex = int(np.nanargmin(ds))                                             # 2. apex
    d_rest = _reach_rest(ds, apex)                                           # 3. rest
    thr = ds[apex] + dwell_frac * (d_rest - ds[apex])                        # 4. threshold
    span = _longest_run(np.isfinite(d) & (ds < thr), int(min_dwell_s * trial.rate))  # 5. run
    return Dwell(trial.name, trial.rate, ds, float(thr), span)


MAX_HEAD_MISS_WINDOW = 0.20   # >20% head missing across the CUP-MOTION window = untrustworthy
MAX_HEAD_RESID_STEP = 30.0    # a head teleport the despike couldn't excise (mm/frame)


def head_quality(trial, dw=None) -> dict:
    """Is the TRUTH trustworthy, i.e. is the DESPIKED head good enough where the drink happens?
    Despiking (mocap.head_centroid) fixes most head jumps/gaps; this flags the residual reps it
    CAN'T save so run.py EXCLUDES them from the headline instead of scoring against a head-
    corrupted dwell. Head quality is judged over the CUP-MOTION window (not the dwell — that's
    circular, since the dwell already drops head-missing frames), which catches drinks where the
    head dropped out and PREVENTED / truncated the dwell. Returns {ok, reason, head_miss, resid_step}."""
    h = trial.head_centroid(despike=True)
    miss = trial.head_missing_frac()                          # over cup-motion window
    resid = float(np.nanmax(np.linalg.norm(np.diff(h, axis=0), axis=1))) if len(h) > 1 else 0.0
    if not np.isfinite(miss) or miss > MAX_HEAD_MISS_WINDOW:
        return dict(ok=False, reason=f"head missing {miss*100:.0f}% across the lift",
                    head_miss=miss, resid_step=resid)
    if resid > MAX_HEAD_RESID_STEP:
        return dict(ok=False, reason=f"residual head jump {resid:.0f}mm/fr (unsaveable)",
                    head_miss=miss, resid_step=resid)
    return dict(ok=True, reason="ok", head_miss=miss, resid_step=resid)


if __name__ == "__main__":   # smoke: print the dwell for one trial
    import sys
    from mocap import load_trial
    tr = load_trial(sys.argv[1] if len(sys.argv) > 1 else "P02_0012")
    dw = dwell_truth(tr)
    print("  head_quality:", head_quality(tr, dw))
    print(f"{dw.name}  thr={dw.thr:.0f}mm  span={dw.span}  dur={dw.dur_s:.2f}s")
