"""STAGE INPUT: the mocap cup + head (Qualisys C3D), and the geometric primitives that
turn them into the ONE signal this whole experiment is about — the cup→head distance.

This is a self-contained slice of drink_study's qtm_c3d + qtm_align, keeping ONLY what the
dwell pipeline needs (cup centroid, head centroid, the cup→head distance, ground-truth
quality, and Kabsch/sync helpers). Everything is a small named function you can call on one
trial and inspect — that is the whole point of this rewrite.

    from mocap import load_trial
    tr = load_trial("P02_0012")          # cup + head markers
    d  = tr.cup_to_head()                # (T,) mm — THE signal, in one obvious place
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# --- where the cleaned C3Ds live (cup + rigid head cluster) -------------------------------
# Resolved from drink_study's cache so we share the DATA (not the code). Override with the
# env var if the tree moves.
import os
_DS = Path(__file__).resolve().parents[1] / "drink_study"
QTM_C3D_HEAD = Path(os.environ.get("OT_QTM_C3D_HEAD", str(_DS / "cache" / "qtm_c3d_cleaned")))

CUP_MARKERS = ("cupdl", "cupdr", "cupul", "cupur")
HEAD_MARKERS = ("FHD", "L_FHd", "R_FHd", "L_BHd", "R_BHd")
FRONT_HEAD = ("L_FHd", "R_FHd")
BACK_HEAD = ("L_BHd", "R_BHd")


@dataclass
class CupTrial:
    name: str
    rate: float                     # mocap Hz (100/120)
    labels: list[str]
    markers: np.ndarray             # (T, M, 3) mm, NaN where missing

    @property
    def n_frames(self) -> int:
        return self.markers.shape[0]

    # -- STAGE: cup position ---------------------------------------------------------------
    def centroid(self, despike: bool = True, max_step_mm: float = 50.0) -> np.ndarray:
        """(T,3) mm — mean of the 4 CUP markers (only the cup, never the head — averaging
        the head in would pull it ~350mm off and corrupt everything downstream). Despike
        NaNs short non-physical excursions (>50mm/frame runs) so sync/Kabsch aren't wrecked
        by a QTM reflection latch; NOT interpolated (we don't fabricate motion)."""
        idx = {l: i for i, l in enumerate(self.labels)}
        cols = [idx[m] for m in CUP_MARKERS if m in idx]
        cup = self.markers[:, cols] if cols else self.markers
        c = np.nanmean(cup, axis=1)
        if not despike or len(c) < 3:
            return c
        step = np.linalg.norm(np.diff(c, axis=0), axis=1)
        jumps = np.where(step > max_step_mm)[0]
        if len(jumps) >= 2:
            bad = np.zeros(len(c), bool)
            for a, b in zip(jumps[:-1], jumps[1:]):
                if (b - a) <= int(self.rate):
                    bad[a + 1: b + 1] = True
            c[bad] = np.nan
        return c

    # -- STAGE: head position (single stable point = future biomech landmark) --------------
    def has_head(self) -> bool:
        return all(l in self.labels for l in FRONT_HEAD + BACK_HEAD)

    def head_centroid(self) -> np.ndarray:
        """(T,3) mm mean of the head markers — one stable head point (what the future
        1-landmark biomech model will provide). The dwell keys off cup→this, no mouth proxy."""
        idx = {l: i for i, l in enumerate(self.labels)}
        cols = [idx[m] for m in HEAD_MARKERS if m in idx]
        return np.nanmean(self.markers[:, cols], axis=1) if cols else np.full((self.n_frames, 3), np.nan)

    # -- STAGE: THE SIGNAL -----------------------------------------------------------------
    def cup_to_head(self) -> np.ndarray:
        """(T,) mm cup-centroid → head-centroid distance. This is the drink signal; the
        truth dwell and the head feature both key off it. Kept as ONE obvious function so a
        'spike' can be inspected here directly instead of hidden inside a resample/Kabsch."""
        return np.linalg.norm(self.centroid() - self.head_centroid(), axis=1)

    # -- STAGE: is this trial usable ground truth? -----------------------------------------
    def gt_quality(self, max_step_mm: float = 50.0, min_lift_mm: float = 120.0) -> dict:
        """Reject defective takes from the MOCAP ALONE: a non-physical centroid step the
        despike couldn't fix, or a dead take where the cup never lifts (<120mm)."""
        c = self.centroid(despike=True)
        finite = c[np.isfinite(c).all(1)]
        if len(finite) < 3:
            return dict(ok=False, reason="too few valid frames")
        max_step = float(np.linalg.norm(np.diff(finite, axis=0), axis=1).max())
        lift = float(np.nanmax(c[:, 2]) - np.nanmin(c[:, 2]))
        if max_step > max_step_mm:
            return dict(ok=False, reason=f"non-physical step {max_step:.0f}mm/frame")
        if lift < min_lift_mm:
            return dict(ok=False, reason=f"dead take (lift {lift:.0f}mm)")
        return dict(ok=True, reason="ok")


def load_trial(name_or_path, root: Path = QTM_C3D_HEAD) -> CupTrial:
    """Load one C3D by stem ("P02_0012"), filename, or path."""
    import ezc3d
    p = Path(name_or_path)
    if not p.is_absolute() and p.suffix.lower() != ".c3d":
        p = Path(root) / f"{name_or_path}.c3d"
    elif not p.is_absolute():
        p = Path(root) / p.name if not p.exists() else p
    c = ezc3d.c3d(str(p))
    pts = c.data["points"]                                  # 4 x M x T
    labels = list(c.parameters["POINT"]["LABELS"]["value"])
    rate = float(c.parameters["POINT"]["RATE"]["value"][0])
    markers = np.transpose(pts[:3], (2, 1, 0)).astype(np.float64)   # (T, M, 3)
    return CupTrial(p.stem, rate, labels, markers)


# =========================================================================================
# Sync + alignment primitives (mocap lab frame <-> video track W0). Small, pure, testable.
# =========================================================================================
COMMON_HZ = 60.0
VIDEO_FPS = 60.0
MIN_SYNC_CORR = 0.7


def speed(xyz: np.ndarray, rate: float) -> np.ndarray:
    """(T,) speed of a 3D track (small gaps interpolated so the gradient is clean)."""
    x = xyz.copy()
    for k in range(3):
        col = x[:, k]; idx = np.arange(len(col)); g = ~np.isnan(col)
        if g.sum() > 2:
            x[:, k] = np.interp(idx, idx[g], col[g])
    return np.linalg.norm(np.gradient(x, 1.0 / rate, axis=0), axis=1)


def resample(xyz: np.ndarray, rate: float, to_hz: float = COMMON_HZ) -> np.ndarray:
    """Resample a (T,3) track from `rate` Hz to `to_hz` Hz."""
    n = xyz.shape[0]
    t0 = np.arange(n) / rate
    t1 = np.arange(0, t0[-1], 1.0 / to_hz)
    out = np.empty((len(t1), 3))
    for k in range(3):
        col = xyz[:, k]; g = ~np.isnan(col)
        out[:, k] = np.interp(t1, t0[g], col[g]) if g.sum() > 2 else np.nan
    return out


def xcorr_lag(a: np.ndarray, b: np.ndarray):
    """Lag (samples, +ve => b later) maximizing normalized speed-curve correlation, and the
    peak correlation value in [0,1] (sync confidence)."""
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    c = np.correlate(a, b, mode="full") / min(len(a), len(b))
    k = int(np.argmax(c))
    return k - (len(b) - 1), float(c[k])


def _kabsch_weighted(P, Q, w):
    w = w / (w.sum() + 1e-12)
    Pc = (w[:, None] * P).sum(0); Qc = (w[:, None] * Q).sum(0)
    H = (w[:, None] * (P - Pc)).T @ (Q - Qc)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, Qc - R @ Pc


def kabsch(P: np.ndarray, Q: np.ndarray, robust: bool = True, iters: int = 10):
    """Rotation+translation mapping P -> Q (no scale). Returns (R, t, rms_mm). Robust=IRLS
    with Huber weights so the fit locks onto the agreeing (rest/transport) frames and isn't
    dragged by the occluded apex where the video track is biased."""
    R, t = _kabsch_weighted(P, Q, np.ones(len(P)))
    if robust:
        for _ in range(iters):
            r = np.linalg.norm(Q - (P @ R.T + t), axis=1)
            c = 1.345 * (np.median(r) + 1e-6)
            w = np.where(r <= c, 1.0, c / (r + 1e-9))
            R, t = _kabsch_weighted(P, Q, w)
        r = np.linalg.norm(Q - (P @ R.T + t), axis=1)
        inl = r <= 1.345 * (np.median(r) + 1e-6)
        rms = float(np.sqrt(np.mean(r[inl] ** 2))) if inl.any() else float(np.sqrt(np.mean(r ** 2)))
    else:
        rms = float(np.sqrt(np.mean(np.sum((Q - (P @ R.T + t)) ** 2, axis=1))))
    return R, t, rms


if __name__ == "__main__":   # smoke: inspect the signal on one trial
    import sys
    tr = load_trial(sys.argv[1] if len(sys.argv) > 1 else "P02_0012")
    d = tr.cup_to_head()
    print(f"{tr.name}  rate={tr.rate:.0f}Hz  T={tr.n_frames}  has_head={tr.has_head()}")
    print(f"  cup->head mm: min={np.nanmin(d):.0f}  max={np.nanmax(d):.0f}")
    print(f"  gt_quality: {tr.gt_quality()}")
