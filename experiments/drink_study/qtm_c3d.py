"""Loader + inventory for the Qualisys cup-mocap C3D trials.

The drink study's optical mocap is *cup ground truth*: every one of the 772
trials carries the same 4 cup markers (`cupdl, cupdr, cupul, cupur`) at 100 Hz,
fully gap-filled, in millimetres, in the QTM lab frame. Four rigid markers on
the cup give the cup's full 6-DoF pose (centroid + orientation) — the exact
quantity the video pipeline (YOLO -> triangulate -> consensus-KF) estimates.

This module does NOT solve the QTM-lab -> Charuco-W0 alignment or the temporal
sync; it just loads the raw mocap and inventories what's there so the alignment
step has a clean foundation.

Usage:
    python qtm_c3d.py                 # print inventory (participants, counts, sides)
    python qtm_c3d.py --inventory-json cache/qtm_inventory.json

    from qtm_c3d import load_trial, list_trials
    tr = load_trial("P02_0021")       # CupTrial: markers (T,4,3) mm, pose, etc.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    from _paths import QTM_C3D
except ImportError:  # allow `python experiments/drink_study/qtm_c3d.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _paths import QTM_C3D

CUP_MARKERS = ("cupdl", "cupdr", "cupul", "cupur")  # down/up x left/right


@dataclass
class CupTrial:
    """One Qualisys cup-mocap trial."""
    name: str                       # stem, e.g. "P02_0021"
    path: Path
    rate: float                     # Hz (100)
    labels: list[str]               # marker labels in column order
    markers: np.ndarray             # (T, M, 3) mm, NaN where missing
    participant: str | None         # "P02" etc., parsed from name
    side: str | None                # "left" | "right" | None (DTL/DTR or by folder)

    @property
    def n_frames(self) -> int:
        return self.markers.shape[0]

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.rate

    @property
    def missing_frac(self) -> float:
        return float(np.isnan(self.markers[..., 0]).mean())

    def centroid(self, despike: bool = True, max_step_mm: float = 50.0) -> np.ndarray:
        """(T,3) mm — mean of the 4 cup markers (cup position).

        QTM sometimes loses the cup and latches onto a reflection/stray marker for a
        stretch of frames — e.g. P10_0029 teleports ~590 mm at frame 577, sits frozen
        at a wrong location (z below the table) for ~43 frames, then jumps back. These
        EXCURSION BLOCKS are bad ground truth that wreck the speed-curve sync and the
        position error. With despike=True they are detected as runs bracketed by
        >max_step_mm jumps and NaN'd (marked missing) — NOT interpolated, since we
        don't want to fabricate 0.4 s of motion. True drink motion is <~10 mm/frame so
        a 50 mm/frame jump is unambiguously non-physical. Returns NaN over excised
        runs; callers already handle NaN (sync/Kabsch mask them out)."""
        c = np.nanmean(self.markers, axis=1)
        if not despike or len(c) < 3:
            return c
        step = np.linalg.norm(np.diff(c, axis=0), axis=1)        # (T-1,)
        jumps = np.where(step > max_step_mm)[0]                   # indices i: c[i]->c[i+1] is a jump
        if len(jumps) >= 2:
            # excise frames between an out-jump and the next in-jump (the excursion)
            bad = np.zeros(len(c), bool)
            for a, b in zip(jumps[:-1], jumps[1:]):
                # frames a+1 .. b are off the main trajectory; excise if the run is
                # short enough to be a glitch (< ~1 s) not a real phase
                if (b - a) <= int(self.rate):
                    bad[a + 1: b + 1] = True
            c[bad] = np.nan
        return c

    def gt_quality(self, max_step_mm: float = 50.0, min_lift_mm: float = 120.0) -> dict:
        """Judge whether this take is usable optical ground truth, from the MOCAP
        ALONE (no video, no sync). Two failure modes seen in the QTM export:
          * non-physical step: a single-frame centroid jump > max_step_mm that the
            despike couldn't repair (a one-way teleport / hard gap-fill step, e.g.
            P060026 steps ~470 mm; a real drink moves <~10 mm/frame). The trajectory
            is broken even though positions look plausible afterwards.
          * dead/static take: cup never lifts (< min_lift_mm) — not a real drink.
        Returns {'ok', 'reason', 'max_step_mm', 'lift_mm'} so callers can reject
        defective reps BEFORE attempting to sync/score them."""
        c = self.centroid(despike=True)          # excise repairable excursions first
        finite = c[np.isfinite(c).all(1)]
        if len(finite) < 3:
            return dict(ok=False, reason="too few valid frames", max_step_mm=None, lift_mm=0.0)
        max_step = float(np.linalg.norm(np.diff(finite, axis=0), axis=1).max())
        lift = float(np.nanmax(c[:, 2]) - np.nanmin(c[:, 2]))
        if max_step > max_step_mm:
            return dict(ok=False, reason=f"non-physical step {max_step:.0f}mm/frame",
                        max_step_mm=round(max_step, 1), lift_mm=round(lift, 1))
        if lift < min_lift_mm:
            return dict(ok=False, reason=f"dead take (lift {lift:.0f}mm)",
                        max_step_mm=round(max_step, 1), lift_mm=round(lift, 1))
        return dict(ok=True, reason="ok", max_step_mm=round(max_step, 1), lift_mm=round(lift, 1))

    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Rigid cup pose per frame from the 4 markers.

        Returns (R, t): R is (T,3,3) rotation of a cup-local frame into the lab
        frame, t is (T,3) mm centroid. The local frame is built from the marker
        layout: x = right (left->right markers), y = up (down->up), z = x cross y.
        Frames with any missing marker yield NaN.
        """
        idx = {l: i for i, l in enumerate(self.labels)}
        try:
            dl, dr, ul, ur = (idx[m] for m in CUP_MARKERS)
        except KeyError:  # non-standard label set; fall back to centroid only
            t = self.centroid()
            R = np.full((self.n_frames, 3, 3), np.nan)
            return R, t
        m = self.markers
        right = (m[:, dr] + m[:, ur]) - (m[:, dl] + m[:, ul])   # left -> right
        up = (m[:, ul] + m[:, ur]) - (m[:, dl] + m[:, dr])      # down -> up
        x = _normalize(right)
        z = _normalize(np.cross(x, up))
        y = np.cross(z, x)
        R = np.stack([x, y, z], axis=-1)                        # columns are axes
        return R, self.centroid()


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, np.nan, n)


def _parse_name(stem: str) -> tuple[str | None, str | None]:
    """(participant, side) from a C3D stem.

    DTL#### / DTR#### -> drinking-task left/right (no participant id in name).
    P02_0021, P030011 -> participant P02 / P03; side unknown from name alone.
    """
    if stem.upper().startswith("DTL"):
        return None, "left"
    if stem.upper().startswith("DTR"):
        return None, "right"
    m = re.match(r"(P\d{2})", stem)
    return (m.group(1) if m else None), None


def list_trials(root: Path = QTM_C3D) -> list[Path]:
    return sorted(Path(root).glob("*.c3d"))


def load_trial(name_or_path, root: Path = QTM_C3D) -> CupTrial:
    """Load one C3D. Accepts a stem ("P02_0021"), filename, or full path."""
    import ezc3d

    p = Path(name_or_path)
    if not p.is_absolute() and p.suffix.lower() != ".c3d":
        p = Path(root) / f"{name_or_path}.c3d"
    elif not p.is_absolute():
        p = Path(root) / p.name if not p.exists() else p

    c = ezc3d.c3d(str(p))
    pts = c.data["points"]                                  # 4 x M x T (x,y,z,residual)
    labels = list(c.parameters["POINT"]["LABELS"]["value"])
    rate = float(c.parameters["POINT"]["RATE"]["value"][0])
    markers = np.transpose(pts[:3], (2, 1, 0)).astype(np.float64)  # (T, M, 3)
    part, side = _parse_name(p.stem)
    return CupTrial(p.stem, p, rate, labels, markers, part, side)


def inventory(root: Path = QTM_C3D) -> dict:
    """Summarize the whole C3D set without loading marker arrays into memory twice."""
    import ezc3d

    trials = list_trials(root)
    rows = []
    by_part: dict[str, int] = {}
    label_sigs: dict[tuple, int] = {}
    for p in trials:
        c = ezc3d.c3d(str(p))
        pts = c.data["points"]
        labels = tuple(c.parameters["POINT"]["LABELS"]["value"])
        rate = float(c.parameters["POINT"]["RATE"]["value"][0])
        nframes = pts.shape[2]
        miss = float(np.isnan(pts[0]).mean())
        part, side = _parse_name(p.stem)
        key = part or ("DT" + (side or "?")[0].upper())
        by_part[key] = by_part.get(key, 0) + 1
        label_sigs[labels] = label_sigs.get(labels, 0) + 1
        rows.append(dict(name=p.stem, participant=part, side=side,
                         rate=rate, n_frames=nframes, duration_s=round(nframes / rate, 2),
                         missing_frac=round(miss, 4), n_markers=pts.shape[1]))
    return dict(
        root=str(root),
        n_trials=len(trials),
        label_signatures={", ".join(k): v for k, v in label_sigs.items()},
        by_participant=dict(sorted(by_part.items())),
        trials=rows,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=QTM_C3D)
    ap.add_argument("--inventory-json", type=Path, default=None,
                    help="also write the full per-trial inventory here")
    args = ap.parse_args()

    inv = inventory(args.root)
    print(f"QTM cup mocap: {inv['n_trials']} trials in {inv['root']}")
    print(f"label signatures: {inv['label_signatures']}")
    print("by participant/set:")
    for k, n in inv["by_participant"].items():
        print(f"  {k:6s} {n}")
    # quick health line
    durs = [r["duration_s"] for r in inv["trials"]]
    miss = [r["missing_frac"] for r in inv["trials"]]
    rates = sorted({r["rate"] for r in inv["trials"]})
    print(f"rate(s): {rates} Hz   duration {min(durs):.1f}-{max(durs):.1f}s   "
          f"max missing_frac {max(miss):.4f}")
    if args.inventory_json:
        args.inventory_json.write_text(json.dumps(inv, indent=2))
        print(f"wrote {args.inventory_json}")


if __name__ == "__main__":
    main()
