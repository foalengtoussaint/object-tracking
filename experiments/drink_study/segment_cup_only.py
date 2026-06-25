"""Cup-ONLY drink-task phase segmentation (no pose).

Question being tested: can we recover the drink-task phases from the cup 3D
trajectory alone (our cache/track3d/*.json RTS track), with no hand/head pose?

What cup-only CAN'T give:
  * "reaching" — that is the hand moving toward a still cup, invisible to the
    cup. So it collapses into rest_pre. We output 5 phases:
        rest_pre -> forward_transport -> drinking -> back_transport -> rest_post

Everything is computed from rotation-invariant signals (cup speed, and cup
displacement from its rest position) because the calibration world frame is the
per-session Charuco board frame — there is no guaranteed vertical axis.

Algorithm:
  1. rest_pos = median cup xyz over the first 0.5 s (assumed rest_pre).
  2. speed (mm/s, median-smoothed) and disp = |cup - rest_pos|.
  3. Cup-motion window via hysteresis: onset > ONSET sustained CUP_MIN_RUN;
     offset trails down to OFFSET (so slow place-down isn't cut early).
  4. drinking = inside the window, cup near its peak displacement
     (disp > peak - DRINK_DISP_PAD) AND slow (speed < DRINK_SPEED), sustained
     >= MIN_PHASE frames. This is the dwell at the mouth.
  5. forward_transport = window before first drink; back_transport = after last
     drink. If no drink dwell, split the window at peak displacement.

Usage:
    python experiments/drink_study/segment_cup_only.py            # plot a sample
    python experiments/drink_study/segment_cup_only.py --plot P01_...stem
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FPS = 60.0
TRACKDIR = Path("experiments/drink_study/cache/track3d")

# thresholds (mm/s, mm, frames) — tuned on the real 60fps drink trials
ONSET            = 150.0   # cup-motion onset (above triangulation noise floor)
OFFSET           = 80.0    # cup-motion offset (trailing place-down)
CUP_MIN_RUN      = max(int(0.15 * FPS), 3)
DRINK_SPEED      = 120.0   # cup nearly still at the mouth
DRINK_DISP_PAD   = 90.0    # "near peak" = within this of max displacement
MIN_PHASE        = max(int(0.20 * FPS), 5)

PHASE_NAMES = ["rest_pre", "forward_transport", "drinking", "back_transport", "rest_post"]
P_REST_PRE, P_FWD, P_DRINK, P_BACK, P_REST_POST = range(5)
PHASE_COLORS = {
    "rest_pre": "#cccccc", "forward_transport": "#4c9be8", "drinking": "#e8554c",
    "back_transport": "#f0a23b", "rest_post": "#999999",
}


def _median_smooth(x: np.ndarray, w: int = 7) -> np.ndarray:
    half = w // 2
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.median(x[max(0, i - half):min(len(x), i + half + 1)])
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, i, T = [], 0, len(mask)
    while i < T:
        if mask[i]:
            j = i + 1
            while j < T and mask[j]:
                j += 1
            out.append((i, j)); i = j
        else:
            i += 1
    return out


def _intervals(phase: np.ndarray) -> list[tuple[str, int, int]]:
    out, i, T = [], 0, len(phase)
    while i < T:
        j = i + 1
        while j < T and phase[j] == phase[i]:
            j += 1
        out.append((PHASE_NAMES[int(phase[i])], i, j)); i = j
    return out


def load_track(stem_or_path) -> tuple[str, np.ndarray]:
    p = Path(stem_or_path)
    if not p.exists():
        p = TRACKDIR / (str(stem_or_path) if str(stem_or_path).endswith(".json")
                        else f"{stem_or_path}.json")
    d = json.loads(p.read_text())
    fr = d["frames"]
    # prefer RTS (smoothed, full coverage); fall back to kf then consensus
    xyz = np.full((len(fr), 3), np.nan)
    for i, f in enumerate(fr):
        v = f["rts"] or f["kf"] or f["consensus"]
        if v is not None:
            xyz[i] = v
    return p.stem, xyz


def segment_cup_only(xyz: np.ndarray, fps: float = FPS) -> dict:
    T = len(xyz)
    phase = np.full(T, P_REST_PRE, dtype=np.int8)
    valid = np.isfinite(xyz).all(1)
    if valid.sum() < 10:
        return {"phase": phase, "intervals": _intervals(phase), "speed": np.zeros(T),
                "disp": np.zeros(T), "grasp": (0, 0), "drink_runs": []}

    # fill gaps by interpolation so speed/disp are continuous
    idx = np.flatnonzero(valid)
    filled = xyz.copy()
    for ax in range(3):
        filled[:, ax] = np.interp(np.arange(T), idx, xyz[idx, ax])

    # speed
    step = np.r_[0.0, np.linalg.norm(np.diff(filled, axis=0), axis=1)] * fps
    speed = _median_smooth(step.astype(float), 7)

    # rest position from first 0.5 s, displacement from rest
    rw = max(int(0.5 * fps), 10)
    rest_pos = np.median(filled[:min(rw, T)], axis=0)
    disp = np.linalg.norm(filled - rest_pos, axis=1)

    # cup-motion window: hysteresis
    onset_runs = [(s, e) for s, e in _runs(speed > ONSET) if e - s >= CUP_MIN_RUN]
    if not onset_runs:
        return {"phase": phase, "intervals": _intervals(phase), "speed": speed,
                "disp": disp, "grasp": (0, 0), "drink_runs": []}
    grasp_start = onset_runs[0][0]
    grasp_end = onset_runs[-1][1]
    loose = np.flatnonzero(speed > OFFSET)
    tail = loose[loose >= grasp_end - 1]
    if tail.size:
        grasp_end = int(tail[-1]) + 1

    in_win = np.zeros(T, dtype=bool)
    in_win[grasp_start:grasp_end] = True

    # drinking dwell: near peak displacement + slow, inside the window
    peak_disp = disp[in_win].max() if in_win.any() else 0.0
    near_peak = disp > (peak_disp - DRINK_DISP_PAD)
    drink_mask = in_win & near_peak & (speed < DRINK_SPEED)
    drink_runs = [(s, e) for s, e in _runs(drink_mask) if e - s >= MIN_PHASE]

    # compose
    phase[grasp_start:grasp_end] = P_FWD          # placeholder = all transport
    phase[grasp_end:] = P_REST_POST
    if drink_runs:
        d0, d1 = drink_runs[0][0], drink_runs[-1][1]
        phase[grasp_start:d0] = P_FWD
        phase[d0:d1] = P_DRINK
        phase[d1:grasp_end] = P_BACK
    else:
        # no dwell -> split window at peak displacement frame
        apex = grasp_start + int(np.argmax(disp[grasp_start:grasp_end])) if grasp_end > grasp_start else grasp_start
        phase[grasp_start:apex] = P_FWD
        phase[apex:grasp_end] = P_BACK

    return {"phase": phase, "intervals": _intervals(phase), "speed": speed,
            "disp": disp, "grasp": (grasp_start, grasp_end), "drink_runs": drink_runs,
            "peak_disp": float(peak_disp)}


def plot_reps(stems: list[str], out: Path) -> None:
    n = len(stems)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.8 * n), squeeze=False)
    for k, stem in enumerate(stems):
        name, xyz = load_track(stem)
        r = segment_cup_only(xyz)
        ax = axes[k][0]
        t = np.arange(len(xyz)) / FPS
        ax.plot(t, r["speed"], color="#333", lw=1.0, label="cup speed (mm/s)")
        ax2 = ax.twinx()
        ax2.plot(t, r["disp"], color="#7a4ce8", lw=1.0, ls="--", label="disp from rest (mm)")
        for nm, s, e in r["intervals"]:
            ax.axvspan(s / FPS, e / FPS, color=PHASE_COLORS[nm], alpha=0.30, lw=0)
        ax.set_title(f"{name}   phases: " +
                     " | ".join(f"{nm}:{(e-s)/FPS:.1f}s" for nm, s, e in r["intervals"]),
                     fontsize=8)
        ax.set_ylabel("speed mm/s"); ax2.set_ylabel("disp mm")
        ax.set_xlim(0, t[-1])
    # legend (phase colors)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.4) for c in PHASE_COLORS.values()]
    fig.legend(handles, list(PHASE_COLORS.keys()), loc="upper center",
               ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=110)
    print(f"wrote {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", nargs="*", default=None, help="track stems to plot")
    ap.add_argument("--out", default="experiments/drink_study/cache/cup_only_seg.png")
    args = ap.parse_args()
    if args.plot:
        stems = args.plot
    else:
        # default sample: a couple P01, plus P19 and P23 (fast-shallow drinker)
        allf = sorted(TRACKDIR.glob("*.json"))
        pick = []
        for pref in ("P01_", "P06_", "P19_", "P23_"):
            m = [f.stem for f in allf if f.name.startswith(pref)]
            if m:
                pick.append(m[len(m) // 2])
        stems = pick
    print(f"segmenting {len(stems)} reps: {stems}", flush=True)
    plot_reps(stems, Path(args.out))


if __name__ == "__main__":
    main()
