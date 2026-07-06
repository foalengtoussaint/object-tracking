"""Worst reps: OMC vs MMC cup→head distance + the HEAD-centroid health that drives both.

The P19 overlay showed the HEAD marker (not the cup) is the one jumping / dropping out. So the
cup→head distance — which the truth dwell keys off — is corrupted by a noisy HEAD, not a bad cup
track or a bad model. This graph makes that visible per worst rep:

  top row  : cup→head distance, two ways —
             OMC (mocap cup → mocap head, all optical)   solid
             MMC (tracked/video cup → mocap head)         dashed   [what proxy21 is fed]
             + the truth dwell band + threshold
  bottom row: HEAD-centroid health — per-frame step (spikes = jumps) and MISSING frames
             (grey = head cluster gap, i.e. gap-filled/absent). Where the head is missing or
             jumps, BOTH distances go bad → the dwell is wrong.

    python experiments/drink_dwell/omc_mmc.py [--n 6]
Writes ../drink_study/slides/omc_mmc_worst.png
"""
from __future__ import annotations
import sys as _s, pathlib as _p
_s.path.insert(0, str(_p.Path(__file__).resolve().parent))
import argparse, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mocap import load_trial
from truth import dwell_truth
import features as F

HERE = _p.Path(__file__).resolve().parent
_DS = HERE.parents[0] / "drink_study"
RESULTS = HERE / "cache" / "results.json"
ALIGN = _DS / "cache" / "qtm_align.json"
OUT = _DS / "slides" / "omc_mmc_worst.png"


def worst(n):
    al = json.load(open(ALIGN)); m = {}
    for v in al.values():
        if isinstance(v, dict) and v.get("ok"):
            for r in v["reps"]:
                m[r["video"]] = (r["c3d"], r["lag"])
    d = json.load(open(RESULTS)); ip = d["perrep_cols"].index("proxy21")
    rows = sorted(((v[ip], k) for k, v in d["perrep"].items()), reverse=True)[:n]
    return [(err, k, *m[k]) for err, k in rows if k in m]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    reps = worst(a.n)
    ncol = 3; nrow = int(np.ceil(len(reps) / ncol))
    fig, axes = plt.subplots(nrow * 2, ncol, figsize=(6 * ncol, 4.2 * nrow),
                             squeeze=False, gridspec_kw={"height_ratios": [2.4, 1] * nrow})
    for ax in axes.flat:
        ax.axis("off")

    for i, (err, video, c3d, lag) in enumerate(reps):
        row = (i // ncol) * 2; col = i % ncol
        tr = load_trial(c3d)
        rate = tr.rate
        omc = tr.cup_to_head()                              # mocap cup -> mocap head (T_m,)
        head = tr.head_centroid()                           # (T_m,3)
        dw = dwell_truth(tr)
        tm = np.arange(len(omc)) / rate
        # MMC: tracked cup -> mocap head, on the 60Hz track grid
        npz_files = glob.glob(str(F.FUSED_DIR / f"*{video}*.npz"))
        mmc = None
        if npz_files:
            cup_world = np.asarray(np.load(npz_files[0], allow_pickle=True)["fused"], float)
            m4 = F.head_distance(video, cup_world, len(cup_world))
            if m4 is not None:
                mmc = m4[:, 0]; mmc[m4[:, 3] < 0.5] = np.nan   # blank uncovered frames

        # --- top: the two distances ---
        ax = axes[row][col]; ax.axis("on")
        ax.plot(tm, omc, color="#1a8a3a", lw=1.8, label="OMC (mocap cup→head)")
        if mmc is not None:
            tt = np.arange(len(mmc)) / F.HZ + lag / F.HZ
            ax.plot(tt, mmc, color="#c0392b", lw=1.6, ls="--", label="MMC (tracked cup→head)")
        if np.isfinite(dw.thr):
            ax.axhline(dw.thr, color="#888", lw=1, ls=":", label=f"dwell thr {dw.thr:.0f}mm")
        if dw.span:
            ax.axvspan(dw.span[0] / rate, dw.span[1] / rate, color="#e8930a", alpha=0.2,
                       label=f"truth dwell {dw.dur_s:.2f}s")
        ax.set_ylabel("cup→head (mm)", fontsize=9)
        ax.set_title(f"{video.split('_')[0]} {'R' if 'right' in video else 'L'}  "
                     f"proxy21 err {err:.0f}ms", fontsize=10, fontweight="bold")
        ax.legend(fontsize=6.5, loc="upper right", frameon=False)
        ax.tick_params(labelbottom=False)

        # --- bottom: head health ---
        axh = axes[row + 1][col]; axh.axis("on")
        step = np.r_[0.0, np.linalg.norm(np.diff(head, axis=0), axis=1)]
        miss = np.isnan(head[:, 0])
        axh.plot(tm, np.clip(step, 0, 60), color="#333", lw=0.9)
        axh.axhline(20, color="#c0392b", lw=0.6, ls=":")           # jump threshold
        # shade missing-head spans
        for s, e in _runs(miss):
            axh.axvspan(s / rate, e / rate, color="#bbb", alpha=0.5)
        axh.set_ylabel("head step\n(mm/fr)", fontsize=8)
        axh.set_xlabel("time (s)", fontsize=9)
        axh.set_ylim(0, 62)
        nmiss = int(miss.sum())
        axh.text(0.98, 0.85, f"head missing {nmiss}/{len(head)}  max step {step.max():.0f}mm",
                 transform=axh.transAxes, ha="right", fontsize=7, color="#c0392b")

    fig.suptitle("Worst reps — cup→head distance (OMC vs MMC) driven by HEAD-marker jumps / gaps "
                 "(grey = head missing)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=135)
    print(f"wrote {OUT}", flush=True)


def _runs(mask):
    out = []; i = 0; T = len(mask)
    while i < T:
        if mask[i]:
            j = i
            while j < T and mask[j]:
                j += 1
            out.append((i, j)); i = j
        else:
            i += 1
    return out


if __name__ == "__main__":
    main()
