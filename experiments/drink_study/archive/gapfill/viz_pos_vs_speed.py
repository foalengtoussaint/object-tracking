"""Visualize WHY better position != better segmentation.

One drinking rep, two stacked panels over the drinking phase:
  TOP   : POSITION (reach axis) — true / KF / TCN. TCN hugs the truth (better).
  BOTTOM: SPEED (mm/s)          — true / KF / TCN + the 120 gate. TCN buzzes
          ABOVE the others and crosses the gate more (worse segmentation).

Shows concretely: TCN closer in position yet faster/wigglier in speed, because
speed is the derivative — closer-on-average-but-wigglier = better pos, worse speed.

    python experiments/drink_study/viz_pos_vs_speed.py
Run from repo root; GPU. Writes cache/pos_vs_speed.png
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "drink_study"))
import tune_interp as T
import segment_cup_only as S
import qtm_align as Q
import learn_seq_kf as M
import learn_velfill_v2 as V2
import learn_velocity_fill as VF
from kf_consensus import kf_rts_on_consensus

CACHE = ROOT / "experiments" / "drink_study" / "cache"
DEV, SEQ, HZ = M.DEV, M.SEQ, Q.COMMON_HZ


def fill_noc(base_l, cons_l, valid, vel):
    out = base_l.copy()
    for s, e in VF.gaps_of(valid):
        p0 = cons_l[s - 1] if s - 1 >= 0 and valid[s - 1] else base_l[s]
        path = [p0]
        for t in range(s, e + 1):
            path.append(path[-1] + vel[t])
        out[s:e + 1] = np.array(path[1:])
    return out


def lp(x, hz=6.0):
    b, a = butter(2, hz / (0.5 * HZ), btype="low")
    pad = 3 * max(len(a), len(b))
    return filtfilt(b, a, x, axis=0) if len(x) > pad else x


def speed_of(xyz):
    return np.r_[0, np.linalg.norm(np.diff(lp(xyz), axis=0), axis=1)] * HZ


def main():
    print(f"device {DEV}; building...", flush=True)
    reps = [r for rep in T._reps() if (r := M.build_seq(rep)) is not None]
    for r in reps:
        valid = np.isfinite(r["cons"]).all(1); r["valid"] = valid
        cl = np.zeros((len(r["cons"]), 3)); cl[valid] = (r["cons"][valid] - r["rest"]) @ r["basis"].T
        r["cons_local_t"] = cl; r["base_local_t"] = (r["base"] - r["rest"]) @ r["basis"].T
        r["vel_tgt"] = np.vstack([np.zeros(3), np.diff(r["tgt"], axis=0)]).astype(np.float32)
        x0 = np.linspace(0, 1, len(valid)); x1 = np.linspace(0, 1, SEQ)
        r["gap_seq"] = (np.interp(x1, x0, valid.astype(float)) <= 0.5) & r["tmask"]
    fin = reps[0]["feats"].shape[1]
    rng = np.random.default_rng(1); idx = rng.permutation(len(reps))
    trn = [reps[i] for i in idx[:int(0.75 * len(reps))]]; te = [reps[i] for i in idx[int(0.75 * len(reps)):]]
    X = torch.tensor(np.stack([r["feats"] for r in trn])).transpose(1, 2).to(DEV)
    Vt = torch.tensor(np.stack([r["vel_tgt"] for r in trn])).transpose(1, 2).to(DEV)
    G = torch.tensor(np.stack([r["gap_seq"] for r in trn]).astype(np.float32)).to(DEV)
    net = V2.TCNreg(fin, ch=64, layers=6, p=0.1).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4); net.train()
    print(f"training on {len(trn)} reps...", flush=True)
    for ep in range(300):
        opt.zero_grad(); pv = net(X); m = G.unsqueeze(1)
        loss = (((pv - Vt) ** 2).sum(1, keepdim=True) * m).sum() / m.sum().clamp(min=1)
        loss.backward(); opt.step()
    net.eval()

    # pick a clear example: a held-out rep with a gap during the dwell and a sizeable dwell
    best = None
    with torch.no_grad():
        for r in te:
            if not (~r["valid"]).any():
                continue
            pv = net(torch.tensor(r["feats"][None]).transpose(1, 2).to(DEV))[0].cpu().numpy().T
            x0 = np.linspace(0, 1, SEQ); x1 = np.linspace(0, 1, r["Tn"])
            vel_t = np.stack([np.interp(x1, x0, pv[:, k]) for k in range(3)], 1)
            tcn = fill_noc(r["base_local_t"], r["cons_local_t"], r["valid"], vel_t) @ r["basis"] + r["rest"]
            kf = r["base"]
            true = np.stack([np.interp(x1, x0, r["tgt"][:, k]) for k in range(3)], 1) @ r["basis"] + r["rest"]
            runs = S.segment_cup_only(true, fps=HZ)["drink_runs"]
            if not runs:
                continue
            d0, d1 = runs[0][0], runs[-1][1]
            if d1 - d0 < 20:
                continue
            # gap overlapping the dwell?
            gapfrac = (~r["valid"][d0:d1]).mean()
            if gapfrac < 0.2:
                continue
            # how much better is TCN in position over the dwell (reach axis)?
            tl = ((true - r["rest"]) @ r["basis"].T)[:, 0]
            kl = ((kf - r["rest"]) @ r["basis"].T)[:, 0]
            cl = ((tcn - r["rest"]) @ r["basis"].T)[:, 0]
            kf_err = np.abs(kl[d0:d1] - tl[d0:d1]).mean()
            tcn_err = np.abs(cl[d0:d1] - tl[d0:d1]).mean()
            gain = kf_err - tcn_err          # >0 = TCN better in position
            score = gain
            if best is None or score > best[0]:
                best = (score, r["video"], true, kf, tcn, d0, d1, r)
    _, vid, true, kf, tcn, d0, d1, r = best
    print(f"example rep {vid}: dwell [{d0},{d1}] ({(d1-d0)/HZ*1000:.0f}ms), "
          f"gap {(~r['valid'][d0:d1]).mean()*100:.0f}% of dwell", flush=True)

    # the PRODUCTION pipeline: TCN-fill the gaps, then KF+RTS over the whole thing
    rts = kf_rts_on_consensus(tcn, HZ)

    # window a little around the dwell for context
    a = max(0, d0 - 15); b = min(len(true), d1 + 15)
    t = (np.arange(a, b) - d0) / HZ * 1000        # ms relative to dwell start
    basis, rest = r["basis"], r["rest"]
    reach = lambda xyz: ((xyz - rest) @ basis.T)[:, 0]
    tl, kl, cl, rl = reach(true), reach(kf), reach(tcn), reach(rts)
    ts, ks, cs, rs = speed_of(true), speed_of(kf), speed_of(tcn), speed_of(rts)
    # gap-exit seam frames within the window (where the 1-frame snap lives)
    seams = [e + 1 for s, e in VF.gaps_of(r["valid"])
             if e + 1 < len(r["valid"]) and r["valid"][e + 1] and a <= e + 1 < b]

    fig, (axp, axs) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    C = {"true": "#111", "kf": "#1f77b4", "tcn": "#d62728", "rts": "#2ca02c"}
    # shade the TRUE drinking phase
    for ax in (axp, axs):
        ax.axvspan((d0 - d0) / HZ * 1000, (d1 - d0) / HZ * 1000, color="#e8554c", alpha=0.10, lw=0,
                   label="true drinking phase")
        # shade the gap region(s) within the window
        for s, e in VF.gaps_of(r["valid"]):
            if e >= a and s <= b:
                ax.axvspan((max(s, a) - d0) / HZ * 1000, (min(e, b) - d0) / HZ * 1000,
                           color="#999", alpha=0.12, lw=0)

    for x in seams:
        for ax in (axp, axs):
            ax.axvline((x - d0) / HZ * 1000, color="#d62728", ls=":", lw=1.0, alpha=0.6)
    axp.plot(t, tl[a:b], color=C["true"], lw=2.4, label="TRUE (mocap)")
    axp.plot(t, kl[a:b], color=C["kf"], lw=1.6, label="KF+RTS (plain, coasts gap)")
    axp.plot(t, cl[a:b], color=C["tcn"], lw=1.4, alpha=0.7, label="TCN fill (raw)")
    axp.plot(t, rl[a:b], color=C["rts"], lw=2.0, label="TCN fill → KF+RTS (pipeline)")
    kf_e = np.abs(kl[d0:d1] - tl[d0:d1]).mean(); tcn_e = np.abs(cl[d0:d1] - tl[d0:d1]).mean()
    rts_e = np.abs(rl[d0:d1] - tl[d0:d1]).mean()
    axp.set_ylabel("reach-axis POSITION (mm)\nhigher = closer to mouth")
    axp.set_title(f"Rep {vid}: filled+smoothed track hugs truth "
                  f"(in-dwell |err|  KF {kf_e:.0f}  raw-fill {tcn_e:.0f}  fill→RTS {rts_e:.0f} mm)", fontsize=11)
    axp.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    axp.grid(alpha=0.25)

    axs.axhline(S.DRINK_SPEED, color="#444", ls="--", lw=1.2,
                label=f"drinking speed gate ({S.DRINK_SPEED:.0f} mm/s)")
    axs.plot(t, ts[a:b], color=C["true"], lw=2.4, label="TRUE (mocap)")
    axs.plot(t, ks[a:b], color=C["kf"], lw=1.6, label="KF+RTS (plain)")
    axs.plot(t, cs[a:b], color=C["tcn"], lw=1.4, alpha=0.7, label="TCN fill (raw) — SEAM SPIKE")
    axs.plot(t, rs[a:b], color=C["rts"], lw=2.0, label="TCN fill → KF+RTS")
    seg = slice(d0, d1)
    axs.set_ylabel("SPEED (mm/s)\nsegmenter gates on THIS")
    axs.set_xlabel("time relative to true drinking-phase start (ms)  "
                   "(red dotted = gap-exit seam)")
    axs.set_title(f"The spike is ONLY at the seam & KF+RTS removes it  (in-dwell path-speed  "
                  f"true {ts[seg].mean():.0f}  raw-fill {cs[seg].mean():.0f}  fill→RTS {rs[seg].mean():.0f} mm/s)",
                  fontsize=11)
    axs.legend(loc="upper right", fontsize=9, framealpha=0.9, ncol=2)
    axs.grid(alpha=0.25)
    # cap y so the ~1200mm/s raw-fill seam spike goes off-scale (annotated) and the
    # true/KF/RTS traces + the 120 gate stay readable
    axs.set_ylim(0, 400)
    for x in seams:
        peak = cs[x] if x < len(cs) else 0
        if peak > 400:
            axs.annotate(f"raw seam {peak:.0f}→ off-scale", xy=((x - d0) / HZ * 1000, 395),
                         xytext=((x - d0) / HZ * 1000, 350), fontsize=8, color=C["tcn"],
                         ha="center", arrowprops=dict(arrowstyle="->", color=C["tcn"]))

    fig.suptitle("The excess speed is a 1-frame SEAM artifact at the gap edge — "
                 "TCN-fill → KF+RTS removes it & keeps the apex",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = CACHE / "pos_vs_speed.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
