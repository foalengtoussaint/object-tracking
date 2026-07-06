"""Plot the reps where PLAIN KF+RTS beats the TCN-fill pipeline (the blow-ups).

For a handful of named reps, show over the drinking phase:
  TOP   : reach-axis POSITION — true / plain KF+RTS / TCN-fill->KF+RTS
  BOTTOM: SPEED + the 120 gate; gap regions shaded, gap-exit seams marked.

Goal: SEE why the TCN fill hurts an easy rep the KF had nailed.

    python experiments/drink_study/viz_kf_wins.py [--reps SUBSTR ...]
Run from repo root; GPU. Writes cache/kf_wins.png
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import sys, argparse
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
import learn_velfill_fused as FU
from kf_consensus import kf_rts_on_consensus

CACHE = ROOT / "experiments" / "drink_study" / "cache"
DEV, SEQ, HZ = M.DEV, M.SEQ, Q.COMMON_HZ
DEFAULT_REPS = ["P13_P13_drinking_right_20240216_161430",
                "P06_P06_drinking_left_20240123_105548",
                "P24_P24_drinking_left_20240724_105932"]


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
    b, a = butter(2, hz / (0.5 * HZ), btype="low"); p = 3 * max(len(a), len(b))
    return filtfilt(b, a, x, axis=0) if len(x) > p else x


def speed_of(xyz):
    return np.r_[0, np.linalg.norm(np.diff(lp(xyz), axis=0), axis=1)] * HZ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="*", default=DEFAULT_REPS)
    args = ap.parse_args()
    print(f"device {DEV}; building...", flush=True)
    reps = [r for rep in T._reps() if (r := M.build_seq(rep)) is not None]
    for r in reps:
        v = np.isfinite(r["cons"]).all(1); r["valid"] = v
        cl = np.zeros((len(r["cons"]), 3)); cl[v] = (r["cons"][v] - r["rest"]) @ r["basis"].T
        r["cons_local_t"] = cl; r["base_local_t"] = (r["base"] - r["rest"]) @ r["basis"].T
        r["vel_tgt"] = np.vstack([np.zeros(3), np.diff(r["tgt"], axis=0)]).astype(np.float32)
        x0 = np.linspace(0, 1, len(v))
        r["gap_seq"] = (np.interp(np.linspace(0, 1, SEQ), x0, v.astype(float)) <= 0.5) & r["tmask"]
    fin = reps[0]["feats"].shape[1]
    rng = np.random.default_rng(1); idx = rng.permutation(len(reps))
    trn = [reps[i] for i in idx[:int(.75 * len(reps))]]
    X = torch.tensor(np.stack([r["feats"] for r in trn])).transpose(1, 2).to(DEV)
    Vt = torch.tensor(np.stack([r["vel_tgt"] for r in trn])).transpose(1, 2).to(DEV)
    G = torch.tensor(np.stack([r["gap_seq"] for r in trn]).astype(np.float32)).to(DEV)
    net = V2.TCNreg(fin, ch=64, layers=6, p=0.1).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4); net.train()
    print(f"training on {len(trn)} reps...", flush=True)
    for ep in range(300):
        opt.zero_grad(); pv = net(X); m = G.unsqueeze(1)
        (((pv - Vt) ** 2).sum(1, keepdim=True) * m).sum().div(m.sum().clamp(min=1)).backward(); opt.step()
    net.eval()

    picks = []
    for sub in args.reps:
        cand = [r for r in reps if sub in r["video"]]
        if cand:
            picks.append(cand[0])
        else:
            print(f"  !! no rep matching {sub}", flush=True)
    n = len(picks)
    fig, axes = plt.subplots(2, n, figsize=(6.2 * n, 8), squeeze=False)
    with torch.no_grad():
        for col, r in enumerate(picks):
            pv = net(torch.tensor(r["feats"][None]).transpose(1, 2).to(DEV))[0].cpu().numpy().T
            x0 = np.linspace(0, 1, SEQ); x1 = np.linspace(0, 1, r["Tn"])
            vel_t = np.stack([np.interp(x1, x0, pv[:, k]) for k in range(3)], 1)
            tcn = fill_noc(r["base_local_t"], r["cons_local_t"], r["valid"], vel_t) @ r["basis"] + r["rest"]
            rts = kf_rts_on_consensus(tcn, HZ); kf = r["base"]
            fused = FU.kf_rts_fused(r["cons"], vel_t @ r["basis"], gate=3.0)   # option 1
            true = np.stack([np.interp(x1, x0, r["tgt"][:, k]) for k in range(3)], 1) @ r["basis"] + r["rest"]
            runs = S.segment_cup_only(true, fps=HZ)["drink_runs"]
            d0, d1 = (runs[0][0], runs[-1][1]) if runs else (0, len(true))
            a = max(0, d0 - 15); b = min(len(true), d1 + 15)
            basis, rest = r["basis"], r["rest"]
            reach = lambda xyz: ((xyz - rest) @ basis.T)[:, 0]
            t = (np.arange(a, b) - d0) / HZ * 1000
            tl, kl, rl = reach(true), reach(kf), reach(rts)
            fl = reach(fused)                                    # option-1 fused KF
            cl_tcn = reach(tcn)                                  # raw TCN fill, pre-KF+RTS
            ts, ks, rs = speed_of(true), speed_of(kf), speed_of(rts)
            fs = speed_of(fused)
            # raw detections (consensus): valid triangulation points, reach axis
            det = r["cons"]; detv = np.isfinite(det).all(1)
            det_reach = np.full(len(det), np.nan)
            det_reach[detv] = ((det[detv] - rest) @ basis.T)[:, 0]
            ke = np.abs(kl[d0:d1] - tl[d0:d1]).mean(); re = np.abs(rl[d0:d1] - tl[d0:d1]).mean()
            fe = np.abs(fl[d0:d1] - tl[d0:d1]).mean()
            axp, axs = axes[0][col], axes[1][col]
            for ax in (axp, axs):
                for s, e in VF.gaps_of(r["valid"]):
                    if e >= a and s <= b:
                        ax.axvspan((max(s, a) - d0) / HZ * 1000, (min(e, b) - d0) / HZ * 1000,
                                   color="#999", alpha=0.15, lw=0)
                        if e + 1 < len(r["valid"]) and r["valid"][e + 1] and a <= e + 1 < b:
                            ax.axvline((e + 1 - d0) / HZ * 1000, color="#d62728", ls=":", lw=1.0, alpha=0.6)
            axp.plot(t, tl[a:b], "#111", lw=2.4, label="TRUE (mocap)")
            axp.plot(t, kl[a:b], "#1f77b4", lw=1.6, label="plain KF+RTS")
            axp.plot(t, cl_tcn[a:b], "#ff7f0e", lw=1.3, ls="--", label="raw TCN fill (hard-anchor)")
            axp.plot(t, rl[a:b], "#2ca02c", lw=1.5, alpha=0.7, label="hardfill → KF+RTS")
            axp.plot(t, fl[a:b], "#9467bd", lw=2.2, label="FUSED KF (option 1)")
            axp.scatter(t, det_reach[a:b], s=16, c="#8b008b", zorder=5, marker="o",
                        label="raw detections (consensus)")
            axp.set_title(f"{r['video'][:24]}\nin-dwell |err|  KF {ke:.0f}  hardfill {re:.0f}  FUSED {fe:.0f} mm",
                          fontsize=10, color=("#070" if fe <= ke + 1 else "#b00"))
            if col == 0:
                axp.set_ylabel("reach POSITION (mm)")
            axp.legend(fontsize=8, loc="lower right"); axp.grid(alpha=0.25)
            cs_tcn = speed_of(tcn)
            axs.axhline(S.DRINK_SPEED, color="#444", ls="--", lw=1.0)
            axs.plot(t, ts[a:b], "#111", lw=2.4)
            axs.plot(t, ks[a:b], "#1f77b4", lw=1.6)
            axs.plot(t, cs_tcn[a:b], "#ff7f0e", lw=1.3, ls="--")
            axs.plot(t, rs[a:b], "#2ca02c", lw=1.5, alpha=0.7)
            axs.plot(t, fs[a:b], "#9467bd", lw=2.2)
            if col == 0:
                axs.set_ylabel("SPEED (mm/s)")
            axs.set_xlabel("ms rel. drink start  (grey=gap, red=seam)")
            axs.grid(alpha=0.25); axs.set_ylim(0, 300)
    fig.suptitle("The blow-up reps: hard-anchor fill (green) is welded to a bad boundary detection; "
                 "the FUSED KF (purple) rejects it like plain KF (blue) and recovers", fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = CACHE / "kf_wins.png"
    fig.savefig(out, dpi=115)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
