"""Option 1: TCN velocity as a KF MEASUREMENT (soft anchor), not a hard-anchored fill.

The problem (user): the hard-anchored fill (fill_track: start at cons[s-1], integrate,
land on cons[e+1]) WELDS the trajectory to the raw boundary detections. If a boundary
detection is a bad outlier, the fill is forced through it -- error baked in BEFORE any
smoothing. Plain KF+RTS rejects that outlier via its innovation test; the fill can't.

Fix: ONE KF pass over the whole rep. The TCN never builds a position path. Instead:
  - VISIBLE frames -> POSITION measurement = detection (consensus), with an innovation
    GATE so a wild outlier is down-weighted/rejected (this is the outlier rejection the
    hard-anchor throws away).
  - GAP frames     -> VELOCITY measurement = TCN velocity, measuring state[3:6].
The KF integrates the TCN velocity through its OWN filtered (outlier-rejected) state, so
the gap start is the KF's estimate, not the raw bad anchor. Then RTS smooth.

Compares on the SAME test reps:
  kf            : plain KF+RTS on consensus (current baseline; rejects outliers, coasts gaps)
  hardfill      : fill_track velocity-fill -> KF+RTS  (the -14% pipeline, hard anchor)
  fused         : THIS (velocity as measurement, gated position)  [+ gate sweep]

Single 75/25 split first; report drinking-phase pos err + the P13/P06 blow-up reps.
Run from repo root; GPU.
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import json, sys
from pathlib import Path
import numpy as np
import torch

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
Qp = 200.0 ** 2          # process noise (same as kf_consensus)
R_POS = 30.0 ** 2        # detection position noise
R_VEL = 60.0 ** 2        # TCN velocity meas noise (mm/frame->mm/s handled below)


def fill_noc(base_l, cons_l, valid, vel):
    out = base_l.copy()
    for s, e in VF.gaps_of(valid):
        p0 = cons_l[s - 1] if s - 1 >= 0 and valid[s - 1] else base_l[s]
        path = [p0]
        for t in range(s, e + 1):
            path.append(path[-1] + vel[t])
        out[s:e + 1] = np.array(path[1:])
    return out


def kf_rts_fused(cons, vel_world, fps=HZ, q=Qp, r_pos=R_POS, r_vel=R_VEL, gate=None):
    """One KF+RTS pass fusing detections (position, gated) + TCN velocity (velocity meas).
      cons      : (T,3) detections, NaN where occluded -> position measurement when valid
      vel_world : (T,3) TCN per-FRAME displacement in WORLD frame -> velocity measurement
                  applied on GAP frames (where cons is NaN). Converted to per-second.
      gate      : if not None, chi-square gate (in std) on the POSITION innovation; a
                  detection whose Mahalanobis distance exceeds `gate` is inflated (down-
                  weighted) rather than trusted -> outlier rejection the hard-anchor lacks.
    """
    Tn = len(cons); dt = 1.0 / fps
    F = np.eye(6); F[:3, 3:] = dt * np.eye(3)
    Hp = np.zeros((3, 6)); Hp[:, :3] = np.eye(3)          # position measurement
    Hv = np.zeros((3, 6)); Hv[:, 3:] = np.eye(3)          # velocity measurement
    Qm = np.zeros((6, 6))
    Qm[:3, :3] = q * dt ** 3 / 3 * np.eye(3); Qm[:3, 3:] = q * dt ** 2 / 2 * np.eye(3)
    Qm[3:, :3] = q * dt ** 2 / 2 * np.eye(3); Qm[3:, 3:] = q * dt * np.eye(3)
    Rp = r_pos * np.eye(3); Rv = r_vel * np.eye(3)
    valid = np.isfinite(cons).all(1); idx = np.flatnonzero(valid)
    if len(idx) < 2:
        return np.full((Tn, 3), np.nan)
    x = np.zeros(6); x[:3] = cons[idx[0]]
    P = np.diag([50, 50, 50, 500, 500, 500.0]) ** 2
    xs_p, Ps_p, xs_u, Ps_u = [], [], [], []
    for t in range(Tn):
        x = F @ x; P = F @ P @ F.T + Qm
        xs_p.append(x.copy()); Ps_p.append(P.copy())
        if valid[t]:
            # POSITION update from detection, with optional innovation gate
            z = cons[t]; y = z - Hp @ x
            Sp = Hp @ P @ Hp.T + Rp
            R_eff = Rp
            if gate is not None:
                md2 = float(y @ np.linalg.inv(Sp) @ y)      # squared Mahalanobis dist
                if md2 > gate ** 2:                          # outlier: inflate its noise
                    R_eff = Rp * (md2 / gate ** 2)           # down-weight ~ how far out it is
                    Sp = Hp @ P @ Hp.T + R_eff
            K = P @ Hp.T @ np.linalg.inv(Sp)
            x = x + K @ y; P = (np.eye(6) - K @ Hp) @ P
        elif np.isfinite(vel_world[t]).all():
            # VELOCITY update from TCN (per-frame disp -> per-second) on gap frames
            zv = vel_world[t] * fps
            yv = zv - Hv @ x
            Sv = Hv @ P @ Hv.T + Rv
            K = P @ Hv.T @ np.linalg.inv(Sv)
            x = x + K @ yv; P = (np.eye(6) - K @ Hv) @ P
        xs_u.append(x.copy()); Ps_u.append(P.copy())
    xs_s = [None] * Tn; xs_s[-1] = xs_u[-1]
    for t in range(Tn - 2, -1, -1):
        C = Ps_u[t] @ F.T @ np.linalg.inv(Ps_p[t + 1])
        xs_s[t] = xs_u[t] + C @ (xs_s[t + 1] - xs_p[t + 1])
    return np.array([s[:3] for s in xs_s])


def drinkerr(trk, r):
    sc = T._score(trk, r["mr"])
    if sc is None:
        return None
    _, err, mfi = sc; mfi = np.clip(mfi, 0, len(r["ph"]) - 1)
    dm = r["ph"][mfi] == S.P_DRINK
    return float(np.median(err[dm])) if dm.any() else None


def main():
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
    trn = [reps[i] for i in idx[:int(.75 * len(reps))]]; te = [reps[i] for i in idx[int(.75 * len(reps)):]]
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

    GATES = [None, 5.0, 3.0]      # no gate, loose, tight
    methods = ["kf", "hardfill"] + [f"fused_g{g}" for g in GATES]
    err = {m_: {} for m_ in methods}
    watch = ["P13_P13_drinking_right_20240216_161430", "P06_P06_drinking_left_20240123_105548",
             "P24_P24_drinking_left_20240724_105932"]
    watch_rows = {}
    with torch.no_grad():
        for r in te:
            if not (~r["valid"]).any():
                continue
            pv = net(torch.tensor(r["feats"][None]).transpose(1, 2).to(DEV))[0].cpu().numpy().T
            x0 = np.linspace(0, 1, SEQ); x1 = np.linspace(0, 1, r["Tn"])
            vel_t = np.stack([np.interp(x1, x0, pv[:, k]) for k in range(3)], 1)   # local per-frame
            vel_world = vel_t @ r["basis"]                                          # rotate to world (no translation for a delta)
            tcn = fill_noc(r["base_local_t"], r["cons_local_t"], r["valid"], vel_t) @ r["basis"] + r["rest"]
            tracks = {"kf": r["base"],
                      "hardfill": kf_rts_on_consensus(tcn, HZ)}
            for g in GATES:
                tracks[f"fused_g{g}"] = kf_rts_fused(r["cons"], vel_world, gate=g)
            row = {}
            for name, trk in tracks.items():
                de = drinkerr(trk, r)
                if de is not None:
                    err[name][r["video"]] = de; row[name] = de
            for w in watch:
                if w in r["video"]:
                    watch_rows[r["video"]] = row
    common = sorted(set.intersection(*[set(err[m_]) for m_ in methods]))
    base = np.median([err["kf"][k] for k in common])
    print(f"\n=== drinking-phase pos err (median mm), n={len(common)} ===")
    for m_ in methods:
        a = np.array([err[m_][k] for k in common])
        wins = (a < np.array([err["kf"][k] for k in common])).sum()
        print(f"  {m_:<12}: {np.median(a):5.1f}mm  ({(np.median(a)/base-1)*100:+.0f}%)   "
              f"mean {a.mean():5.1f}   beats-KF {wins}/{len(common)}")
    print(f"\n=== the blow-up reps (hardfill was WAY worse than KF here) ===")
    for v, row in watch_rows.items():
        print(f"  {v[:40]:<40} " + "  ".join(f"{m_.replace('fused_g','g')} {row.get(m_, float('nan')):.0f}"
                                             for m_ in methods))
    json.dump({"n": len(common), "median": {m_: float(np.median([err[m_][k] for k in common])) for m_ in methods},
               "mean": {m_: float(np.mean([err[m_][k] for k in common])) for m_ in methods},
               "watch": watch_rows},
              open(CACHE / "learn_velfill_fused.json", "w"), indent=2)
    print("\nwrote cache/learn_velfill_fused.json", flush=True)


if __name__ == "__main__":
    main()
