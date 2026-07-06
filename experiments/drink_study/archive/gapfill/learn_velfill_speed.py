"""Velocity-fill TCN + a SCALAR-SPEED loss term, swept.

The base model trains on VECTOR per-frame displacement (pv-V)^2 in gaps. That
minimizes position error but does NOT bound the speed MAGNITUDE ||step|| that the
phase segmenter gates on (<120 mm/s) -> the fill wanders, inflating path-speed and
hurting segmentation even while position improves.

Add a scalar-speed term:  L = vec_mse  +  lam * mean(|  ||pv|| - ||V||  |)
penalizing the predicted step MAGNITUDE deviating from the true step magnitude,
on gap frames. Sweep lam in {0, 0.3, 1.0}. Report, per lam:
  - apex/drinking POSITION error (the -13% headline we must not lose)
  - in-dwell PATH speed + %>120 gate (segmentation health)

Single 75/25 split to see if the lever moves; LOPO-confirm the winner after.
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
from scipy.signal import butter, filtfilt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "drink_study"))
import tune_interp as T
import segment_cup_only as S
import qtm_align as Q
import learn_seq_kf as M
import learn_velfill_v2 as V2
import learn_velocity_fill as VF

CACHE = ROOT / "experiments" / "drink_study" / "cache"
DEV, SEQ, HZ = M.DEV, M.SEQ, Q.COMMON_HZ
LAMS = [0.0, 0.3, 1.0]


def fill_noc(base_l, cons_l, valid, vel):
    out = base_l.copy()
    for s, e in VF.gaps_of(valid):
        p0 = cons_l[s - 1] if s - 1 >= 0 and valid[s - 1] else base_l[s]
        path = [p0]
        for t in range(s, e + 1):
            path.append(path[-1] + vel[t])
        out[s:e + 1] = np.array(path[1:])
    return out


def lp(x, hz):
    b, a = butter(2, hz / (0.5 * HZ), btype="low")
    pad = 3 * max(len(a), len(b))
    return filtfilt(b, a, x, axis=0) if len(x) > pad else x


def truth_span(true):
    runs = S.segment_cup_only(true, fps=HZ)["drink_runs"]
    return (runs[0][0], runs[-1][1]) if runs else None


def in_dwell_pathspeed(xyz, d0, d1):
    seg = xyz[d0:d1]
    if len(seg) < 4 or not np.isfinite(seg).all():
        return None
    step = np.linalg.norm(np.diff(lp(seg, 6.0), axis=0), axis=1) * HZ
    return float(step.mean()), float((step > S.DRINK_SPEED).mean())


def build():
    reps = [r for rep in T._reps() if (r := M.build_seq(rep)) is not None]
    for r in reps:
        valid = np.isfinite(r["cons"]).all(1); r["valid"] = valid
        cl = np.zeros((len(r["cons"]), 3)); cl[valid] = (r["cons"][valid] - r["rest"]) @ r["basis"].T
        r["cons_local_t"] = cl; r["base_local_t"] = (r["base"] - r["rest"]) @ r["basis"].T
        r["vel_tgt"] = np.vstack([np.zeros(3), np.diff(r["tgt"], axis=0)]).astype(np.float32)
        x0 = np.linspace(0, 1, len(valid)); x1 = np.linspace(0, 1, SEQ)
        r["gap_seq"] = (np.interp(x1, x0, valid.astype(float)) <= 0.5) & r["tmask"]
    return reps


def train_eval(lam, trn, te, fin):
    X = torch.tensor(np.stack([r["feats"] for r in trn])).transpose(1, 2).to(DEV)
    Vt = torch.tensor(np.stack([r["vel_tgt"] for r in trn])).transpose(1, 2).to(DEV)
    G = torch.tensor(np.stack([r["gap_seq"] for r in trn]).astype(np.float32)).to(DEV)
    Vspd = torch.linalg.norm(Vt, dim=1, keepdim=True)            # true ||step||, (B,1,SEQ)
    net = V2.TCNreg(fin, ch=64, layers=6, p=0.1).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4); net.train()
    for ep in range(300):
        opt.zero_grad(); pv = net(X); m = G.unsqueeze(1)
        vec = (((pv - Vt) ** 2).sum(1, keepdim=True) * m).sum() / m.sum().clamp(min=1)
        pspd = torch.linalg.norm(pv, dim=1, keepdim=True)
        spd = ((pspd - Vspd).abs() * m).sum() / m.sum().clamp(min=1)
        (vec + lam * spd).backward(); opt.step()
    net.eval()
    pos_kf, pos_tcn, common = {}, {}, []
    path_tcn, over_tcn = [], []
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
            # position error in drinking phase (the -13% metric)
            for trk, store in [(kf, pos_kf), (tcn, pos_tcn)]:
                sc = T._score(trk, r["mr"])
                if sc is None:
                    continue
                _, err, mfi = sc; mfi = np.clip(mfi, 0, len(r["ph"]) - 1)
                dm = r["ph"][mfi] == S.P_DRINK
                if dm.any():
                    store[r["video"]] = float(np.median(err[dm]))
            if r["video"] in pos_kf and r["video"] in pos_tcn:
                common.append(r["video"])
            # in-dwell path-speed of the TCN track (segmentation health)
            sp = truth_span(true)
            if sp:
                ps = in_dwell_pathspeed(tcn, *sp)
                if ps:
                    path_tcn.append(ps[0]); over_tcn.append(ps[1])
    common = [k for k in common if k in pos_kf and k in pos_tcn]
    B = np.array([pos_kf[k] for k in common]); SC = np.array([pos_tcn[k] for k in common])
    return dict(n=len(common), kf_pos=float(np.median(B)), tcn_pos=float(np.median(SC)),
                pos_gain_pct=float((np.median(SC) / np.median(B) - 1) * 100),
                path_speed=float(np.median(path_tcn)), over_gate=float(np.median(over_tcn)))


def main():
    print(f"device {DEV}; building...", flush=True)
    reps = build(); fin = reps[0]["feats"].shape[1]
    rng = np.random.default_rng(1); idx = rng.permutation(len(reps))
    trn = [reps[i] for i in idx[:int(0.75 * len(reps))]]; te = [reps[i] for i in idx[int(0.75 * len(reps)):]]
    print(f"{len(trn)} train / {len(te)} test reps; sweeping lam={LAMS}", flush=True)
    res = {}
    for lam in LAMS:
        r = train_eval(lam, trn, te, fin)
        res[str(lam)] = r
        print(f"\n  lam={lam}:  drink-pos KF {r['kf_pos']:.1f} -> TCN {r['tcn_pos']:.1f}mm "
              f"({r['pos_gain_pct']:+.0f}%)   |   in-dwell path-speed {r['path_speed']:.0f}mm/s "
              f"(truth ~48), %>120 {100*r['over_gate']:.0f}%", flush=True)
    print("\n=== summary (lam: pos-gain% / path-speed / %>120) — want pos-gain stays <<0, "
          "path-speed drops toward 48, %>120 drops ===")
    for lam in LAMS:
        r = res[str(lam)]
        print(f"  lam={lam:<4} pos {r['pos_gain_pct']:+5.0f}%   path {r['path_speed']:3.0f}mm/s   "
              f">120 {100*r['over_gate']:2.0f}%")
    json.dump(res, open(CACHE / "learn_velfill_speed.json", "w"), indent=2)
    print("\nwrote cache/learn_velfill_speed.json", flush=True)


if __name__ == "__main__":
    main()
