"""WHY does drinking-phase duration underestimate, and why does TCN make it worse?

Two questions, measured (not reasoned):

Q1. The ~300ms UNDERESTIMATE (shared by KF/lin/TCN). The dwell is
    `near_peak (disp>peak-90) AND speed<120`. We measure, relative to the
    MOCAP-truth dwell [d0,d1]:
      - entry shift (track_d0 - truth_d0)   >0 = starts dwell LATE
      - exit  shift (track_d1 - truth_d1)   <0 = ends dwell EARLY
      - at the truth-dwell EDGES, which gate is binding on the track:
        is speed>=120 (speed gate clips) or disp<=peak-90 (disp gate clips)?

Q2. TCN worse than KF. We measure median IN-DWELL speed for truth/kf/tcn —
    if TCN's in-dwell speed is higher (closer to truth's real motion), it
    trips the speed<120 gate more, shaving the dwell.

Single 75/25 split is fine here: these are mechanism diagnostics (distributions,
not a sub-mm headline). Run from repo root; GPU. ~30s.
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

CACHE = ROOT / "experiments" / "drink_study" / "cache"
DEV, SEQ, HZ = M.DEV, M.SEQ, Q.COMMON_HZ
F2MS = 1000.0 / HZ


def fill_noc(base_l, cons_l, valid, vel):
    out = base_l.copy()
    for s, e in VF.gaps_of(valid):
        p0 = cons_l[s - 1] if s - 1 >= 0 and valid[s - 1] else base_l[s]
        path = [p0]
        for t in range(s, e + 1):
            path.append(path[-1] + vel[t])
        out[s:e + 1] = np.array(path[1:])
    return out


def drink_span(xyz):
    """Return (d0,d1) drinking-phase span in frames, the seg dict, or None."""
    d = S.segment_cup_only(xyz, fps=HZ)
    runs = d["drink_runs"]
    if not runs:
        return None, d
    return (runs[0][0], runs[-1][1]), d


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
        if ep % 100 == 0:
            print(f"  ep {ep} loss {loss.item():.1f}", flush=True)
    net.eval()

    entry, exit_ = {"kf": [], "tcn": []}, {"kf": [], "tcn": []}
    edge_gate = {"speed_only": 0, "disp_only": 0, "both": 0, "neither": 0}  # at truth dwell edges, KF track
    indwell_speed = {"truth": [], "kf": [], "tcn": []}
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

            tspan, td = drink_span(true)
            if tspan is None:
                continue
            td0, td1 = tspan
            for name, trk in [("kf", kf), ("tcn", tcn)]:
                sp, sd = drink_span(trk)
                if sp is None:
                    continue
                entry[name].append((sp[0] - td0) * F2MS)
                exit_[name].append((sp[1] - td1) * F2MS)

            # Q1 gate analysis: at the truth dwell EDGES (just inside), what does the KF
            # track's speed/disp say? These are the frames truth calls "drinking" but the
            # track might not. Use KF's own speed/disp arrays + its own peak.
            _, kd = drink_span(kf)
            kf_sp, kf_disp = kd["speed"], kd["disp"]
            peak = kd.get("peak_disp", kf_disp.max())
            for f in (td0, max(td0, td1 - 1)):
                if f >= len(kf_sp):
                    continue
                fast = kf_sp[f] >= S.DRINK_SPEED            # speed gate would EXCLUDE
                low = kf_disp[f] <= (peak - S.DRINK_DISP_PAD)  # disp gate would EXCLUDE
                key = ("both" if fast and low else "speed_only" if fast else
                       "disp_only" if low else "neither")
                edge_gate[key] += 1

            # Q2: median speed INSIDE the truth dwell, per track
            for name, d in [("truth", td), ("kf", kd), ("tcn", drink_span(tcn)[1])]:
                seg = slice(td0, td1)
                indwell_speed[name].append(float(np.median(d["speed"][seg])))

    print("\n=== Q1. WHERE is the dwell lost? (ms; +entry=starts late, -exit=ends early) ===")
    for name in ("kf", "tcn"):
        e = np.array(entry[name]); x = np.array(exit_[name])
        print(f"  {name}: entry {np.median(e):+.0f}ms (late if +),  exit {np.median(x):+.0f}ms (early if -),  "
              f"net lost {np.median(e) - np.median(x):.0f}ms")
    tot = sum(edge_gate.values())
    print(f"\n=== Q1. At truth-dwell EDGES, which gate excludes the frame? (KF track, n={tot}) ===")
    for k, v in edge_gate.items():
        print(f"  {k:<11}: {v:4d}  ({100*v/max(tot,1):.0f}%)")
    print("  (speed_only / both => the speed<120 gate is the binding constraint)")
    print("\n=== Q2. median IN-DWELL speed inside the TRUTH dwell (mm/s) ===")
    for name in ("truth", "kf", "tcn"):
        a = np.array(indwell_speed[name])
        print(f"  {name:<6}: {np.median(a):.0f} mm/s  (gate is <{S.DRINK_SPEED:.0f})")
    print(f"  -> if tcn > kf, TCN trips the speed gate more inside the dwell -> shorter dwell")

    json.dump({"entry_kf": float(np.median(entry["kf"])), "exit_kf": float(np.median(exit_["kf"])),
               "entry_tcn": float(np.median(entry["tcn"])), "exit_tcn": float(np.median(exit_["tcn"])),
               "edge_gate": edge_gate,
               "indwell_speed": {k: float(np.median(v)) for k, v in indwell_speed.items()}},
              open(CACHE / "phase_seg_why.json", "w"), indent=2)
    print("\nwrote cache/phase_seg_why.json", flush=True)


if __name__ == "__main__":
    main()
