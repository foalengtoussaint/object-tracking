"""Three questions, measured:

Q1. KF in-dwell speed (73) ~= TCN (71), yet KF segments BETTER. What actually
    separates them? Compare, inside the truth dwell, KF vs TCN:
      - speed std (not just median) and # frames over the 120 gate
      - and the dwell-edge speed specifically (the frames that decide d0/d1).

Q2. Is the ~70 mm/s 'speed' JITTER or real motion? Decompose each track's
    in-dwell speed into:
      - COHERENT speed = |net displacement over dwell| / duration  (real transport)
      - PATH speed     = mean per-frame |Δ| * fps                  (what the gate sees)
    PATH >> COHERENT and PATH >> truth's path  =>  jitter (path padding with no net travel).
    Also high-freq power: speed of the 6Hz-filtered vs a heavier 3Hz-filtered track.

Q3. If jitter: does per-frame velocity DIRECTION wobble? Measure mean cosine
    between consecutive velocity vectors inside the dwell (1=always same dir,
    ~0=random). Truth should be ~1 (smooth), tracks lower if jittery.

Single 75/25 split; mechanism diagnostic. Run from repo root; GPU. ~30s.
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
    if len(x) <= pad:
        return x
    return filtfilt(b, a, x, axis=0)


def truth_span(true):
    d = S.segment_cup_only(true, fps=HZ)
    runs = d["drink_runs"]
    return (runs[0][0], runs[-1][1]) if runs else None


def metrics_in_dwell(xyz, d0, d1):
    """Decompose motion inside [d0,d1]. xyz raw track (may have NaN gaps already filled
    upstream for kf/tcn; truth is dense)."""
    seg = xyz[d0:d1]
    if len(seg) < 4 or not np.isfinite(seg).all():
        return None
    # path speed (6Hz, what the segmenter sees)
    s6 = lp(seg, 6.0)
    step = np.linalg.norm(np.diff(s6, axis=0), axis=1)
    path_speed = step.mean() * HZ
    over_gate = float((step * HZ > S.DRINK_SPEED).mean())     # frac frames over 120
    # coherent speed = net displacement / time
    net = np.linalg.norm(s6[-1] - s6[0])
    dur = (d1 - d0) / HZ
    coh_speed = net / max(dur, 1e-6)
    # high-freq: how much speed survives only because of >3Hz content
    s3 = lp(seg, 3.0)
    path_speed_3 = np.linalg.norm(np.diff(s3, axis=0), axis=1).mean() * HZ
    # direction coherence: cos between consecutive velocity vectors
    v = np.diff(s6, axis=0)
    vn = np.linalg.norm(v, axis=1, keepdims=True)
    vu = v / np.clip(vn, 1e-6, None)
    cos = (vu[1:] * vu[:-1]).sum(1)
    dir_coh = float(np.median(cos)) if len(cos) else np.nan
    return dict(path=path_speed, coh=coh_speed, over=over_gate,
                path3=path_speed_3, dir_coh=dir_coh, std=float((step * HZ).std()))


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
        if ep % 150 == 0:
            print(f"  ep {ep} loss {loss.item():.1f}", flush=True)
    net.eval()

    agg = {k: {m_: [] for m_ in ("path", "coh", "over", "path3", "dir_coh", "std")}
           for k in ("truth", "kf", "tcn")}
    edge_speed = {"kf": [], "tcn": []}     # speed at the truth-dwell exit frame
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
            sp = truth_span(true)
            if sp is None:
                continue
            d0, d1 = sp
            for name, trk in [("truth", true), ("kf", kf), ("tcn", tcn)]:
                mm = metrics_in_dwell(trk, d0, d1)
                if mm:
                    for k_, v_ in mm.items():
                        agg[name][k_].append(v_)
            # exit-edge speed: speed at the few frames around truth d1
            for name, trk in [("kf", kf), ("tcn", tcn)]:
                s6 = lp(trk, 6.0)
                step = np.r_[0, np.linalg.norm(np.diff(s6, axis=0), axis=1)] * HZ
                w = slice(max(0, d1 - 3), min(len(step), d1 + 3))
                if np.isfinite(step[w]).all() and (w.stop - w.start) > 0:
                    edge_speed[name].append(float(np.median(step[w])))

    def med(name, k): return float(np.median(agg[name][k]))
    print("\n=== Q2/Q3. In-dwell motion decomposition (mm/s unless noted) ===")
    print(f"  {'track':<7}{'PATH spd':>9}{'COH spd':>9}{'path-3Hz':>9}{'%>120':>7}{'dir-cos':>8}{'spd-std':>8}")
    for name in ("truth", "kf", "tcn"):
        print(f"  {name:<7}{med(name,'path'):>9.0f}{med(name,'coh'):>9.0f}{med(name,'path3'):>9.0f}"
              f"{100*med(name,'over'):>6.0f}%{med(name,'dir_coh'):>8.2f}{med(name,'std'):>8.0f}")
    print("  PATH>>COH => path padded by jitter (no net travel). dir-cos<1 => velocity wobbles.")
    print("  path-3Hz << path-6Hz => the speed is HIGH-FREQUENCY (jitter), killed by heavier smoothing.")
    print("\n=== Q1. Speed at the TRUTH-DWELL EXIT edge (the frame that sets d1) ===")
    for name in ("kf", "tcn"):
        a = np.array(edge_speed[name])
        print(f"  {name}: {np.median(a):.0f} mm/s   (gate <{S.DRINK_SPEED:.0f}; higher => exits dwell earlier)")

    json.dump({"in_dwell": {n: {k: med(n, k) for k in agg[n]} for n in agg},
               "exit_edge_speed": {n: float(np.median(edge_speed[n])) for n in edge_speed}},
              open(CACHE / "phase_seg_jitter.json", "w"), indent=2)
    print("\nwrote cache/phase_seg_jitter.json", flush=True)


if __name__ == "__main__":
    main()
