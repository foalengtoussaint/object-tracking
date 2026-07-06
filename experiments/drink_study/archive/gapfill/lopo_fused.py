"""LOPO confirmation of the FUSED KF (option 1) + CACHE the tracks for reuse.

Fused KF: detections as gated position measurements + TCN velocity as velocity
measurements -> single KF+RTS. Beats hard-anchor fill on the single split (median
-32%, mean 26.2, only 6 blow-ups vs 16). This confirms it leave-one-participant-out.

CACHES per rep (cache/lopo_fused/<video>.npz) so downstream work (esp. improving
PHASE SEGMENTATION) needs NO GPU / NO retraining -- just reload the tracks:
  kf    : plain KF+RTS on consensus (world)
  fused : fused KF track (world)
  hard  : hard-anchor fill -> KF+RTS (world)   [for reference]
  true  : mocap-truth cup on the track grid (world)
  valid : per-frame detection-present mask
  ph    : per-frame phase index of the mocap-truth segmentation
  drinkerr_{kf,fused,hard} : drinking-phase median position error (mm)
Plus a summary cache/lopo_fused.json.

    python experiments/drink_study/lopo_fused.py
Run from repo root; GPU. ~11 min (23 folds, ~17s each). Per-fold prints (flush).
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import json, sys, time
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
import learn_velfill_fused as FU
from kf_consensus import kf_rts_on_consensus

CACHE = ROOT / "experiments" / "drink_study" / "cache"
TRKDIR = CACHE / "lopo_fused"          # per-rep track cache
DEV, SEQ, HZ = M.DEV, M.SEQ, Q.COMMON_HZ
GATE = 3.0


def fill_kf(base_l, cons_l, valid, vel):
    """Hard-anchor fill, but anchored on the KF/RTS value at the gap edge (user's idea)
    instead of the raw detection cons[s-1]. Denoised anchor = outlier already rejected."""
    out = base_l.copy()
    for s, e in FU.VF.gaps_of(valid):
        p0 = base_l[s - 1] if s - 1 >= 0 else base_l[s]
        path = [p0]
        for t in range(s, e + 1):
            path.append(path[-1] + vel[t])
        out[s:e + 1] = np.array(path[1:])
    return out


def main():
    TRKDIR.mkdir(exist_ok=True)
    print(f"device {DEV}; building sequences...", flush=True)
    reps = [r for rep in T._reps() if (r := M.build_seq(rep)) is not None]
    for r in reps:
        v = np.isfinite(r["cons"]).all(1); r["valid"] = v
        cl = np.zeros((len(r["cons"]), 3)); cl[v] = (r["cons"][v] - r["rest"]) @ r["basis"].T
        r["cons_local_t"] = cl; r["base_local_t"] = (r["base"] - r["rest"]) @ r["basis"].T
        r["vel_tgt"] = np.vstack([np.zeros(3), np.diff(r["tgt"], axis=0)]).astype(np.float32)
        x0 = np.linspace(0, 1, len(v))
        r["gap_seq"] = (np.interp(np.linspace(0, 1, SEQ), x0, v.astype(float)) <= 0.5) & r["tmask"]
    pids = sorted({r["pid"] for r in reps}); fin = reps[0]["feats"].shape[1]
    print(f"{len(reps)} reps, {len(pids)} participants; caching tracks to {TRKDIR}", flush=True)

    err = {"kf": {}, "hard": {}, "hard_kf": {}, "fused": {}}
    t0 = time.time()
    for fi, held in enumerate(pids):
        trn = [r for r in reps if r["pid"] != held]; te = [r for r in reps if r["pid"] == held]
        X = torch.tensor(np.stack([r["feats"] for r in trn])).transpose(1, 2).to(DEV)
        Vt = torch.tensor(np.stack([r["vel_tgt"] for r in trn])).transpose(1, 2).to(DEV)
        G = torch.tensor(np.stack([r["gap_seq"] for r in trn]).astype(np.float32)).to(DEV)
        net = V2.TCNreg(fin, ch=64, layers=6, p=0.1).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4); net.train()
        for ep in range(300):
            opt.zero_grad(); pv = net(X); m = G.unsqueeze(1)
            (((pv - Vt) ** 2).sum(1, keepdim=True) * m).sum().div(m.sum().clamp(min=1)).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            for r in te:
                pv = net(torch.tensor(r["feats"][None]).transpose(1, 2).to(DEV))[0].cpu().numpy().T
                x0 = np.linspace(0, 1, SEQ); x1 = np.linspace(0, 1, r["Tn"])
                vel_t = np.stack([np.interp(x1, x0, pv[:, k]) for k in range(3)], 1)
                vel_world = vel_t @ r["basis"]
                hardfill = FU.fill_noc(r["base_local_t"], r["cons_local_t"], r["valid"], vel_t) @ r["basis"] + r["rest"]
                hardkf = fill_kf(r["base_local_t"], r["cons_local_t"], r["valid"], vel_t) @ r["basis"] + r["rest"]
                tracks = {"kf": r["base"],
                          "hard": kf_rts_on_consensus(hardfill, HZ),
                          "hard_kf": kf_rts_on_consensus(hardkf, HZ),
                          "fused": FU.kf_rts_fused(r["cons"], vel_world, gate=GATE)}
                true = np.stack([np.interp(x1, x0, r["tgt"][:, k]) for k in range(3)], 1) @ r["basis"] + r["rest"]
                des = {}
                for name, trk in tracks.items():
                    de = FU.drinkerr(trk, r)
                    if de is not None:
                        err[name][r["video"]] = de
                    des[name] = de if de is not None else np.nan
                # CACHE tracks + phases + the RAW TCN VELOCITY (rep-local) + rep frame, so ANY
                # future fill/anchor variant is reconstructable with NO GPU / NO retraining.
                np.savez_compressed(
                    TRKDIR / f"{r['video']}.npz",
                    kf=tracks["kf"].astype(np.float32), hard=tracks["hard"].astype(np.float32),
                    hard_kf=tracks["hard_kf"].astype(np.float32),
                    fused=tracks["fused"].astype(np.float32), true=true.astype(np.float32),
                    cons=r["cons"].astype(np.float32), valid=r["valid"],
                    ph=r["ph"].astype(np.int8), tmask=r["tmask"],
                    vel_t=vel_t.astype(np.float32),                 # raw TCN velocity, rep-local (Tn,3)
                    basis=r["basis"].astype(np.float32), rest=r["rest"].astype(np.float32),
                    base_local=r["base_local_t"].astype(np.float32),
                    cons_local=r["cons_local_t"].astype(np.float32),
                    drinkerr_kf=des["kf"], drinkerr_hard=des["hard"],
                    drinkerr_hard_kf=des["hard_kf"], drinkerr_fused=des["fused"],
                    pid=held, video=r["video"])
        el = time.time() - t0
        print(f"  [{fi+1}/{len(pids)}] held {held}: {len(te)} reps cached  "
              f"({el:.0f}s, ~{el/(fi+1)*(len(pids)-fi-1):.0f}s left)", flush=True)

    common = sorted(set.intersection(*[set(err[m_]) for m_ in err]))
    B = np.array([err["kf"][k] for k in common])
    out = {"n": len(common), "gate": GATE, "median": {}, "mean": {}, "beats_kf": {},
           "losses_gt5mm": {}, "cache_dir": str(TRKDIR)}
    print(f"\n=== FILL VARIANTS (LOPO), {len(common)} reps ===")
    print(f"  {'method':<9}{'median':>8}{'mean':>8}{'beats-KF':>10}{'loss>5mm':>10}")
    for m_ in ("kf", "hard", "hard_kf", "fused"):
        a = np.array([err[m_][k] for k in common])
        out["median"][m_] = float(np.median(a)); out["mean"][m_] = float(a.mean())
        out["beats_kf"][m_] = int((a < B).sum()); out["losses_gt5mm"][m_] = int(((a - B) > 5).sum())
        print(f"  {m_:<9}{np.median(a):>7.1f}m{a.mean():>7.1f}m{(a<B).sum():>9d}{((a-B)>5).sum():>10d}"
              f"   ({(np.median(a)/np.median(B)-1)*100:+.0f}%)")
    json.dump(out, open(CACHE / "lopo_fused.json", "w"), indent=2)
    print(f"\nwrote cache/lopo_fused.json  +  {len(common)}+ per-rep tracks in {TRKDIR}/", flush=True)


if __name__ == "__main__":
    main()
