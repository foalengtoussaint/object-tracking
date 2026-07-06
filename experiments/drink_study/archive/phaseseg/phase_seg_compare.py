"""Does the gap-fill method change PHASE SEGMENTATION? (LOPO, vs mocap-truth phases)

Question (user): how do the segmented phases differ between
  (1) KF+RTS track          — the current pipeline
  (2) endpoint LINEAR fill   — chord across each gap
  (3) learned TCN velocity-fill
when each is segmented with segment_cup_only and compared to the phases obtained
from the MOCAP-truth cup track (the ground-truth segmentation).

We report two things per track, both vs mocap-truth:
  A. BOUNDARY timing error  — mean |Δframe| over the phase transitions, in ms.
  B. DRINKING-phase DURATION error — |dur_track - dur_truth| in ms (what the
     segmentation is actually FOR clinically), + signed bias.

LOPO (not a single split) because sub-split differences this session have
repeatedly been noise. ~15 s/fold, ~11 min total. Per-rep cached so a re-run is free.

    python experiments/drink_study/phase_seg_compare.py
Run from repo root; GPU.
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
import learn_velocity_fill as VF

CACHE = ROOT / "experiments" / "drink_study" / "cache"
DEV = M.DEV
SEQ = M.SEQ
HZ = Q.COMMON_HZ


def fill_noc(base_l, cons_l, valid, vel):
    """Entry-anchor-only velocity integration (the -13% recipe, NO exit correction)."""
    out = base_l.copy()
    for s, e in VF.gaps_of(valid):
        p0 = cons_l[s - 1] if s - 1 >= 0 and valid[s - 1] else base_l[s]
        path = [p0]
        for t in range(s, e + 1):
            path.append(path[-1] + vel[t])
        out[s:e + 1] = np.array(path[1:])
    return out


def lin_fill(base_l, cons_l, valid):
    """Endpoint-anchored straight chord across each gap."""
    out = base_l.copy()
    for s, e in VF.gaps_of(valid):
        p0 = cons_l[s - 1] if s - 1 >= 0 and valid[s - 1] else base_l[s]
        p1 = cons_l[e + 1] if e + 1 < len(valid) and valid[e + 1] else base_l[e]
        for k, t in enumerate(range(s, e + 1)):
            w = (k + 1) / (e - s + 2)
            out[t] = (1 - w) * p0 + w * p1
    return out


def phase_info(xyz):
    """Phase array + boundary-frame dict + drinking-phase duration in frames."""
    ph = S.segment_cup_only(xyz, fps=HZ)["phase"]
    b = {}
    for k in range(1, len(ph)):
        if ph[k] != ph[k - 1]:
            b[(int(ph[k - 1]), int(ph[k]))] = k
    drink = int((ph == S.P_DRINK).sum())
    return ph, b, drink


def main():
    print(f"device {DEV}; building sequences...", flush=True)
    reps = [r for rep in T._reps() if (r := M.build_seq(rep)) is not None]
    for r in reps:
        valid = np.isfinite(r["cons"]).all(1); r["valid"] = valid
        cl = np.zeros((len(r["cons"]), 3)); cl[valid] = (r["cons"][valid] - r["rest"]) @ r["basis"].T
        r["cons_local_t"] = cl; r["base_local_t"] = (r["base"] - r["rest"]) @ r["basis"].T
        r["vel_tgt"] = np.vstack([np.zeros(3), np.diff(r["tgt"], axis=0)]).astype(np.float32)
        x0 = np.linspace(0, 1, len(valid)); x1 = np.linspace(0, 1, SEQ)
        r["gap_seq"] = (np.interp(x1, x0, valid.astype(float)) <= 0.5) & r["tmask"]
    pids = sorted({r["pid"] for r in reps}); fin = reps[0]["feats"].shape[1]
    print(f"{len(reps)} reps, {len(pids)} participants", flush=True)

    # accumulators: per-rep, keyed by video so KF/lin/tcn line up on the SAME reps
    bnd = {"kf": {}, "lin": {}, "tcn": {}}      # boundary timing err (ms)
    ddur = {"kf": {}, "lin": {}, "tcn": {}}     # signed drinking-duration err (ms)
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
            loss = (((pv - Vt) ** 2).sum(1, keepdim=True) * m).sum() / m.sum().clamp(min=1)
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            for r in te:
                if not (~r["valid"]).any():        # no gap -> all three tracks identical, skip
                    continue
                pv = net(torch.tensor(r["feats"][None]).transpose(1, 2).to(DEV))[0].cpu().numpy().T
                x0 = np.linspace(0, 1, SEQ); x1 = np.linspace(0, 1, r["Tn"])
                vel_t = np.stack([np.interp(x1, x0, pv[:, k]) for k in range(3)], 1)
                tcn = fill_noc(r["base_local_t"], r["cons_local_t"], r["valid"], vel_t) @ r["basis"] + r["rest"]
                lin = lin_fill(r["base_local_t"], r["cons_local_t"], r["valid"]) @ r["basis"] + r["rest"]
                kf = r["base"]
                true = np.stack([np.interp(x1, x0, r["tgt"][:, k]) for k in range(3)], 1) @ r["basis"] + r["rest"]
                _, tb, tdr = phase_info(true)
                for name, trk in [("kf", kf), ("lin", lin), ("tcn", tcn)]:
                    _, pb, pdr = phase_info(trk)
                    errs = [abs(pb[k] - tb[k]) for k in tb if k in pb]
                    if errs:
                        bnd[name][r["video"]] = float(np.mean(errs) / HZ * 1000)
                    ddur[name][r["video"]] = float((pdr - tdr) / HZ * 1000)   # signed
        el = time.time() - t0
        print(f"  [{fi+1}/{len(pids)}] held {held}: {len(te)} reps   "
              f"({el:.0f}s, ~{el/(fi+1)*(len(pids)-fi-1):.0f}s left)", flush=True)

    # compare on the COMMON rep set
    common_b = sorted(set(bnd["kf"]) & set(bnd["lin"]) & set(bnd["tcn"]))
    common_d = sorted(set(ddur["kf"]) & set(ddur["lin"]) & set(ddur["tcn"]))
    out = {"n_boundary": len(common_b), "n_duration": len(common_d), "boundary_ms": {}, "drink_dur_err_ms": {}}
    print(f"\n=== PHASE BOUNDARY timing error vs mocap-truth ({len(common_b)} reps with gaps) ===")
    print(f"  {'track':<12}{'median':>8}{'mean':>8}{'p90':>8}")
    for name in ("kf", "lin", "tcn"):
        a = np.array([bnd[name][k] for k in common_b])
        out["boundary_ms"][name] = {"median": float(np.median(a)), "mean": float(a.mean()),
                                    "p90": float(np.percentile(a, 90))}
        print(f"  {name:<12}{np.median(a):>7.0f}m{a.mean():>7.0f}m{np.percentile(a,90):>7.0f}m")
    print(f"\n=== DRINKING-phase DURATION error vs mocap-truth ({len(common_d)} reps) ===")
    print(f"  {'track':<12}{'|err| med':>10}{'|err| mean':>11}{'bias':>8}")
    for name in ("kf", "lin", "tcn"):
        a = np.array([ddur[name][k] for k in common_d])
        out["drink_dur_err_ms"][name] = {"abs_median": float(np.median(np.abs(a))),
                                         "abs_mean": float(np.abs(a).mean()), "bias": float(a.mean())}
        print(f"  {name:<12}{np.median(np.abs(a)):>9.0f}m{np.abs(a).mean():>10.0f}m{a.mean():>+7.0f}m")
    # head-to-head: TCN vs KF on matched reps (drinking-duration |err|)
    ak = np.abs([ddur["kf"][k] for k in common_d]); at = np.abs([ddur["tcn"][k] for k in common_d])
    out["tcn_vs_kf_dur"] = {"tcn_better": int((at < ak).sum()), "tcn_worse": int((at > ak).sum()),
                            "tie": int((at == ak).sum())}
    print(f"\n  drink-duration |err|: TCN better on {(at<ak).sum()}, worse {(at>ak).sum()}, "
          f"tie {(at==ak).sum()} of {len(common_d)}")
    json.dump(out, open(CACHE / "phase_seg_compare.json", "w"), indent=2)
    print("\nwrote cache/phase_seg_compare.json", flush=True)


if __name__ == "__main__":
    main()
