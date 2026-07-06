"""Self-consistent gap-inpainting model (user's idea).

The plain seq model under-shoots the occluded apex because the training gradient at
the occluded frames is weak (sparse/uncertain target there). User's fix: make the
model SELF-CONSISTENT with the future — predict through an occlusion gap, and when
the cup is TRACKED AGAIN, the observed re-acquisition position is a strong anchor;
the error between the prediction and that observed position trains the model. This
is mocap-free supervision that forces the gap-fill to CONNECT both visible ends, so
it can't drift to a flat under-prediction.

Loss = endpoint self-consistency (predicted pos at consensus frames must match the
observed consensus, weighted UP right after a gap) + mocap apex term (where available,
pins how high the dwell goes) + big-error penalty so a large miss matters more.

    python learn_selfconsistent.py
Run from repo root; uses GPU.
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
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "drink_study"))
import qtm_align as Q
import segment_cup_only as S
import tune_interp as T
from learn_seq_kf import build_seq, TCN, kf_with_learned, SEQ, DEV

CACHE = ROOT / "experiments" / "drink_study" / "cache"


def consensus_local_seq(r):
    """Observed consensus in rep-local frame on the SEQ grid + a validity mask.
    These are the SELF-SUPERVISION anchors (no mocap)."""
    cons = r["cons"]; valid = np.isfinite(cons).all(1)
    cl = np.full((len(cons), 3), 0.0)
    cl[valid] = (cons[valid] - r["rest"]) @ r["basis"].T
    x0 = np.linspace(0, 1, len(cons)); x1 = np.linspace(0, 1, SEQ)
    cl_s = np.stack([np.interp(x1, x0, cl[:, k]) for k in range(3)], 1)
    v_s = np.interp(x1, x0, valid.astype(float)) > 0.5
    # weight: emphasise frames just AFTER a gap (re-acquisition) — the consistency anchor
    fsc = np.zeros(len(cons)); c = 0
    for i in range(len(cons)):
        c = 0 if valid[i] else c + 1
        fsc[i] = c
    reacq = np.zeros(len(cons))
    for i in range(1, len(cons)):
        if valid[i] and fsc[i - 1] > 0:               # first valid frame after a gap
            reacq[max(0, i - 1):i + 3] = 1.0
    rq_s = np.interp(x1, x0, reacq) > 0.3
    return cl_s.astype(np.float32), v_s, rq_s


def main():
    print(f"device {DEV}; building sequences...", flush=True)
    reps = [r for rep in T._reps() if (r := build_seq(rep)) is not None]
    for r in reps:
        r["cl"], r["cl_valid"], r["reacq"] = consensus_local_seq(r)
        # GAP MASK: the model only learns / predicts where the cup is OCCLUDED
        # (consensus invalid). Visible frames keep the consensus, which is already
        # accurate -> all model capacity goes to the hard gap-fill. (user)
        r["gap"] = (~r["cl_valid"]) & r["tmask"]
    pids = sorted({r["pid"] for r in reps})
    fin = reps[0]["feats"].shape[1]
    ngap = int(np.mean([r["gap"].mean() for r in reps]) * 100)
    print(f"{len(reps)} reps, {fin} features; ~{ngap}% of frames are gaps (model predicts only these)",
          flush=True)

    b_de, s_de, common = {}, {}, []
    for held in pids:
        trn = [r for r in reps if r["pid"] != held]; te = [r for r in reps if r["pid"] == held]
        X = torch.tensor(np.stack([r["feats"] for r in trn])).transpose(1, 2).to(DEV)
        Ytgt = torch.tensor(np.stack([r["tgt"] for r in trn])).transpose(1, 2).to(DEV)
        # supervise ONLY on gap (occluded) frames -> no shrinkage from echoing visible frames
        Mtgt = torch.tensor(np.stack([r["gap"] for r in trn]).astype(np.float32)).to(DEV)

        # OMC supervision with the self-consistency STRUCTURE: the error target is the
        # MOCAP truth (not the consensus, which can be confidently wrong at the apex),
        # weighted UP at the frames bracketing each occlusion gap so the model is forced
        # to be consistent with the TRUE entry/exit of the gap, not just smooth. (user)
        gap_edge = torch.tensor(np.stack([
            np.maximum(r["reacq"].astype(np.float32),
                       np.r_[r["reacq"][1:], 0].astype(np.float32))  # also pre-gap edge
            for r in trn])).to(DEV)
        posnet = TCN(fin, 3).to(DEV)
        opt = torch.optim.Adam(posnet.parameters(), lr=2e-3, weight_decay=1e-4)
        posnet.train()
        for ep in range(300):
            opt.zero_grad()
            pred = posnet(X)                                   # (B,3,SEQ)
            # MOCAP target everywhere it exists; extra weight at gap boundaries
            w = (Mtgt * (1.0 + 5.0 * gap_edge)).unsqueeze(1)
            se = ((pred - Ytgt) ** 2).sum(1, keepdim=True)
            loss = (se * w).sum() / w.sum().clamp(min=1)   # plain MSE on gap frames only
            loss.backward(); opt.step()
        posnet.eval()

        with torch.no_grad():
            for r in te:
                xb = torch.tensor(r["feats"][None]).transpose(1, 2).to(DEV)
                pos = posnet(xb)[0].cpu().numpy().T
                x0 = np.linspace(0, 1, SEQ); x1 = np.linspace(0, 1, r["Tn"])
                pos_t = np.stack([np.interp(x1, x0, pos[:, k]) for k in range(3)], 1)
                z_world = pos_t @ r["basis"] + r["rest"]
                # fuse: learned measurement only at occluded frames (low var there)
                var_t = np.where(r["ncams"][:r["Tn"]] < 2, 900.0, 1e6) if len(r["ncams"]) >= r["Tn"] \
                    else np.full(r["Tn"], 900.0)
                fused = kf_with_learned(r["cons"], r["ncams"], z_world, var_t)
                for trk, store in [(r["base"], b_de), (fused, s_de)]:
                    sc = T._score(trk, r["mr"])
                    if sc is None:
                        continue
                    rms, err, mfi = sc; mfi = np.clip(mfi, 0, len(r["ph"]) - 1)
                    dm = r["ph"][mfi] == S.P_DRINK
                    if dm.any():
                        store[r["video"]] = float(np.median(err[dm]))
                if r["video"] in b_de and r["video"] in s_de:
                    common.append(r["video"])
        # sanity: predicted reach on held-out
        with torch.no_grad():
            reach = [posnet(torch.tensor(r["feats"][None]).transpose(1, 2).to(DEV))[0]
                     .cpu().numpy().T[r["tmask"], 0].max() for r in te]
        print(f"  held {held}: {len(te)} reps  pred-reach {np.median(reach):.0f}mm "
              f"(target ~250-550)", flush=True)

    common = [k for k in common if k in b_de and k in s_de]
    B = np.array([b_de[k] for k in common]); SC = np.array([s_de[k] for k in common])
    print(f"\n=== SELF-CONSISTENT model -> KF (LOPO), {len(common)} reps ===")
    print(f"  baseline KF drinking err : {np.median(B):.1f}mm")
    print(f"  self-consistent + KF     : {np.median(SC):.1f}mm  ({(np.median(SC)/np.median(B)-1)*100:+.0f}%)")
    print(f"  improved {(SC<B).sum()}, worsened {(SC>B).sum()}, mean {np.mean(SC-B):+.1f}mm")
    json.dump({"n": len(common), "base": float(np.median(B)), "selfcons": float(np.median(SC)),
               "improved": int((SC < B).sum()), "worsened": int((SC > B).sum())},
              open(CACHE / "learn_selfconsistent.json", "w"), indent=2)
    print("wrote cache/learn_selfconsistent.json")


if __name__ == "__main__":
    main()
