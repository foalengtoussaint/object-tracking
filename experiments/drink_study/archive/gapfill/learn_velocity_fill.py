"""Learned-VELOCITY gap fill with self-consistent boundary correction (user's idea).

Don't predict absolute position in the gap (that floats / shrinks). Predict the
MOVEMENT (per-frame velocity, rep-local) and INTEGRATE from the last visible
position. Then 'rerun until it reaches the next consensus': the integrated path is
corrected so it lands exactly on the gap-EXIT consensus (endpoint mismatch
redistributed across the gap). Result is anchored at the gap entry AND exit, smooth
by construction, and reaches the true apex because it continues the real motion.

Velocity target = mocap per-frame displacement. LOPO. OMC only for training/scoring;
test input is consensus/cameras/occlusion + the model's own integration.

    python learn_velocity_fill.py
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
import qtm_align as Q
import segment_cup_only as S
import tune_interp as T
from kf_consensus import kf_rts_on_consensus
from learn_seq_kf import build_seq, TCN, kf_with_learned, SEQ, DEV

CACHE = ROOT / "experiments" / "drink_study" / "cache"


def gaps_of(valid):
    """List of (start, end) inclusive index ranges where valid is False (gaps)."""
    out = []; i = 0; n = len(valid)
    while i < n:
        if not valid[i]:
            j = i
            while j < n and not valid[j]:
                j += 1
            out.append((i, j - 1)); i = j
        else:
            i += 1
    return out


def fill_track(base_local, cons_local, valid, vel_pred):
    """Fill each occlusion gap by integrating the predicted velocity from the last
    visible position, then correcting so the path LANDS on the next visible position
    (self-consistent endpoints). Returns a (T,3) rep-local filled track."""
    out = base_local.copy()
    for s, e in gaps_of(valid):
        p0 = cons_local[s - 1] if s - 1 >= 0 and valid[s - 1] else base_local[s]
        # integrate predicted velocity across the gap from p0
        path = [p0]
        for t in range(s, e + 1):
            path.append(path[-1] + vel_pred[t])
        path = np.array(path[1:])                 # (gap_len,3)
        # endpoint anchor: where the cup is actually seen again
        if e + 1 < len(valid) and valid[e + 1]:
            p_exit_true = cons_local[e + 1]
            p_exit_pred = path[-1] + vel_pred[min(e + 1, len(vel_pred) - 1)]  # one more step
            drift = p_exit_true - p_exit_pred
            # redistribute drift linearly across the gap so it connects to both ends
            w = np.linspace(0, 1, len(path) + 1)[1:][:, None]
            path = path + w * drift
        out[s:e + 1] = path
    return out


def main():
    print(f"device {DEV}; building sequences...", flush=True)
    reps = [r for rep in T._reps() if (r := build_seq(rep)) is not None]
    for r in reps:
        valid = np.isfinite(r["cons"]).all(1)
        r["valid"] = valid
        rest = r["rest"]; basis = r["basis"]
        # rep-local consensus (visible) and base/kf-local, on the TRACK grid (Tn)
        cl = np.zeros((len(r["cons"]), 3))
        cl[valid] = (r["cons"][valid] - rest) @ basis.T
        r["cons_local_t"] = cl
        r["base_local_t"] = (r["base"] - rest) @ basis.T
        # velocity TARGET on the SEQ grid: mocap per-frame displacement (rep-local)
        tgt = r["tgt"]                            # (SEQ,3) true cup pos, rep-local
        vel = np.vstack([np.zeros(3), np.diff(tgt, axis=0)]).astype(np.float32)
        r["vel_tgt"] = vel
        # gap mask on SEQ grid (occluded & has a target)
        x0 = np.linspace(0, 1, len(valid)); x1 = np.linspace(0, 1, SEQ)
        v_seq = np.interp(x1, x0, valid.astype(float)) > 0.5
        r["gap_seq"] = (~v_seq) & r["tmask"]
    pids = sorted({r["pid"] for r in reps})
    fin = reps[0]["feats"].shape[1]
    print(f"{len(reps)} reps, {fin} features", flush=True)

    b_de, s_de, common = {}, {}, []
    for held in pids:
        trn = [r for r in reps if r["pid"] != held]; te = [r for r in reps if r["pid"] == held]
        X = torch.tensor(np.stack([r["feats"] for r in trn])).transpose(1, 2).to(DEV)
        V = torch.tensor(np.stack([r["vel_tgt"] for r in trn])).transpose(1, 2).to(DEV)
        # supervise velocity on GAP frames (+ a little on the frames around them)
        Gm = torch.tensor(np.stack([r["gap_seq"] for r in trn]).astype(np.float32)).to(DEV)
        net = TCN(fin, 3).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
        net.train()
        for ep in range(300):
            opt.zero_grad()
            pv = net(X)
            m = Gm.unsqueeze(1)
            se = ((pv - V) ** 2).sum(1, keepdim=True)
            loss = (se * m).sum() / m.sum().clamp(min=1)
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            for r in te:
                xb = torch.tensor(r["feats"][None]).transpose(1, 2).to(DEV)
                pv = net(xb)[0].cpu().numpy().T          # (SEQ,3) velocity, rep-local
                # SEQ -> Tn
                x0 = np.linspace(0, 1, SEQ); x1 = np.linspace(0, 1, r["Tn"])
                vel_t = np.stack([np.interp(x1, x0, pv[:, k]) for k in range(3)], 1)
                filled_local = fill_track(r["base_local_t"], r["cons_local_t"], r["valid"], vel_t)
                track_world = filled_local @ r["basis"] + r["rest"]
                # the filled track IS the answer on gaps; keep consensus elsewhere already.
                for trk, store in [(r["base"], b_de), (track_world, s_de)]:
                    sc = T._score(trk, r["mr"])
                    if sc is None:
                        continue
                    rms, err, mfi = sc; mfi = np.clip(mfi, 0, len(r["ph"]) - 1)
                    dm = r["ph"][mfi] == S.P_DRINK
                    if dm.any():
                        store[r["video"]] = float(np.median(err[dm]))
                if r["video"] in b_de and r["video"] in s_de:
                    common.append(r["video"])
        # sanity: apex reach in the filled track on gap reps
        with torch.no_grad():
            rch = []
            for r in te:
                pv = net(torch.tensor(r["feats"][None]).transpose(1, 2).to(DEV))[0].cpu().numpy().T
                x0 = np.linspace(0, 1, SEQ); x1 = np.linspace(0, 1, r["Tn"])
                vel_t = np.stack([np.interp(x1, x0, pv[:, k]) for k in range(3)], 1)
                fl = fill_track(r["base_local_t"], r["cons_local_t"], r["valid"], vel_t)
                if r["valid"].mean() < 1.0:
                    rch.append(fl[:, 0].max())
        print(f"  held {held}: {len(te)} reps  filled-apex {np.median(rch):.0f}mm (true ~500)", flush=True)

    common = [k for k in common if k in b_de and k in s_de]
    B = np.array([b_de[k] for k in common]); SC = np.array([s_de[k] for k in common])
    print(f"\n=== VELOCITY-FILL (self-consistent endpoints, LOPO), {len(common)} reps ===")
    print(f"  baseline KF drinking err : {np.median(B):.1f}mm")
    print(f"  velocity-fill            : {np.median(SC):.1f}mm  ({(np.median(SC)/np.median(B)-1)*100:+.0f}%)")
    print(f"  improved {(SC<B).sum()}, worsened {(SC>B).sum()}, mean {np.mean(SC-B):+.1f}mm")
    json.dump({"n": len(common), "base": float(np.median(B)), "velfill": float(np.median(SC)),
               "improved": int((SC < B).sum()), "worsened": int((SC > B).sum())},
              open(CACHE / "learn_velocity_fill.json", "w"), indent=2)
    print("wrote cache/learn_velocity_fill.json")


if __name__ == "__main__":
    main()
