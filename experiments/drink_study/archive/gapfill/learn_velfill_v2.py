"""Improved velocity-fill: augmentation + regularization + integrated-position loss.

Diagnosis: the v1 velocity model OVERFITS (gap-velocity test/train error 2.4x). So we
attack generalization, not capacity:
  1. SYNTHETIC-GAP AUGMENTATION — randomly mask extra visible frames each epoch to
     create many more 'fill this gap' examples; + left/right mirror; + consensus jitter.
  2. REGULARIZATION — dropout between conv layers + higher weight decay.
  3. INTEGRATED-POSITION LOSS — integrate predicted velocity across each gap and
     penalise the accumulated POSITION drift vs mocap (not just per-frame velocity),
     which directly targets the long-gap overshoot.

LOPO eval vs the v1 -11% baseline. OMC only for training/scoring.

    python learn_velfill_v2.py
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
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "drink_study"))
import qtm_align as Q
import segment_cup_only as S
import tune_interp as T
import learn_velocity_fill as VF
import learn_seq_kf as M
from kf_consensus import kf_rts_on_consensus

CACHE = ROOT / "experiments" / "drink_study" / "cache"
DEV = M.DEV
SEQ = M.SEQ


class TCNreg(nn.Module):
    """Dilated TCN with dropout, + an optional GLOBAL-CONTEXT block that looks at the
    WHOLE sequence at once: pool over time -> Linear -> broadcast back to every frame.
    This gives each frame a direct summary of the entire rep (both gap ends, overall
    shape), which the local dilated convs build up only indirectly."""
    def __init__(self, fin, nout=3, ch=64, layers=6, p=0.2, global_ctx=False,
                 attn=False, heads=4):
        super().__init__()
        blocks = []; c = fin
        for i in range(layers):
            d = 2 ** i
            blocks += [nn.Conv1d(c, ch, 5, padding=2 * d, dilation=d), nn.ReLU(),
                       nn.BatchNorm1d(ch), nn.Dropout(p)]
            c = ch
        self.body = nn.Sequential(*blocks)
        self.global_ctx = global_ctx
        if global_ctx:
            self.gfc = nn.Sequential(nn.Linear(2 * ch, ch), nn.ReLU(), nn.Linear(ch, ch))
        self.attn = attn
        if attn:
            # one lightweight self-attention layer: every frame attends to every other.
            # learned positional embedding so it knows frame ORDER (attention is order-blind).
            self.pos = nn.Parameter(torch.randn(1, 512, ch) * 0.02)   # >= SEQ
            self.mha = nn.MultiheadAttention(ch, heads, dropout=p, batch_first=True)
            self.ln = nn.LayerNorm(ch)
        self.head = nn.Conv1d(ch, nout, 1)

    def forward(self, x):
        h = self.body(x)                                  # (B,ch,T)
        if self.global_ctx:
            g = torch.cat([h.mean(2), h.amax(2)], dim=1)
            h = h + self.gfc(g).unsqueeze(2)
        if self.attn:
            z = h.transpose(1, 2)                         # (B,T,ch) for attention
            z = z + self.pos[:, :z.shape[1]]              # add positional embedding
            a, _ = self.mha(z, z, z)                      # every frame attends to every frame
            z = self.ln(z + a)                            # residual + norm
            h = z.transpose(1, 2)                         # back to (B,ch,T)
        return self.head(h)


# ---- feature column indices in build_seq's 17-feature layout ----
# [0:3]=cons_local, [3]=cons_valid, [4:14]=kept(10), [14]=mpx, [15]=ncams, [16]=occ
CONS = slice(0, 3); CVALID = 3; OCC = 16; NCAMS = 15


def augment(feats, vel, gap, rng):
    """Per-sample augmentation (numpy): synthetic extra gaps + mirror + jitter.
    feats:(SEQ,17) vel:(SEQ,3) gap:(SEQ,) bool. Returns augmented copies + new gap mask."""
    f = feats.copy(); v = vel.copy(); g = gap.copy()
    # 1) synthetic gaps: blank some currently-visible runs (set cons=0, valid=0, occ=1)
    vis = f[:, CVALID] > 0.5
    if vis.sum() > 40 and rng.random() < 0.8:
        n_new = rng.integers(1, 4)
        for _ in range(n_new):
            L = int(rng.integers(4, 20))
            start = int(rng.integers(0, max(1, SEQ - L)))
            sl = slice(start, start + L)
            f[sl, CONS] = 0.0; f[sl, CVALID] = 0.0; f[sl, OCC] = 1.0
            f[sl, NCAMS] = 0.0
            g[sl] = True                          # now a supervised gap
    # 2) consensus jitter on visible frames (robustness)
    visnow = f[:, CVALID] > 0.5
    f[visnow, CONS] += rng.normal(0, 3.0, size=(visnow.sum(), 3)).astype(np.float32)
    # 3) lateral mirror (flip axis-1 of the rep-local frame = left/right symmetry)
    if rng.random() < 0.5:
        f[:, 1] *= -1; v[:, 1] *= -1
        f[:, 1 + 0] = f[:, 1 + 0]  # noop, axis2 is index1 in CONS? cons axes are 0,1,2
    return f, v, g


def integ_pos_loss(pv, vel_true, valid_seq, gap, basis_unused=None):
    """ENDPOINT loss, fully vectorised (no Python loop): integrate predicted velocities
    across EACH gap and compare where that LANDS to the true net displacement. Uses the
    per-gap cumulative velocity-error: drift[t] = csum[t] - csum[gap_anchor[t]] is the
    accumulated landing error SINCE the gap started; at the LAST frame of each gap that
    drift IS the endpoint error. So we penalise drift only at gap-end frames. pv,(B,3,SEQ)."""
    verr = pv - vel_true                                       # (B,3,SEQ)
    csum = torch.cumsum(verr, dim=2)
    B, C, Tn = csum.shape
    g = (gap > 0.5)                                            # (B,SEQ)
    pos = torch.arange(Tn, device=csum.device).view(1, -1)
    nongap_idx = torch.where(~g, pos.expand(B, -1), torch.full_like(pos.expand(B, -1), -1))
    anchor = torch.cummax(nongap_idx, dim=1)[0].clamp(min=0)   # last visible idx ≤ t
    a = anchor.unsqueeze(1).expand(B, C, Tn)
    drift = csum - torch.gather(csum, 2, a)                    # landing error since gap entry
    # gap-END frames: a gap frame whose NEXT frame is not a gap (or sequence end)
    nxt = torch.zeros_like(g); nxt[:, :-1] = g[:, 1:]
    gap_end = g & (~nxt)                                       # (B,SEQ) bool
    m = gap_end.unsqueeze(1).float()
    se = (drift ** 2).sum(1, keepdim=True)
    return (se * m).sum() / m.sum().clamp(min=1)


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
    pids = sorted({r["pid"] for r in reps}); fin = reps[0]["feats"].shape[1]
    rng = np.random.default_rng(0)
    print(f"{len(reps)} reps", flush=True)

    b_de, s_de, common = {}, {}, []
    for held in pids:
        trn = [r for r in reps if r["pid"] != held]; te = [r for r in reps if r["pid"] == held]
        feats0 = np.stack([r["feats"] for r in trn])
        vel0 = np.stack([r["vel_tgt"] for r in trn])
        gap0 = np.stack([r["gap_seq"] for r in trn])
        net = TCNreg(fin).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=3e-4)
        net.train()
        for ep in range(320):
            # fresh augmentation each epoch
            fa = np.empty_like(feats0); va = np.empty_like(vel0); ga = np.empty_like(gap0)
            for i in range(len(trn)):
                fa[i], va[i], ga[i] = augment(feats0[i], vel0[i], gap0[i], rng)
            X = torch.tensor(fa).transpose(1, 2).to(DEV)
            V = torch.tensor(va).transpose(1, 2).to(DEV)
            G = torch.tensor(ga.astype(np.float32)).to(DEV)
            opt.zero_grad()
            pv = net(X); m = G.unsqueeze(1)
            vel_loss = (((pv - V) ** 2).sum(1, keepdim=True) * m).sum() / m.sum().clamp(min=1)
            pos_loss = integ_pos_loss(pv, V, G, G)
            (vel_loss + 0.1 * pos_loss).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            for r in te:
                pv = net(torch.tensor(r["feats"][None]).transpose(1, 2).to(DEV))[0].cpu().numpy().T
                x0 = np.linspace(0, 1, SEQ); x1 = np.linspace(0, 1, r["Tn"])
                vel_t = np.stack([np.interp(x1, x0, pv[:, k]) for k in range(3)], 1)
                tcn = VF.fill_track(r["base_local_t"], r["cons_local_t"], r["valid"], vel_t)
                tcn_w = tcn @ r["basis"] + r["rest"]
                for trk, store in [(r["base"], b_de), (tcn_w, s_de)]:
                    sc = T._score(trk, r["mr"])
                    if sc is None:
                        continue
                    _, err, mfi = sc; mfi = np.clip(mfi, 0, len(r["ph"]) - 1)
                    dm = r["ph"][mfi] == S.P_DRINK
                    if dm.any():
                        store[r["video"]] = float(np.median(err[dm]))
                if r["video"] in b_de and r["video"] in s_de:
                    common.append(r["video"])
        print(f"  held {held}: {len(te)} reps", flush=True)

    common = [k for k in common if k in b_de and k in s_de]
    B = np.array([b_de[k] for k in common]); SC = np.array([s_de[k] for k in common])
    print(f"\n=== VELFILL v2 (aug+reg+intloss, LOPO), {len(common)} reps ===")
    print(f"  baseline KF : {np.median(B):.1f}mm   p90 {np.percentile(B,90):.1f}")
    print(f"  velfill v2  : {np.median(SC):.1f}mm   p90 {np.percentile(SC,90):.1f}  "
          f"({(np.median(SC)/np.median(B)-1)*100:+.0f}%)")
    print(f"  improved {(SC<B).sum()}, worsened {(SC>B).sum()}  (v1 was -11%, 291/225)")
    json.dump({"n": len(common), "base": float(np.median(B)), "v2": float(np.median(SC)),
               "base_p90": float(np.percentile(B, 90)), "v2_p90": float(np.percentile(SC, 90)),
               "improved": int((SC < B).sum()), "worsened": int((SC > B).sum())},
              open(CACHE / "learn_velfill_v2.json", "w"), indent=2)
    print("wrote cache/learn_velfill_v2.json")


if __name__ == "__main__":
    main()
