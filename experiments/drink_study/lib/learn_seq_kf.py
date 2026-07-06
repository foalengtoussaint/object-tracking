"""Separate sequence model -> learned MEASUREMENT into the KF (user's final idea).

Architecture: a TCN sees the WHOLE rep's tracking sequence and outputs, per frame,
a predicted cup position AND its own uncertainty (heteroscedastic). That prediction
enters the KF as a measurement with that uncertainty, ALONGSIDE the real consensus:

    consensus ----------------\
                               KF / RTS -> final track
    TCN -> (z_learned, R_learned) /

The KF's gain fuses them: where consensus is good it dominates; at the occluded apex
(no consensus) but the TCN is confident, the learned measurement carries the state;
where the TCN is unsure (clean frames) its large R_learned makes the KF ignore it ->
no 'coarse on clean frames' damage. Trained Gaussian-NLL vs mocap, LOPO.

    python learn_seq_kf.py
Run from repo root. Uses GPU if available.
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "drink_study"))
import qtm_align as Q
import segment_cup_only as S
import tune_interp as T
from kf_consensus import kf_rts_on_consensus
from learn_correction import _rep_frame
import torch
import torch.nn as nn

from _paths import CACHE
TRACKDIR = CACHE / "track3d_clean3d_refill"
CAMS = [f"cam_{i}" for i in range(1, 11)]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEQ = 360                       # fixed sequence length (resample); ~6s @60Hz covers a rep


def build_seq(rep):
    """One rep: input sequence feats (T,F), target cup pos (T,3) rep-local, occ mask,
    plus the bits needed to map back and KF-fuse. Resampled to SEQ length."""
    vpath = TRACKDIR / (rep["video"] + ".json")
    if not vpath.exists():
        return None
    tr = Q.load_trial(rep["c3d"])
    if not tr.gt_quality()["ok"]:
        return None
    d = json.loads(vpath.read_text())["frames"]
    cons = np.array([f["consensus"] if f.get("consensus") else [np.nan] * 3 for f in d], float)
    kept = np.array([[1.0 if c in set(f.get("kept", [])) else 0.0 for c in CAMS] for f in d])
    mpx = np.array([f.get("median_px") if f.get("median_px") is not None else 30.0 for f in d], float)
    ncams = kept.sum(1)
    if np.isfinite(cons).all(1).sum() < 20:
        return None
    base = kf_rts_on_consensus(cons)
    mr = Q._resample(tr.centroid(), tr.rate)
    vr = Q._resample(base, Q.VIDEO_FPS)
    lag, corr = Q._xcorr_lag(Q._speed(vr, Q.COMMON_HZ), Q._speed(mr, Q.COMMON_HZ))
    if corr < Q.MIN_SYNC_CORR:
        return None
    # build a per-frame mocap-in-KF-frame target by syncing (lag) then nearest map
    if lag >= 0:
        v = vr[lag:]; mo = mr[:len(v)]; mfi = np.arange(len(v))
    else:
        mo = mr[-lag:]; v = vr[:len(mo)]; mfi = np.arange(-lag, -lag + len(mo))
    Ln = min(len(v), len(mo)); v, mo, mfi = v[:Ln], mo[:Ln], mfi[:Ln]
    ok = ~(np.isnan(v).any(1) | np.isnan(mo).any(1))
    if ok.sum() < 10:
        return None
    R, t, _ = Q.kabsch(mo[ok], v[ok], robust=True)
    rest = np.median(base[:30], 0); basis = _rep_frame(base, rest)
    Tn = len(d)
    # target cup (rep-local) at every track frame: map mocap->frame via mfi (else NaN)
    tgt = np.full((Tn, 3), np.nan)
    moW = (mr @ R.T + t)                       # mocap in KF frame, on mocap grid
    # align mocap grid to track frames through the same lag
    for i, fi in enumerate(np.clip(mfi, 0, Tn - 1)):
        mi = (fi - lag) if lag >= 0 else (fi + (-lag))
        if 0 <= mi < len(moW):
            tgt[fi] = (moW[mi] - rest) @ basis.T
    consL = np.where(np.isfinite(cons).all(1)[:, None], (cons - rest) @ basis.T, np.nan)
    occ = (ncams < 2).astype(float)
    feats = np.concatenate([np.nan_to_num(consL), np.isfinite(consL[:, :1]).astype(float),
                            kept, mpx[:, None] / 30.0, ncams[:, None] / 10.0, occ[:, None]], axis=1)

    # anchors for a shape-prior feature (added per-fold with the LOPO shape):
    # rep's own rest/peak/reach-direction + movement window, on the SEQ grid frac.
    seg = S.segment_cup_only(base, fps=Q.COMMON_HZ)
    disp_b = np.linalg.norm(base - rest, axis=1)
    mv = np.isin(seg["phase"], [S.P_FWD, S.P_DRINK, S.P_BACK])
    if mv.sum() >= 10:
        mi = np.flatnonzero(mv); lo, hi = mi.min(), mi.max()
        pkf = lo + int(np.argmax(disp_b[lo:hi + 1]))
        peak = float(disp_b[pkf]); pkdir = (base[pkf] - rest); pkdir /= np.linalg.norm(pkdir) + 1e-9
        # per-frame fraction through the movement (for warping the shape later)
        frac = np.zeros(Tn)
        frac[lo:pkf + 1] = np.linspace(0, 0.5, pkf - lo + 1) if pkf > lo else 0
        frac[pkf:hi + 1] = np.linspace(0.5, 1.0, hi - pkf + 1) if hi > pkf else 0.5
    else:
        peak = 0.0; pkdir = np.zeros(3); frac = np.zeros(Tn)

    def rs(a):                                  # resample (T,k) -> (SEQ,k) linear
        x0 = np.linspace(0, 1, len(a)); x1 = np.linspace(0, 1, SEQ)
        return np.stack([np.interp(x1, x0, a[:, j]) for j in range(a.shape[1])], 1)
    tmask = np.isfinite(tgt).all(1)
    return dict(feats=rs(feats).astype(np.float32),
                tgt=rs(np.nan_to_num(tgt)).astype(np.float32),
                tmask=rs(tmask[:, None].astype(float))[:, 0] > 0.5,
                ncams_seq=rs(ncams[:, None])[:, 0], Tn=Tn, basis=basis, rest=rest,
                frac_seq=rs(frac[:, None])[:, 0], peak=peak,
                pkdir_local=(pkdir @ basis.T).astype(np.float32),   # reach dir in rep frame
                base=base, mr=mr, cons=cons, ncams=ncams,
                ph=seg["phase"],
                video=rep["video"], pid=rep["video"].split("_")[0])


class TCN(nn.Module):
    """Shared dilated-conv body; `nout` output channels (3 for position, 1 for var)."""
    def __init__(self, fin, nout, ch=64, layers=6, var=False):
        super().__init__()
        blocks = []; c = fin
        for i in range(layers):
            d = 2 ** i
            blocks += [nn.Conv1d(c, ch, 5, padding=2 * d, dilation=d), nn.ReLU(), nn.BatchNorm1d(ch)]
            c = ch
        self.body = nn.Sequential(*blocks)
        self.head = nn.Conv1d(ch, nout, 1)
        self.var = var

    def forward(self, x):
        y = self.head(self.body(x))
        return nn.functional.softplus(y) + 625.0 if self.var else y    # var floored at 25mm^2


def big_error_loss(pos, tgt, mask, w, p=1.5):
    """Position loss that makes a BIG error matter MORE than a small one (user): the
    per-frame squared error is raised to power p>1, so the model can't minimise the
    mean by predicting flat-zero and eating a huge apex miss (plain MSE's failure on a
    mostly-rest signal). Also weighted per-frame by w (apex/occluded frames up)."""
    m = (mask.unsqueeze(1) * w.unsqueeze(1))
    se = ((pos - tgt) ** 2).sum(1, keepdim=True)           # squared error per frame
    loss = (se + 1.0) ** p                                  # superlinear in the error
    return (loss * m).sum() / m.sum().clamp(min=1)


def var_nll(varpred, sq_resid, mask):
    """Train the SEPARATE variance model: predict the position model's realized
    squared error. NLL of an isotropic Gaussian whose var = varpred, residual fixed."""
    m = mask.unsqueeze(1)
    loss = 0.5 * (sq_resid / varpred + 3 * torch.log(varpred))
    return (loss * m).sum() / m.sum().clamp(min=1)


def kf_with_learned(cons, ncams, z_learned, r_learned, fps=Q.COMMON_HZ,
                    q=200.0**2, r=30.0**2):
    """KF/RTS fusing real consensus (R=r) AND the learned measurement (R=r_learned[t])."""
    Tn = len(cons); dt = 1.0 / fps
    F = np.eye(6); F[:3, 3:] = dt * np.eye(3)
    H = np.zeros((3, 6)); H[:, :3] = np.eye(3)
    Qm = np.zeros((6, 6))
    Qm[:3, :3] = q * dt**3 / 3 * np.eye(3); Qm[:3, 3:] = q * dt**2 / 2 * np.eye(3)
    Qm[3:, :3] = q * dt**2 / 2 * np.eye(3); Qm[3:, 3:] = q * dt * np.eye(3)
    valid = np.isfinite(cons).all(1); idx = np.flatnonzero(valid)
    if len(idx) < 2:
        return np.full((Tn, 3), np.nan)
    x = np.zeros(6); x[:3] = cons[idx[0]]
    P = np.diag([50, 50, 50, 500, 500, 500.0])**2
    xs_p, Ps_p, xs_u, Ps_u = [], [], [], []
    for tt in range(Tn):
        x = F @ x; P = F @ P @ F.T + Qm
        xs_p.append(x.copy()); Ps_p.append(P.copy())
        if valid[tt]:                            # real consensus
            y = cons[tt] - H @ x; Sm = H @ P @ H.T + r * np.eye(3)
            K = P @ H.T @ np.linalg.inv(Sm); x = x + K @ y; P = (np.eye(6) - K @ H) @ P
        # learned measurement (always available; its R says how much to trust it)
        y = z_learned[tt] - H @ x; Sm = H @ P @ H.T + r_learned[tt] * np.eye(3)
        K = P @ H.T @ np.linalg.inv(Sm); x = x + K @ y; P = (np.eye(6) - K @ H) @ P
        xs_u.append(x.copy()); Ps_u.append(P.copy())
    xs_s = [None] * Tn; xs_s[-1] = xs_u[-1]
    for tt in range(Tn - 2, -1, -1):
        C = Ps_u[tt] @ F.T @ np.linalg.inv(Ps_p[tt + 1])
        xs_s[tt] = xs_u[tt] + C @ (xs_s[tt + 1] - xs_p[tt + 1])
    return np.array([s[:3] for s in xs_s])


def main():
    print(f"device {DEV}; building sequences (ETA ~5 min)...", flush=True)
    reps = [r for rep in T._reps() if (r := build_seq(rep)) is not None]
    pids = sorted({r["pid"] for r in reps})

    def shape_feat(r, shape):
        """3-channel shape-prior position (rep-local) on the SEQ grid: where the
        learned drink template says the cup is, anchored to this rep's peak/dir."""
        mag = np.interp(r["frac_seq"], T.SHAPE_T, shape) * r["peak"]   # |reach|(t)
        return (mag[:, None] * r["pkdir_local"][None, :]).astype(np.float32)  # (SEQ,3)

    fin = reps[0]["feats"].shape[1] + 3       # + shape-prior position channels
    print(f"{len(reps)} reps, {fin} features (incl. shape-prior input)", flush=True)

    b_de, s_de, common = {}, {}, []
    for held in pids:
        trn = [r for r in reps if r["pid"] != held]; te = [r for r in reps if r["pid"] == held]
        shape = T.learn_shapes(exclude_pid=held)          # LOPO drink template (disp)

        def feats_with_shape(rr):
            return np.concatenate([rr["feats"], shape_feat(rr, shape)], axis=1)
        Xtr = torch.tensor(np.stack([feats_with_shape(r) for r in trn])).transpose(1, 2).to(DEV)
        Ytr = torch.tensor(np.stack([r["tgt"] for r in trn])).transpose(1, 2).to(DEV)
        Mtr = torch.tensor(np.stack([r["tmask"] for r in trn]).astype(np.float32)).to(DEV)
        # per-frame weight: emphasise apex/occluded frames (displacement-from-rest +
        # occlusion) so the reach isn't drowned by the many rest frames.
        disp_w = np.stack([np.abs(r["tgt"][:, 0]) / 200.0 for r in trn])      # reach magnitude
        occ_w = np.stack([(r["ncams_seq"] < 2).astype(np.float32) for r in trn])
        W = torch.tensor((1.0 + disp_w + 2.0 * occ_w).astype(np.float32)).to(DEV)

        # --- Model A: POSITION (big-error loss so it can't collapse to flat) ---
        posnet = TCN(fin, 3).to(DEV)
        optp = torch.optim.Adam(posnet.parameters(), lr=2e-3, weight_decay=1e-4)
        posnet.train()
        for ep in range(260):
            optp.zero_grad()
            loss = big_error_loss(posnet(Xtr), Ytr, Mtr, W)
            loss.backward(); optp.step()
        posnet.eval()

        # --- Model B: SEPARATE VARIANCE model, trained on Model A's residuals ---
        with torch.no_grad():
            sq = ((posnet(Xtr) - Ytr) ** 2).sum(1, keepdim=True).detach()    # realized error^2
        varnet = TCN(fin, 1, var=True).to(DEV)
        optv = torch.optim.Adam(varnet.parameters(), lr=2e-3, weight_decay=1e-4)
        varnet.train()
        for ep in range(160):
            optv.zero_grad()
            loss = var_nll(varnet(Xtr), sq, Mtr)
            loss.backward(); optv.step()
        varnet.eval()

        with torch.no_grad():
            for r in te:
                xb = torch.tensor(feats_with_shape(r)[None]).transpose(1, 2).to(DEV)
                pos = posnet(xb)[0].cpu().numpy().T            # (SEQ,3) rep-local
                var = varnet(xb)[0, 0].cpu().numpy().clip(625.0, 1e6)
                # resample SEQ -> Tn, map to world
                x0 = np.linspace(0, 1, SEQ); x1 = np.linspace(0, 1, r["Tn"])
                pos_t = np.stack([np.interp(x1, x0, pos[:, k]) for k in range(3)], 1)
                var_t = np.interp(x1, x0, var)
                z_world = pos_t @ r["basis"] + r["rest"]
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
        # held-out sanity: prediction error AND predicted reach range (collapse check)
        with torch.no_grad():
            pe, reach = [], []
            for r in te:
                p = posnet(torch.tensor(feats_with_shape(r)[None]).transpose(1, 2).to(DEV))[0].cpu().numpy().T
                mk = r["tmask"]
                pe.append(np.median(np.linalg.norm(p[mk] - r["tgt"][mk], axis=1)))
                reach.append(p[mk, 0].max())            # how far the model thinks the cup reaches
        print(f"  held {held}: {len(te)} reps  pred-err {np.median(pe):.0f}mm  "
              f"pred-reach {np.median(reach):.0f}mm (target ~250-550 => collapse if <100)", flush=True)

    common = [k for k in common if k in b_de and k in s_de]
    B = np.array([b_de[k] for k in common]); SQ = np.array([s_de[k] for k in common])
    print(f"\n=== SEQ model -> learned measurement in KF (LOPO), {len(common)} reps ===")
    print(f"  baseline KF drinking err   : {np.median(B):.1f}mm")
    print(f"  seq-measurement + KF       : {np.median(SQ):.1f}mm  ({(np.median(SQ)/np.median(B)-1)*100:+.0f}%)")
    print(f"  improved {(SQ<B).sum()}, worsened {(SQ>B).sum()}, mean {np.mean(SQ-B):+.1f}mm")
    json.dump({"n": len(common), "base": float(np.median(B)), "seq": float(np.median(SQ)),
               "improved": int((SQ < B).sum()), "worsened": int((SQ > B).sum())},
              open(CACHE / "learn_seq_kf.json", "w"), indent=2)
    print("wrote cache/learn_seq_kf.json")


if __name__ == "__main__":
    main()
