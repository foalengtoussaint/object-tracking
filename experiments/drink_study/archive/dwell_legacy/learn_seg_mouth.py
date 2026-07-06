"""Segmenter with the HEAD signal: cup->mouth distance/velocity as FEATURES, and the
mouth-based dwell as TRUTH (the correct label -- see project_mouth_dwell_truth).

Built on learn_seg_hybrid (13 fused-track kinematic ch + 4 detection-occlusion ch). Adds
4 head channels from mouth_features (cup->mouth mm, approach velocity, normalised distance,
present-flag), all brought onto the 60Hz track grid via the qtm_align pairing + sync lag.

To separate "the head TRUTH matters" from "the head FEATURE matters", one LOPO run trains
TWO clf models against the SAME mouth truth:
  * base17  = 13 fused + 4 occ            (no head feature)  -- ceiling of the OLD inputs
  * mouth21 = 13 fused + 4 occ + 4 head   (with head feature)
scored vs each other and vs the tuned gate (the production floor). All against mouth truth.

    python experiments/drink_study/learn_seg_mouth.py [--epochs 250]
Run from repo root; GPU. Writes cache/learn_seg_mouth.json
"""
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import glob, json, time, argparse, numpy as np, torch, torch.nn as nn
import learn_seg as LS, learn_seq_kf as M, tune_interp as T
import mouth_features as MF
from _paths import CACHE
HZ = LS.HZ; SEQ_L = LS.SEQ; DEV = LS.DEV; TUN = LS.TUN
LF = LS.CACHE
OCC_IDX = [3, 14, 15, 16]                       # present, mpx, ncams, occ (as in hybrid)


def build():
    print(f"device {DEV}; building detection occlusion feats (build_seq)...", flush=True)
    occ = {}
    for rep in T._reps():
        r = M.build_seq(rep)
        if r is not None:
            occ[rep['video']] = r['feats'][:, OCC_IDX]
    print(f"{len(occ)} reps have detection feats; loading fused cache + head signals...", flush=True)
    reps = []
    n_nohead = 0
    for f in sorted(glob.glob(str(LF / '*.npz'))):
        d = np.load(f, allow_pickle=True); v = str(d['video'])
        if v not in occ:
            continue
        r = LS.build_rep(d)                        # 13ch fused feats + native T
        T_ = r['T']
        rest = np.asarray(d['rest'], float); basis = np.asarray(d['basis'], float)
        cup_world = np.asarray(d['fused'], float)                  # (T,3) lab-frame cup
        mth = MF.tracked_cup_to_head_channels(v, T_, cup_world)  # (T,4) TRACKED cup -> mocap head
        pts = MF.points_channels(v, T_, rest, basis)  # (T,16) cup+head markers, SHARED space
        dst = MF.dist_channels(v, T_, cup_world)      # (T,6) cup-to-head-marker distances
        msk = MF.mouth_truth_mask(v, T_)          # (T,) mouth dwell truth
        if mth is None or pts is None or dst is None or msk is None or msk.sum() < 3:
            n_nohead += 1
            continue
        msp = MF.mouth_span(v, T_)
        # normalise the raw mm channels to the same ~unit scale as other feats
        mth = mth.copy(); mth[:, 0] = mth[:, 0] / 600.0; mth[:, 1] = mth[:, 1] / 20.0
        fx13 = LS.resample(r['feats'], SEQ_L)                       # (SEQ,13)
        fxocc = LS.resample(occ[v], SEQ_L)                         # (SEQ,4)
        fxprox = LS.resample(mth, SEQ_L)                           # (SEQ,4) mouth proxy
        fxpts = LS.resample(pts, SEQ_L)                            # (SEQ,16) points-in-shared-space
        fxdst = LS.resample(dst, SEQ_L)                            # (SEQ,6) cup-to-marker dists
        base = np.concatenate([fx13, fxocc], 1)                    # (SEQ,17)
        r['fx17'] = base.astype(np.float32)                                        # no head
        r['fx_prox'] = np.concatenate([base, fxprox], 1).astype(np.float32)        # +proxy       (21)
        r['fx_pts'] = np.concatenate([base, fxpts], 1).astype(np.float32)          # +points      (33)
        r['fx_dst'] = np.concatenate([base, fxdst], 1).astype(np.float32)          # +dists       (23)
        r['fx_combo'] = np.concatenate([base, fxprox, fxdst], 1).astype(np.float32)  # proxy+dists (27)
        r['mx'] = LS.resample(msk, SEQ_L).astype(np.float32)        # MOUTH truth mask
        r['tsp'] = msp                                             # MOUTH truth span (track frames)
        reps.append(r)
    print(f"{len(reps)} reps usable ({n_nohead} dropped: no head/pairing/dwell)", flush=True)
    return reps


def train_clf(reps, held, fxkey, epochs):
    trn = [r for r in reps if r['pid'] != held]
    te = [r for r in reps if r['pid'] == held]
    Xtr = torch.tensor(np.stack([r[fxkey] for r in trn])).transpose(1, 2).to(DEV)
    Mtr = torch.tensor(np.stack([r['mx'] for r in trn])).to(DEV)
    cin = Xtr.shape[1]
    clf = LS.TCN(cin, nout=1).to(DEV)
    opt = torch.optim.Adam(clf.parameters(), lr=2e-3, weight_decay=1e-4)
    pos = Mtr.mean().clamp(1e-3, 1 - 1e-3); w = ((1 - pos) / pos).item()
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w, device=DEV)); clf.train()
    for _ in range(epochs):
        opt.zero_grad(); lossf(clf(Xtr, 'frame').squeeze(1), Mtr).backward(); opt.step()
    clf.eval()
    with torch.no_grad():
        ptr = torch.sigmoid(clf(Xtr, 'frame').squeeze(1)).cpu().numpy()
        best_thr, best = 0.5, 1e9
        for thr in np.arange(0.15, 0.96, 0.05):
            des = [LS.errs(r['tsp'], LS.span_from_prob(LS.resample(ptr[k], r['T']), thr), r['T'])[0]
                   for k, r in enumerate(trn)]
            if np.mean(des) < best:
                best = np.mean(des); best_thr = thr
        out = {}
        for r in te:
            x = torch.tensor(r[fxkey][None]).transpose(1, 2).to(DEV)
            pr = LS.resample(torch.sigmoid(clf(x, 'frame').squeeze(1))[0].cpu().numpy(), r['T'])
            out[r['video']] = LS.span_from_prob(pr, best_thr)
    return out, best_thr


METHODS = [('base17', 'fx17'), ('proxy21', 'fx_prox'),
           ('points33', 'fx_pts'), ('dists23', 'fx_dst'),
           ('combo27', 'fx_combo')]     # proxy + distance fallback = the design to beat


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--epochs', type=int, default=250)
    a = ap.parse_args()
    reps = build()
    pids = sorted({r['pid'] for r in reps})
    print(f"{len(reps)} reps, {len(pids)} folds; training {', '.join(m for m,_ in METHODS)} "
          f"per fold (truth=MOUTH)", flush=True)
    P = {nm: {} for nm, _ in METHODS}; Pt = {}
    t0 = time.time()
    for fi, held in enumerate(pids):
        ths = []
        for nm, key in METHODS:
            out, th = train_clf(reps, held, key, a.epochs)
            P[nm].update(out); ths.append(f"{nm[:5]}={th:.2f}")
        for r in [x for x in reps if x['pid'] == held]:
            Pt[r['video']] = LS.geo_span(r['fused'], **TUN)       # tuned gate = production floor
        el = time.time() - t0
        print(f"  [{fi+1}/{len(pids)}] {held}: {' '.join(ths)} "
              f"({el:.0f}s, ~{el/(fi+1)*(len(pids)-fi-1):.0f}s left)", flush=True)

    tsp = {r['video']: r['tsp'] for r in reps}; Tn = {r['video']: r['T'] for r in reps}
    common = [k for k in P['base17'] if all(k in P[nm] for nm, _ in METHODS) and k in Pt]

    def st(pred):
        return np.array([LS.errs(tsp[k], pred[k], Tn[k])[0] for k in common])
    D = {nm: st(P[nm]) for nm, _ in METHODS}; dt = st(Pt)
    print(f"\n=== segmenter vs MOUTH truth, LOPO {len(common)} reps ===")
    print(f"  {'method':<11}{'mean':>7}{'p50':>6}{'p90':>6}{'p95':>7}{'p99':>7}{'max':>7}")
    for nm, de in [('tuned', dt)] + [(nm, D[nm]) for nm, _ in METHODS]:
        print(f"  {nm:<11}{de.mean():>7.0f}{np.percentile(de,50):>6.0f}{np.percentile(de,90):>6.0f}"
              f"{np.percentile(de,95):>7.0f}{np.percentile(de,99):>7.0f}{de.max():>7.0f}", flush=True)
    # each head variant vs the no-head baseline
    d0 = D['base17']
    print(f"\n  head-feature effect vs base17 (no head):")
    for nm, _ in METHODS[1:]:
        de = D[nm]; b = (de < d0 - 1).sum(); wr = (de > d0 + 1).sum()
        print(f"    {nm:<10} mean {d0.mean():.0f}->{de.mean():.0f}  p90 {np.percentile(d0,90):.0f}->"
              f"{np.percentile(de,90):.0f}  max {d0.max():.0f}->{de.max():.0f}  "
              f"(better {b}, worse {wr})", flush=True)
    # does raw geometry (points / dists) fix the reps the PROXY HURT?
    hurt = np.argsort(-(D['proxy21'] - d0))[:12]
    print(f"\n  === reps the PROXY hurt most: do POINTS / DISTS recover them? ===")
    print(f"    {'video':<38}{'base':>6}{'proxy':>7}{'points':>7}{'dists':>7}")
    for i in hurt:
        k = common[i]
        print(f"    {k[:38]:<38}{d0[i]:>6.0f}{D['proxy21'][i]:>7.0f}"
              f"{D['points33'][i]:>7.0f}{D['dists23'][i]:>7.0f}", flush=True)
    js = {'n': len(common), 'truth': 'mouth'}
    js['tuned'] = {'mean': float(dt.mean()), 'p50': float(np.percentile(dt, 50)),
                   'p90': float(np.percentile(dt, 90)), 'p95': float(np.percentile(dt, 95)),
                   'p99': float(np.percentile(dt, 99)), 'max': float(dt.max())}
    for nm, _ in METHODS:
        de = D[nm]
        js[nm] = {'mean': float(de.mean()), 'p50': float(np.percentile(de, 50)),
                  'p90': float(np.percentile(de, 90)), 'p95': float(np.percentile(de, 95)),
                  'p99': float(np.percentile(de, 99)), 'max': float(de.max())}
    label = "[tuned," + ",".join(nm for nm, _ in METHODS) + "]"
    js['perrep_cols'] = ['tuned'] + [nm for nm, _ in METHODS]
    js['perrep'] = {common[i]: [float(dt[i])] + [float(D[nm][i]) for nm, _ in METHODS]
                    for i in range(len(common))}
    json.dump(js, open(CACHE / 'learn_seg_mouth.json', 'w'), indent=2)
    print(f"\nwrote cache/learn_seg_mouth.json  (perrep = {label})", flush=True)


if __name__ == '__main__':
    main()
