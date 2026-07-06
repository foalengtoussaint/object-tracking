"""Diagnostic graphs of the WORST proxy21-TCN reps — LOOK before modeling more.

For each of the worst-N reps (by cached proxy21 error): retrain that rep's held-out LOPO
fold (proxy21 features, plain TCN), get the per-frame P(drink), and plot on one timeline:
  - cup->mouth distance (the signal),
  - the MOUTH-truth dwell band,
  - the TCN's predicted P(drink) + its thresholded span,
so you can SEE the failure mode (late/early edge? split dwell? missed entirely? truth weird?).

    python experiments/drink_study/analysis/plot_worst.py [--n 9] [--epochs 250]
Writes slides/worst_proxy21_grid.png
"""
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import json, argparse, numpy as np, torch
import learn_seg_mouth as LM
import learn_seg as LS
import mouth_features as MF
from _paths import CACHE, DS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HZ = LS.HZ


def worst_videos(n):
    """Worst n reps by proxy21 vs the BRIDGED truth; errmap[k] = (proxy21_err, base17_err)."""
    d = json.load(open(CACHE / 'learn_seg_mouth.json'))
    ip = d['perrep_cols'].index('proxy21'); ib = d['perrep_cols'].index('base17')
    rows = sorted(((v[ip], v[ib], k) for k, v in d['perrep'].items() if v[ip] is not None),
                  reverse=True)
    return [k for _, _, k in rows[:n]], {k: (p, b) for p, b, k in rows}


def tcn_prob_for(reps, held, fxkey='fx_prox'):
    """Train a TCN (features=fxkey) on all-but-held; return {video: (prob(T,), thr)} held reps."""
    trn = [r for r in reps if r['pid'] != held]
    te = [r for r in reps if r['pid'] == held]
    Xtr = torch.tensor(np.stack([r[fxkey] for r in trn])).transpose(1, 2).to(LS.DEV)
    Mtr = torch.tensor(np.stack([r['mx'] for r in trn])).to(LS.DEV)
    clf = LS.TCN(Xtr.shape[1], nout=1).to(LS.DEV)
    opt = torch.optim.Adam(clf.parameters(), lr=2e-3, weight_decay=1e-4)
    pos = Mtr.mean().clamp(1e-3, 1 - 1e-3); w = ((1 - pos) / pos).item()
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w, device=LS.DEV)); clf.train()
    for _ in range(EPOCHS):
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
            x = torch.tensor(r[fxkey][None]).transpose(1, 2).to(LS.DEV)
            pr = LS.resample(torch.sigmoid(clf(x, 'frame').squeeze(1))[0].cpu().numpy(), r['T'])
            out[r['video']] = (pr, best_thr)
    return out


PRED_CACHE = str(CACHE / 'worst_preds_head.json')
FEATS = [('proxy21', 'fx_prox'), ('base17', 'fx17')]      # both, to show where head helps


def load_or_train_preds(reps, pids, byv, refresh=False):
    """Return {video: {method: (prob[T], thr)}} for the folds owning `pids`, for BOTH proxy21
    and base17. Reuses PRED_CACHE; only trains folds/methods not present. GPU-free replot."""
    import os
    cache = json.load(open(PRED_CACHE)) if (os.path.exists(PRED_CACHE) and not refresh) else {}
    def have(v):
        return v in cache and all(m in cache[v] for m, _ in FEATS)
    need_pids = sorted({byv[v]['pid'] for p in pids for v in byv if byv[v]['pid'] == p and not have(v)})
    if need_pids:
        print(f"training {len(need_pids)} fold(s) × {len(FEATS)} models: {need_pids}", flush=True)
        for j, held in enumerate(need_pids):
            for m, key in FEATS:
                for v, (pr, thr) in tcn_prob_for(reps, held, key).items():
                    cache.setdefault(v, {})[m] = {'prob': [round(float(x), 4) for x in pr],
                                                  'thr': float(thr)}
            print(f"  [{j+1}/{len(need_pids)}] {held}", flush=True)
        json.dump(cache, open(PRED_CACHE, 'w'))
        print(f"cached -> {PRED_CACHE} ({len(cache)} reps)", flush=True)
    else:
        print(f"all served from {PRED_CACHE} (no training)", flush=True)
    return {v: {m: (np.array(cache[v][m]['prob']), cache[v][m]['thr']) for m, _ in FEATS}
            for p in pids for v in byv if byv[v]['pid'] == p and have(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=9)
    ap.add_argument('--epochs', type=int, default=250)
    ap.add_argument('--refresh', action='store_true', help='retrain folds even if cached')
    a = ap.parse_args()
    global EPOCHS; EPOCHS = a.epochs
    want, errmap = worst_videos(a.n)
    reps = LM.build()
    byv = {r['video']: r for r in reps}
    want = [v for v in want if v in byv]
    pids = sorted({byv[v]['pid'] for v in want})
    probs = load_or_train_preds(reps, pids, byv, refresh=a.refresh)

    ncol = 3; nrow = int(np.ceil(len(want) / ncol))
    plt.rcParams.update({'font.size': 11})
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.0 * ncol, 3.4 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for idx, v in enumerate(want):
        r = byv[v]; T = r['T']; t = np.arange(T) / HZ
        pr_p, thr_p = probs[v]['proxy21']; pred_p = LS.span_from_prob(pr_p, thr_p)
        pr_b, thr_b = probs[v]['base17'];  pred_b = LS.span_from_prob(pr_b, thr_b)
        tsp = r['tsp']                                   # bridged mouth truth span
        # green = TRACKED cup->head distance (what proxy21 is actually fed), + mocap as ref
        cup_world = np.asarray(np.load([f for f in __import__('glob').glob(
            str(CACHE / 'lopo_fused' / '*.npz'))
            if v in str(np.load(f, allow_pickle=True)['video'])][0],
            allow_pickle=True)['fused'], float)
        mct = MF.tracked_cup_to_head_channels(v, T, cup_world)
        mcm = MF.mouth_channels(v, T)                                        # mocap ref (truth signal)
        # channel 3 = present-flag; blank frames OUTSIDE mocap coverage so the edge
        # gap-fill (a ramp across empty frames) isn't drawn as a false spike
        def _covered(ch):
            if ch is None:
                return np.full(T, np.nan)
            d = ch[:, 0].copy(); d[ch[:, 3] < 0.5] = np.nan
            return d
        dist = _covered(mct)                                                 # TRACKED cup->head
        dist_mocap = _covered(mcm)                                           # mocap ref
        ax = axes.flat[idx]; ax.axis('on')
        ax.plot(t, dist_mocap, color='#9ccfa8', lw=1.0, alpha=0.8)     # faint mocap reference
        ax.plot(t, dist, color='#1a8a3a', lw=1.8)                      # TRACKED (what model sees)
        dmax = max(np.nanmax(dist) * 1.08, 1) if np.isfinite(dist).any() else 1
        ax.set_ylim(0, dmax * 1.30)                        # headroom for the dwell bars on top
        ax.set_ylabel('cup→head (mm): tracked(dark)/mocap(faint)', fontsize=10, color='#1a8a3a')
        ax.tick_params(axis='y', labelcolor='#1a8a3a', labelsize=9)
        ax2 = ax.twinx()
        ax2.plot(t, pr_p, color='#c0392b', lw=1.6, alpha=0.9)         # proxy21 P(drink)
        ax2.plot(t, pr_b, color='#888', lw=1.2, alpha=0.8, ls='--')   # base17 P(drink)
        ax2.set_ylim(0, 1.30); ax2.set_ylabel('P(drink)', fontsize=10, color='#c0392b')
        ax2.tick_params(axis='y', labelcolor='#c0392b', labelsize=9)
        # DWELL BARS as clean lines stacked above the curves (axis fraction, y>1.0 region)
        bars = [('truth',   tsp,    '#e8930a'),
                ('proxy21', pred_p, '#c0392b'),
                ('base17',  pred_b, '#777')]
        y0 = 1.05
        for bi, (nm, sp, col) in enumerate(bars):
            y = y0 + bi * 0.075
            ax2.axhline(y, xmin=0, xmax=1, color='#ddd', lw=0.6, zorder=1)   # faint rail
            if sp:
                ax2.plot([sp[0] / HZ, sp[1] / HZ], [y, y], color=col, lw=5,
                         solid_capstyle='butt', zorder=3, clip_on=False)
            ax2.text(-0.012, y, nm, transform=ax2.get_yaxis_transform(),
                     ha='right', va='center', fontsize=7.5, color=col)
        p = v.split('_')[0]; side = 'R' if 'right' in v else 'L'
        ep = errmap[v][0]; eb = errmap[v][1]
        ax.set_title(f"{p} {side}   proxy21 {ep:.0f} / base17 {eb:.0f} ms",
                     fontsize=11, fontweight='bold', pad=24)   # pad to clear the dwell bars
        ax.set_xlabel('time (s)', fontsize=10); ax.tick_params(axis='x', labelsize=9)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color='#1a8a3a', lw=2, label='TRACKED cup→head (dark) / mocap ref (faint)'),
               Line2D([], [], color='#c0392b', lw=2, label='proxy21 P(drink)'),
               Line2D([], [], color='#888', lw=2, ls='--', label='base17 P(drink)'),
               Line2D([], [], color='#e8930a', lw=5, label='truth dwell (bar)'),
               Line2D([], [], color='#c0392b', lw=5, label='proxy21 dwell (bar)'),
               Line2D([], [], color='#777', lw=5, label='base17 dwell (bar)')]
    fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle('Worst reps — proxy21 (TRACKED cup→head) vs base17 — dwell bars: truth / proxy21 / base17',
                 fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0.05, 1, 0.95], h_pad=2.5)
    out = str(DS / 'slides' / 'worst_proxy21_grid.png')
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}", flush=True)


if __name__ == '__main__':
    main()
