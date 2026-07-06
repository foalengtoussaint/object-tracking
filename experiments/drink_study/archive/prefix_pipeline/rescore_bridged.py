"""Re-score the segmenter against the BRIDGED mouth truth (multi-apex drinks no longer
chopped — see mouth_dwell._bridged_span). build() already loads the bridged truth, so this
just retrains base17 + proxy21 under it (the training target changed, so we must retrain)
and reports vs the pre-bridge numbers.

    python experiments/drink_study/rescore_bridged.py [--epochs 250]
Writes cache/learn_seg_mouth_bridged.json
"""
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import sys, json, time, argparse, numpy as np
sys.path.insert(0, 'experiments/drink_study')
import learn_seg_mouth as LM
import learn_seg as LS
TUN = LM.TUN
METHODS = [('base17', 'fx17'), ('proxy21', 'fx_prox')]
PRE = 'experiments/drink_study/cache/learn_seg_mouth.json'      # pre-bridge results


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--epochs', type=int, default=250)
    a = ap.parse_args()
    reps = LM.build()                                   # BRIDGED truth (mx / tsp)
    pids = sorted({r['pid'] for r in reps})
    print(f"{len(reps)} reps, {len(pids)} folds; retraining {', '.join(m for m,_ in METHODS)} "
          f"vs BRIDGED truth", flush=True)
    P = {nm: {} for nm, _ in METHODS}; Pt = {}
    t0 = time.time()
    for fi, held in enumerate(pids):
        for nm, key in METHODS:
            out, _ = LM.train_clf(reps, held, key, a.epochs); P[nm].update(out)
        for r in [x for x in reps if x['pid'] == held]:
            Pt[r['video']] = LS.geo_span(r['fused'], **TUN)
        el = time.time() - t0
        print(f"  [{fi+1}/{len(pids)}] {held} ({el:.0f}s, ~{el/(fi+1)*(len(pids)-fi-1):.0f}s left)",
              flush=True)

    tsp = {r['video']: r['tsp'] for r in reps}; Tn = {r['video']: r['T'] for r in reps}
    common = [k for k in P['proxy21'] if k in P['base17'] and k in Pt]

    def st(pred):
        return np.array([LS.errs(tsp[k], pred[k], Tn[k])[0] for k in common])
    D = {nm: st(P[nm]) for nm, _ in METHODS}; dt = st(Pt)

    pre = json.load(open(PRE))
    print(f"\n=== vs BRIDGED mouth truth, LOPO {len(common)} reps  (pre-bridge in parens) ===")
    print(f"  {'method':<10}{'mean':>8}{'p50':>7}{'p90':>7}{'p95':>7}{'p99':>7}{'max':>7}")
    for nm, de in [('tuned', dt)] + [(nm, D[nm]) for nm, _ in METHODS]:
        pm = pre.get(nm, {}) if nm != 'tuned' else pre.get('tuned', {})
        pre_mean = f"({pm.get('mean',0):.0f})" if pm else ""
        print(f"  {nm:<10}{de.mean():>8.0f}{np.percentile(de,50):>7.0f}{np.percentile(de,90):>7.0f}"
              f"{np.percentile(de,95):>7.0f}{np.percentile(de,99):>7.0f}{de.max():>7.0f}   "
              f"pre-bridge mean {pre_mean}", flush=True)
    dp = D['proxy21']
    print(f"\n  proxy21 vs BRIDGED truth: mean {pre['proxy21']['mean']:.0f} -> {dp.mean():.0f}  "
          f"p50 {pre['proxy21']['p50']:.0f} -> {np.percentile(dp,50):.0f}  "
          f"max {pre['proxy21']['max']:.0f} -> {dp.max():.0f}", flush=True)
    # biggest improvements
    js = {'n': len(common), 'truth': 'mouth_bridged', 'leave_frac': 0.40,
          'perrep_cols': ['tuned', 'base17', 'proxy21'],
          'perrep': {common[i]: [float(dt[i]), float(D['base17'][i]), float(dp[i])]
                     for i in range(len(common))}}
    for nm, de in [('base17', D['base17']), ('proxy21', dp)]:
        js[nm] = {'mean': float(de.mean()), 'p50': float(np.percentile(de, 50)),
                  'p90': float(np.percentile(de, 90)), 'p95': float(np.percentile(de, 95)),
                  'p99': float(np.percentile(de, 99)), 'max': float(de.max())}
    json.dump(js, open('experiments/drink_study/cache/learn_seg_mouth_bridged.json', 'w'), indent=2)
    print("\nwrote cache/learn_seg_mouth_bridged.json", flush=True)


if __name__ == '__main__':
    main()
