"""Train ONLY the combo27 (base + mouth-proxy + cup->marker distances) method and MERGE it
into the existing cache/learn_seg_mouth.json — the other 4 methods are already cached there,
so don't retrain them. Reuses learn_seg_mouth.build() (which only reads cached features; no
GPU inference / video decode is re-run).

    python experiments/drink_study/add_combo.py [--epochs 250]
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

CACHE = 'experiments/drink_study/cache/learn_seg_mouth.json'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--epochs', type=int, default=250)
    a = ap.parse_args()
    old = json.load(open(CACHE))
    cols = old['perrep_cols']                      # e.g. [tuned,base17,proxy21,points33,dists23]
    if 'combo27' in cols:
        print("combo27 already in cache; nothing to do"); return
    reps = LM.build()
    pids = sorted({r['pid'] for r in reps})
    print(f"{len(reps)} reps, {len(pids)} folds; training ONLY combo27 (reusing cached 4 methods)",
          flush=True)
    Pc = {}
    t0 = time.time()
    for fi, held in enumerate(pids):
        out, th = LM.train_clf(reps, held, 'fx_combo', a.epochs)
        Pc.update(out)
        el = time.time() - t0
        print(f"  [{fi+1}/{len(pids)}] {held}: thr={th:.2f} "
              f"({el:.0f}s, ~{el/(fi+1)*(len(pids)-fi-1):.0f}s left)", flush=True)

    tsp = {r['video']: r['tsp'] for r in reps}; Tn = {r['video']: r['T'] for r in reps}
    # score combo27 on the SAME rep set the cache used (perrep keys)
    common = [k for k in old['perrep'] if k in Pc]
    dc = np.array([LS.errs(tsp[k], Pc[k], Tn[k])[0] for k in common])
    # existing per-method error arrays, aligned to `common`
    idx = {c: i for i, c in enumerate(cols)}
    old_arr = {c: np.array([old['perrep'][k][idx[c]] for k in common]) for c in cols}

    print(f"\n=== segmenter vs MOUTH truth, LOPO {len(common)} reps (combo merged) ===")
    print(f"  {'method':<11}{'mean':>7}{'p50':>6}{'p90':>6}{'p95':>7}{'p99':>7}{'max':>7}")
    show = cols + ['combo27']
    arrs = {**old_arr, 'combo27': dc}
    for nm in show:
        de = arrs[nm]
        print(f"  {nm:<11}{de.mean():>7.0f}{np.percentile(de,50):>6.0f}{np.percentile(de,90):>6.0f}"
              f"{np.percentile(de,95):>7.0f}{np.percentile(de,99):>7.0f}{de.max():>7.0f}", flush=True)
    d0, dp = old_arr['base17'], old_arr['proxy21']
    b = (dc < dp - 1).sum(); wr = (dc > dp + 1).sum()
    print(f"\n  combo27 vs proxy21: better {b}, worse {wr}, tie {len(common)-b-wr}  |  "
          f"mean {dp.mean():.0f}->{dc.mean():.0f}  p95 {np.percentile(dp,95):.0f}->{np.percentile(dc,95):.0f}  "
          f"max {dp.max():.0f}->{dc.max():.0f}", flush=True)
    # on the reps the proxy hurt (proxy worse than base), does combo fix them?
    hurt = np.argsort(-(dp - d0))[:12]
    print(f"\n  === reps the PROXY hurt most: does COMBO fix them? ===")
    print(f"    {'video':<38}{'base':>6}{'proxy':>7}{'combo':>7}")
    for i in hurt:
        print(f"    {common[i][:38]:<38}{d0[i]:>6.0f}{dp[i]:>7.0f}{dc[i]:>7.0f}", flush=True)

    # merge: append combo27 column to every perrep row + summary block
    old['perrep_cols'] = cols + ['combo27']
    cerr = {k: float(LS.errs(tsp[k], Pc[k], Tn[k])[0]) for k in Pc}
    for k in old['perrep']:
        old['perrep'][k] = old['perrep'][k] + [cerr.get(k)]   # None if this rep wasn't predicted
    old['combo27'] = {'mean': float(dc.mean()), 'p50': float(np.percentile(dc, 50)),
                      'p90': float(np.percentile(dc, 90)), 'p95': float(np.percentile(dc, 95)),
                      'p99': float(np.percentile(dc, 99)), 'max': float(dc.max())}
    json.dump(old, open(CACHE, 'w'), indent=2)
    print(f"\nmerged combo27 into {CACHE}", flush=True)


if __name__ == '__main__':
    main()
