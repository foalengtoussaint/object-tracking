"""clf-ONLY LOPO rerun with the RICH 3D+raw features (learn_seg.build_rep now emits 13ch).
Isolated from the 3-head harness so no slow resid gate-search. Compares to the tuned gate
and reports the tail reps (does direction+raw fix P20 place-down / P10 noisy dwell?).

    python experiments/drink_study/learn_seg_clf.py [--epochs 250]
Run from repo root. Per-fold prints. Writes cache/learn_seg_clf.json
"""
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import sys, glob, json, time, argparse, numpy as np, torch, torch.nn as nn
sys.path.insert(0,'experiments/drink_study')
import learn_seg as LS
HZ=LS.HZ; SEQ=LS.SEQ; DEV=LS.DEV; TUN=LS.TUN
CACHE=LS.CACHE

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=250); a=ap.parse_args()
    print(f"device {DEV}; loading + building RICH features...",flush=True)
    reps=[]
    for f in sorted(glob.glob(str(CACHE/'*.npz'))):
        d=np.load(f,allow_pickle=True); r=LS.build_rep(d)
        if r['tsp'] is None: continue
        r['fx']=LS.resample(r['feats'],SEQ).astype(np.float32)
        r['mx']=LS.resample(r['tmask'],SEQ).astype(np.float32)
        reps.append(r)
    pids=sorted({r['pid'] for r in reps}); cin=reps[0]['fx'].shape[1]
    print(f"{len(reps)} reps, {len(pids)} folds, {cin} feature channels",flush=True)
    Pc={}; Pt={}; t0=time.time()
    for fi,held in enumerate(pids):
        trn=[r for r in reps if r['pid']!=held]; te=[r for r in reps if r['pid']==held]
        Xtr=torch.tensor(np.stack([r['fx'] for r in trn])).transpose(1,2).to(DEV)
        Mtr=torch.tensor(np.stack([r['mx'] for r in trn])).to(DEV)
        clf=LS.TCN(cin,nout=1).to(DEV)
        opt=torch.optim.Adam(clf.parameters(),lr=2e-3,weight_decay=1e-4)
        pos=Mtr.mean().clamp(1e-3,1-1e-3); w=((1-pos)/pos).item()
        lossf=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w,device=DEV)); clf.train()
        for _ in range(a.epochs):
            opt.zero_grad(); lossf(clf(Xtr,'frame').squeeze(1),Mtr).backward(); opt.step()
        clf.eval()
        with torch.no_grad():
            ptr=torch.sigmoid(clf(Xtr,'frame').squeeze(1)).cpu().numpy()
            best_thr,best=0.5,1e9
            for thr in np.arange(0.15,0.96,0.05):
                des=[LS.errs(r['tsp'],LS.span_from_prob(LS.resample(ptr[k],r['T']),thr),r['T'])[0]
                     for k,r in enumerate(trn)]
                if np.mean(des)<best: best=np.mean(des); best_thr=thr
            for r in te:
                x=torch.tensor(r['fx'][None]).transpose(1,2).to(DEV)
                pr=LS.resample(torch.sigmoid(clf(x,'frame').squeeze(1))[0].cpu().numpy(),r['T'])
                Pc[r['video']]=LS.span_from_prob(pr,best_thr)
                Pt[r['video']]=LS.geo_span(r['fused'],**TUN)
        el=time.time()-t0
        print(f"  [{fi+1}/{len(pids)}] {held}: {len(te)} reps thr={best_thr:.2f} "
              f"({el:.0f}s, ~{el/(fi+1)*(len(pids)-fi-1):.0f}s left)",flush=True)
    tsp={r['video']:r['tsp'] for r in reps}; Tn={r['video']:r['T'] for r in reps}
    common=[k for k in Pc if k in Pt]
    def stats(P):
        de=[];bn=[]
        for k in common:
            d,b=LS.errs(tsp[k],P[k],Tn[k]); de.append(d); bn.append(b)
        de=np.array(de);bn=np.array(bn)
        return de,bn
    dec,bnc=stats(Pc); det,bnt=stats(Pt)
    print(f"\n=== RICH clf ({cin}ch) vs tuned gate, LOPO {len(common)} reps ===")
    print(f"  {'method':<12}{'|dur|mean':>10}{'p50':>7}{'p90':>7}{'p95':>7}{'bnd_mean':>10}{'bnd_p90':>9}")
    for nm,de,bn in [('tuned',det,bnt),('rich_clf',dec,bnc)]:
        print(f"  {nm:<12}{de.mean():>10.0f}{np.percentile(de,50):>7.0f}{np.percentile(de,90):>7.0f}"
              f"{np.percentile(de,95):>7.0f}{bn.mean():>10.0f}{np.percentile(bn,90):>9.0f}",flush=True)
    print(f"\n  ref: OLD clf (6ch) was mean 161 / p50 100 / p95 512 / bnd 92")
    # tail: worst-under-tuned reps, old vs rich
    order=np.argsort(-det)[:14]
    print(f"\n  === worst-under-tuned reps: |dur| tuned -> rich_clf ===")
    for i in order:
        k=common[i]; a_='better' if dec[i]<det[i]-1 else('WORSE' if dec[i]>det[i]+1 else 'same')
        print(f"    {k[:38]:<40}{det[i]:>6.0f} ->{dec[i]:>6.0f}  {a_}",flush=True)
    better=(dec<det-1).sum(); worse=(dec>det+1).sum()
    print(f"\n  per-rep: rich_clf better {better}, worse {worse}, tie {len(common)-better-worse}")
    json.dump({'n':len(common),'cin':int(cin),
               'rich_clf':{'mean':float(dec.mean()),'p50':float(np.percentile(dec,50)),
                           'p90':float(np.percentile(dec,90)),'p95':float(np.percentile(dec,95)),
                           'p99':float(np.percentile(dec,99)),'max':float(dec.max()),
                           'bnd_mean':float(bnc.mean())},
               'tuned':{'mean':float(det.mean()),'p95':float(np.percentile(det,95)),
                        'p99':float(np.percentile(det,99)),'max':float(det.max()),
                        'bnd_mean':float(bnt.mean())},
               'perrep':{common[i]:[float(det[i]),float(dec[i])] for i in range(len(common))}},
              open('experiments/drink_study/cache/learn_seg_clf.json','w'),indent=2)
    print("\nwrote cache/learn_seg_clf.json",flush=True)

if __name__=='__main__': main()
