"""Segmenter clf on the RAW DETECTION features (same input as the velocity-fill TCN),
instead of hand-derived fused-track kinematics. Does the segmenter do better seeing the
multi-camera detection state (per-cam kept, ncams, occ, mpx agreement, raw consL) --
which directly encodes occlusion, and occlusion spikes when the cup is at the mouth?

Input = learn_seq_kf.build_seq(rep)['feats']  (SEQ,17): consL(3)+present(1)+kept(10)+
mpx(1)+ncams(1)+occ(1).  Truth = mocap dwell mask from cache/lopo_fused (geo_span(true)).
LOPO, clf head only. Compares to the 6ch (161) and 13ch (162) fused-track clfs + tuned (185).

    python experiments/drink_study/learn_seg_det.py [--epochs 250]
Run from repo root; GPU. Per-fold prints. Writes cache/learn_seg_det.json
"""
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import sys, glob, json, time, argparse, numpy as np, torch, torch.nn as nn
sys.path.insert(0,'experiments/drink_study')
import learn_seg as LS, learn_seq_kf as M, tune_interp as T
import segment_cup_only as S
HZ=LS.HZ; SEQ_L=LS.SEQ; DEV=LS.DEV; TUN=LS.TUN
LF_CACHE = LS.CACHE   # lopo_fused dir, for the mocap dwell truth + tuned baseline

def truth_from_cache():
    """video -> (mocap dwell span native-frame, T, fused track) from lopo_fused cache."""
    out={}
    for f in glob.glob(str(LF_CACHE/'*.npz')):
        d=np.load(f,allow_pickle=True); v=str(d['video'])
        ts=LS.geo_span(d['true'])
        if ts is not None:
            out[v]=dict(tsp=ts, T=len(d['true']), fused=d['fused'])
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=250); a=ap.parse_args()
    print(f"device {DEV}; building DETECTION feats via build_seq...",flush=True)
    truth=truth_from_cache()
    reps=[]
    for rep in T._reps():
        r=M.build_seq(rep)
        if r is None: continue
        v=rep['video']
        if v not in truth: continue
        tinfo=truth[v]; Tn=tinfo['T']; tsp=tinfo['tsp']
        # detection feats are on SEQ grid (360). truth dwell is native T. resample feats
        # to native T so span extraction lines up, OR resample truth mask to SEQ. Do SEQ:
        # build a SEQ-grid dwell mask by resampling the native-frame mask.
        m_native=np.zeros(Tn,np.float32); m_native[tsp[0]:tsp[1]]=1.0
        r['fx']=LS.resample(r['feats'], SEQ_L).astype(np.float32)  # (SEQ_L,17) on LS grid
        r['mx']=LS.resample(m_native, SEQ_L).astype(np.float32)    # (SEQ_L,)
        r['tsp']=tsp; r['T']=Tn; r['pid']=v.split('_')[0]; r['video']=v
        r['fused']=tinfo['fused']
        reps.append(r)
    pids=sorted({r['pid'] for r in reps}); cin=reps[0]['fx'].shape[1]
    print(f"{len(reps)} reps, {len(pids)} folds, {cin} DETECTION channels",flush=True)

    Pd={}; Pt={}; t0=time.time()
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
                Pd[r['video']]=LS.span_from_prob(pr,best_thr)
                Pt[r['video']]=LS.geo_span(r['fused'],**TUN)
        el=time.time()-t0
        print(f"  [{fi+1}/{len(pids)}] {held}: {len(te)} reps thr={best_thr:.2f} "
              f"({el:.0f}s, ~{el/(fi+1)*(len(pids)-fi-1):.0f}s left)",flush=True)
    tsp={r['video']:r['tsp'] for r in reps}; Tn={r['video']:r['T'] for r in reps}
    common=[k for k in Pd if k in Pt]
    def st(P):
        de=[];bn=[]
        for k in common:
            d,b=LS.errs(tsp[k],P[k],Tn[k]); de.append(d);bn.append(b)
        return np.array(de),np.array(bn)
    dd,bd=st(Pd); dt,bt=st(Pt)
    print(f"\n=== DETECTION-input clf ({cin}ch) vs tuned gate, LOPO {len(common)} reps ===")
    print(f"  {'method':<12}{'|dur|mean':>10}{'p50':>7}{'p90':>7}{'p95':>7}{'bnd_mean':>10}")
    for nm,de,bn in [('tuned',dt,bt),('det_clf',dd,bd)]:
        print(f"  {nm:<12}{de.mean():>10.0f}{np.percentile(de,50):>7.0f}{np.percentile(de,90):>7.0f}"
              f"{np.percentile(de,95):>7.0f}{bn.mean():>10.0f}",flush=True)
    print(f"  ref: fused-track clf  6ch=161/100/512/92   13ch=162/117/512/94")
    order=np.argsort(-dt)[:14]
    print(f"\n  === worst-under-tuned: |dur| tuned -> det_clf ===")
    for i in order:
        k=common[i]; f='better' if dd[i]<dt[i]-1 else('WORSE' if dd[i]>dt[i]+1 else 'same')
        print(f"    {k[:36]:<38}{dt[i]:>6.0f} ->{dd[i]:>6.0f}  {f}",flush=True)
    better=(dd<dt-1).sum(); worse=(dd>dt+1).sum()
    print(f"\n  per-rep vs tuned: det_clf better {better}, worse {worse}, tie {len(common)-better-worse}")
    json.dump({'n':len(common),'cin':int(cin),
               'det_clf':{'mean':float(dd.mean()),'p50':float(np.percentile(dd,50)),
                          'p95':float(np.percentile(dd,95)),'bnd_mean':float(bd.mean())}},
              open('experiments/drink_study/cache/learn_seg_det.json','w'),indent=2)
    print("\nwrote cache/learn_seg_det.json",flush=True)

if __name__=='__main__': main()
