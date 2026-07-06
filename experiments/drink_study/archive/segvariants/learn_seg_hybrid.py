"""HYBRID segmenter clf: 13ch fused-track features (learn_seg.build_rep, which solved P20
via 3D velocity DIRECTION) + 4 OCCLUSION/agreement channels from the detection tensor
(ncams, occ, mpx, present -- which crushed the P10/P16 occlusion tail in learn_seg_det).

Rationale (measured): fused-track kinematics = RELIABLE median input; raw detection state =
noisy but DIRECTLY marks the cup-at-mouth dwell via occlusion. Combine: keep reliable
kinematics, ADD the occlusion cue. LOPO, clf head, truth = mocap dwell (cache/lopo_fused).

    python experiments/drink_study/learn_seg_hybrid.py [--epochs 250]
Run from repo root; GPU. Writes cache/learn_seg_hybrid.json
"""
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import sys, glob, json, time, argparse, numpy as np, torch, torch.nn as nn
sys.path.insert(0,'experiments/drink_study')
import learn_seg as LS, learn_seq_kf as M, tune_interp as T
HZ=LS.HZ; SEQ_L=LS.SEQ; DEV=LS.DEV; TUN=LS.TUN
LF=LS.CACHE
# occlusion/agreement subset of build_seq's 17ch: mpx[14], ncams[15], occ[16], present[3]
OCC_IDX=[3,14,15,16]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=250); a=ap.parse_args()
    print(f"device {DEV}; building detection occlusion feats (build_seq)...",flush=True)
    # detection occlusion feats per video (SEQ grid) via build_seq
    occ={}
    for rep in T._reps():
        r=M.build_seq(rep)
        if r is not None:
            occ[rep['video']]=r['feats'][:,OCC_IDX]        # (SEQ_seq, 4)
    print(f"{len(occ)} reps have detection feats; loading fused cache...",flush=True)
    # fused-track 13ch feats + truth from lopo_fused cache, matched by video
    reps=[]
    for f in sorted(glob.glob(str(LF/'*.npz'))):
        d=np.load(f,allow_pickle=True); v=str(d['video'])
        if v not in occ: continue
        r=LS.build_rep(d)
        if r['tsp'] is None: continue
        fx13=LS.resample(r['feats'], SEQ_L)                # (SEQ_L,13)
        fxocc=LS.resample(occ[v], SEQ_L)                   # (SEQ_L,4)
        r['fx']=np.concatenate([fx13, fxocc],1).astype(np.float32)   # (SEQ_L,17)
        r['mx']=LS.resample(r['tmask'], SEQ_L).astype(np.float32)
        reps.append(r)
    pids=sorted({r['pid'] for r in reps}); cin=reps[0]['fx'].shape[1]
    print(f"{len(reps)} reps, {len(pids)} folds, {cin}ch (13 fused + 4 occ)",flush=True)

    Ph={}; Pt={}; t0=time.time()
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
                Ph[r['video']]=LS.span_from_prob(pr,best_thr)
                Pt[r['video']]=LS.geo_span(r['fused'],**TUN)
        el=time.time()-t0
        print(f"  [{fi+1}/{len(pids)}] {held}: {len(te)} reps thr={best_thr:.2f} "
              f"({el:.0f}s, ~{el/(fi+1)*(len(pids)-fi-1):.0f}s left)",flush=True)
    tsp={r['video']:r['tsp'] for r in reps}; Tn={r['video']:r['T'] for r in reps}
    common=[k for k in Ph if k in Pt]
    def st(P):
        de=[];bn=[]
        for k in common:
            d,b=LS.errs(tsp[k],P[k],Tn[k]); de.append(d);bn.append(b)
        return np.array(de),np.array(bn)
    dh,bh=st(Ph); dt,bt=st(Pt)
    print(f"\n=== HYBRID clf ({cin}ch=13fused+4occ) vs tuned, LOPO {len(common)} reps ===")
    print(f"  {'method':<10}{'mean':>6}{'p50':>6}{'p90':>6}{'p95':>6}{'p99':>6}{'max':>7}{'bnd':>6}")
    for nm,de,bn in [('tuned',dt,bt),('hybrid',dh,bh)]:
        print(f"  {nm:<10}{de.mean():>6.0f}{np.percentile(de,50):>6.0f}{np.percentile(de,90):>6.0f}"
              f"{np.percentile(de,95):>6.0f}{np.percentile(de,99):>6.0f}{de.max():>7.0f}{bn.mean():>6.0f}",flush=True)
    print(f"  ref p99/max: tuned=1056/3617  6ch=789/3467")
    order=np.argsort(-dt)[:14]
    print(f"\n  === worst-under-tuned: |dur| tuned -> hybrid ===")
    for i in order:
        k=common[i]; f='better' if dh[i]<dt[i]-1 else('WORSE' if dh[i]>dt[i]+1 else 'same')
        print(f"    {k[:36]:<38}{dt[i]:>6.0f} ->{dh[i]:>6.0f}  {f}",flush=True)
    better=(dh<dt-1).sum(); worse=(dh>dt+1).sum()
    print(f"\n  per-rep vs tuned: hybrid better {better}, worse {worse}, tie {len(common)-better-worse}")
    # BIGGEST REGRESSIONS: reps hybrid made most worse THAN TUNED (dh - dt largest)
    reg=dh-dt; ro=np.argsort(-reg)[:14]
    print(f"\n  === biggest REGRESSIONS vs tuned (tuned -> hybrid, worst 14) ===")
    for i in ro:
        print(f"    {common[i][:36]:<38}{dt[i]:>6.0f} ->{dh[i]:>6.0f}   +{reg[i]:.0f}ms worse",flush=True)
    print(f"  regressions >200ms worse: {(reg>200).sum()};  >500ms: {(reg>500).sum()}")
    json.dump({'n':len(common),'cin':int(cin),
               'hybrid':{'mean':float(dh.mean()),'p50':float(np.percentile(dh,50)),
                         'p95':float(np.percentile(dh,95)),'p99':float(np.percentile(dh,99)),
                         'max':float(dh.max()),'bnd_mean':float(bh.mean())},
               'perrep':{common[i]:[float(dt[i]),float(dh[i])] for i in range(len(common))}},
              open('experiments/drink_study/cache/learn_seg_hybrid.json','w'),indent=2)
    print("\nwrote cache/learn_seg_hybrid.json",flush=True)

if __name__=='__main__': main()
