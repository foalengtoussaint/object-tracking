"""Dwell-segmentation video: TRUE (mocap) vs TUNED gate vs HYBRID clf, three timelines
burned on the footage with a moving cursor + a live per-method DRINK/. flag. Lets you
WATCH whether the hybrid's wider dwell window actually matches the real drinking motion
(vs the narrow-floor gate that defines our "truth"). Reuses render_phase_compare's
gpu_decode + best_cam + draw_strip machinery.

    python experiments/drink_study/render_dwell_compare.py P24_P24_drinking_right_20240724_105712 ...
Trains the hybrid LOPO fold for each rep's participant (held-out) to get its dwell.
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import argparse, glob, sys
from pathlib import Path
import numpy as np, cv2, torch, torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_phase_compare as RP        # reuse best_cam, gpu_decode, CLIPS, FPS
import learn_seg as LS, learn_seq_kf as M, tune_interp as T
import gpu_decode

HZ=LS.HZ; SEQ_L=LS.SEQ; DEV=LS.DEV; TUN=LS.TUN; LF=LS.CACHE; OCC=[3,14,15,16]
OUT=Path("experiments/drink_study/cache/dwell_compare"); OUT.mkdir(parents=True, exist_ok=True)
# BGR: true=black-ish, tuned=blue, hybrid=purple; not-drink = dark grey
COL={'true':(60,60,60),'tuned':(200,120,40),'hybrid':(200,90,150)}

def build():
    occ={}
    for rep in T._reps():
        r=M.build_seq(rep)
        if r is not None: occ[rep['video']]=r['feats'][:,OCC]
    reps=[]
    for f in sorted(glob.glob(str(LF/'*.npz'))):
        d=np.load(f,allow_pickle=True); v=str(d['video'])
        if v not in occ: continue
        r=LS.build_rep(d)
        if r['tsp'] is None: continue
        r['fx']=np.concatenate([LS.resample(r['feats'],SEQ_L),LS.resample(occ[v],SEQ_L)],1).astype(np.float32)
        r['mx']=LS.resample(r['tmask'],SEQ_L).astype(np.float32); reps.append(r)
    return reps

def hybrid_span(reps, byv, cin, vid):
    rr=byv[vid]; held=rr['pid']; trn=[r for r in reps if r['pid']!=held]
    X=torch.tensor(np.stack([r['fx'] for r in trn])).transpose(1,2).to(DEV)
    Mt=torch.tensor(np.stack([r['mx'] for r in trn])).to(DEV)
    clf=LS.TCN(cin,nout=1).to(DEV); opt=torch.optim.Adam(clf.parameters(),lr=2e-3,weight_decay=1e-4)
    pos=Mt.mean().clamp(1e-3,1-1e-3); lf=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(((1-pos)/pos).item(),device=DEV)); clf.train()
    for _ in range(250): opt.zero_grad(); lf(clf(X,'frame').squeeze(1),Mt).backward(); opt.step()
    clf.eval()
    with torch.no_grad():
        ptr=torch.sigmoid(clf(X,'frame').squeeze(1)).cpu().numpy()
        bt,bb=0.5,1e9
        for thr in np.arange(0.15,0.96,0.05):
            de=[LS.errs(r['tsp'],LS.span_from_prob(LS.resample(ptr[k],r['T']),thr),r['T'])[0] for k,r in enumerate(trn)]
            if np.mean(de)<bb: bb=np.mean(de); bt=thr
        x=torch.tensor(rr['fx'][None]).transpose(1,2).to(DEV)
        pr=LS.resample(torch.sigmoid(clf(x,'frame').squeeze(1))[0].cpu().numpy(),rr['T'])
    return LS.span_from_prob(pr,bt)

def span_iv(sp, T, key):
    """drink/not-drink intervals for draw_strip from a single dwell span."""
    iv=[]
    if not sp: return [('rest',0,T)]
    if sp[0]>0: iv.append(('rest',0,sp[0]))
    iv.append((key,sp[0],sp[1]))
    if sp[1]<T: iv.append(('rest',sp[1],T))
    return iv

def draw_band(canvas,y,label,sp,T,W,fi,col):
    h=26; x0=110; bw=W-x0-10
    cv2.putText(canvas,label,(4,y+18),0,0.5,(255,255,255),1)
    cv2.rectangle(canvas,(x0,y),(x0+bw,y+h),(45,45,45),-1)
    if sp:
        a=x0+int(sp[0]/T*bw); b=x0+int(sp[1]/T*bw)
        cv2.rectangle(canvas,(a,y),(b,y+h),col,-1)
    cx=x0+int(fi/T*bw); cv2.line(canvas,(cx,y-2),(cx,y+h+2),(255,255,255),2)

def render(vid, reps, byv, cin):
    full=[k for k in byv if vid in k][0]
    d=np.load([x for x in glob.glob(str(LF/'*.npz')) if full in x][0],allow_pickle=True)
    tsp=LS.geo_span(d['true']); tun=LS.geo_span(d['fused'],**TUN)
    hyb=hybrid_span(reps,byv,cin,full)
    stem=byv[full]['video']
    # map to clip via track3d stem field
    import json
    tj=json.loads((RP.TRACK/f"{full}.json").read_text()); clipstem=tj['stem']  # e.g. P24_drinking_right_...
    p=clipstem.split('_')[0]
    cam=RP.best_cam(p, clipstem)  # may fail if det json name differs; fallback cam_3
    cn=cam.split('_')[1] if cam else '3'
    video=RP.CLIPS/p/f"{clipstem}.{cn}.mp4"
    if not video.exists():
        print(f"!! no clip {video}",flush=True); return
    W,H,nv,_=gpu_decode.dims(video)
    T_=min(len(d['true']),nv)
    sf=0.6; vw,vh=int(W*sf),int(H*sf); BAN=118
    outp=OUT/f"{full[:40]}__dwell.mp4"
    wr=cv2.VideoWriter(str(outp),cv2.VideoWriter_fourcc(*"mp4v"),30,(vw,vh+BAN))
    inw=lambda i,sp: 'DRINK' if (sp and sp[0]<=i<sp[1]) else '.'
    for fi,img in enumerate(gpu_decode.frames(video)):
        if fi>=T_: break
        canvas=np.zeros((vh+BAN,vw,3),np.uint8); canvas[BAN:]=cv2.resize(img,(vw,vh))
        draw_band(canvas,6,"TRUE",tsp,T_,vw,fi,COL['true'])
        draw_band(canvas,36,"TUNED",tun,T_,vw,fi,COL['tuned'])
        draw_band(canvas,66,"HYBRID",hyb,T_,vw,fi,COL['hybrid'])
        tags=f"T:{inw(fi,tsp)} tuned:{inw(fi,tun)} H:{inw(fi,hyb)}"
        hilite=(0,255,0) if inw(fi,hyb)=='DRINK' and inw(fi,tsp)=='.' else (255,255,255)
        cv2.putText(canvas,f"t={fi/HZ:4.2f}s  {tags}",(4,BAN-6),0,0.55,hilite,1)
        wr.write(canvas)
    wr.release()
    print(f"wrote {outp}  cam{cn}  TRUE{tsp} tuned{tun} hybrid{hyb}",flush=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("vids",nargs="+"); a=ap.parse_args()
    print("building features...",flush=True)
    reps=build(); byv={r['video']:r for r in reps}; cin=reps[0]['fx'].shape[1]
    for v in a.vids: render(v,reps,byv,cin)

if __name__=="__main__": main()
