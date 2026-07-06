"""PROTOTYPE (go/no-go filter, NOT a result): does a self-ATTENTION segmenter beat the
dilated TCN on the COMBO feature set (proxy + distances) — the set where the TCN BLENDED
and lost to proxy-alone? Hypothesis: attention can weigh 'which channel to trust per frame'
(proxy normally, distances when the proxy looks off), which fixed-dilation convs cannot.

Single held-out split only (fast). NOTE: single-split wins have EVAPORATED before in this
project — treat purely as a filter. If attention clears here, run the full LOPO.

    python experiments/drink_study/proto_attn.py [--epochs 250] [--held P10]
"""
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import sys, argparse, numpy as np, torch, torch.nn as nn
sys.path.insert(0, 'experiments/drink_study')
import learn_seg_mouth as LM
import learn_seg as LS
SEQ = LS.SEQ; DEV = LS.DEV


class AttnSeg(nn.Module):
    """Transformer encoder over the 256-frame sequence -> per-frame logit. Same interface
    as LS.TCN: forward(x=(B,cin,T)) -> (B,1,T)."""
    def __init__(self, cin, d=64, heads=4, layers=3, p=0.1):
        super().__init__()
        self.inp = nn.Conv1d(cin, d, 1)                          # per-frame input projection
        pe = self._sinusoid(SEQ, d)
        self.register_buffer("pe", pe)                           # (1,d,T)
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=4 * d,
                                         dropout=p, batch_first=True, activation="gelu")
        self.body = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Linear(d, 1)

    @staticmethod
    def _sinusoid(T, d):
        pos = np.arange(T)[:, None]; i = np.arange(d)[None, :]
        ang = pos / np.power(10000, (2 * (i // 2)) / d)
        pe = np.zeros((T, d)); pe[:, 0::2] = np.sin(ang[:, 0::2]); pe[:, 1::2] = np.cos(ang[:, 1::2])
        return torch.tensor(pe.T[None], dtype=torch.float32)     # (1,d,T)

    def forward(self, x, mode="frame"):
        h = self.inp(x) + self.pe                                # (B,d,T)
        h = self.body(h.transpose(1, 2))                         # (B,T,d)
        return self.head(h).transpose(1, 2)                      # (B,1,T)


class TCNAttn(nn.Module):
    """The existing dilated TCN body + ONE self-attention layer before the head — the
    smaller idea: keep the conv stack, just give it a channel/time-selection layer so it
    can 'attend' to proxy-vs-distance per frame without a full transformer. Same interface."""
    def __init__(self, cin, ch=48, layers=5, heads=4, p=0.1):
        super().__init__()
        L = []; c = cin
        for i in range(layers):
            dl = 2 ** i
            L += [nn.Conv1d(c, ch, 3, padding=dl, dilation=dl), nn.ReLU(), nn.Dropout(p)]
            c = ch
        self.body = nn.Sequential(*L)
        self.attn = nn.MultiheadAttention(ch, heads, dropout=p, batch_first=True)
        self.norm = nn.LayerNorm(ch)
        self.head = nn.Conv1d(ch, 1, 1)

    def forward(self, x, mode="frame"):
        h = self.body(x).transpose(1, 2)                        # (B,T,ch)
        a, _ = self.attn(h, h, h)
        h = self.norm(h + a).transpose(1, 2)                    # (B,ch,T) residual attn
        return self.head(h)                                     # (B,1,T)


def train_eval(reps, held, fxkey, epochs, model_cls):
    trn = [r for r in reps if r['pid'] != held]
    te = [r for r in reps if r['pid'] == held]
    Xtr = torch.tensor(np.stack([r[fxkey] for r in trn])).transpose(1, 2).to(DEV)
    Mtr = torch.tensor(np.stack([r['mx'] for r in trn])).to(DEV)
    net = model_cls(Xtr.shape[1]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    pos = Mtr.mean().clamp(1e-3, 1 - 1e-3); w = ((1 - pos) / pos).item()
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w, device=DEV)); net.train()
    for _ in range(epochs):
        opt.zero_grad(); lossf(net(Xtr, 'frame').squeeze(1), Mtr).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        ptr = torch.sigmoid(net(Xtr, 'frame').squeeze(1)).cpu().numpy()
        best_thr, best = 0.5, 1e9
        for thr in np.arange(0.15, 0.96, 0.05):
            des = [LS.errs(r['tsp'], LS.span_from_prob(LS.resample(ptr[k], r['T']), thr), r['T'])[0]
                   for k, r in enumerate(trn)]
            if np.mean(des) < best:
                best = np.mean(des); best_thr = thr
        des = []
        for r in te:
            x = torch.tensor(r[fxkey][None]).transpose(1, 2).to(DEV)
            pr = LS.resample(torch.sigmoid(net(x, 'frame').squeeze(1))[0].cpu().numpy(), r['T'])
            des.append(LS.errs(r['tsp'], LS.span_from_prob(pr, best_thr), r['T'])[0])
    return np.array(des), best_thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=250)
    ap.add_argument('--held', default='P10')     # a hard fold (proxy failures live here)
    a = ap.parse_args()
    reps = LM.build()
    print(f"single split: held-out {a.held}; TCN vs TCN+attn vs Attn on proxy21 & combo27 "
          f"({a.epochs} ep)\n", flush=True)
    rows = []
    for fxkey, name in [('fx_prox', 'proxy21'), ('fx_combo', 'combo27')]:
        for model_cls, mname in [(LS.TCN, 'TCN'), (TCNAttn, 'TCN+attn'), (AttnSeg, 'Attn')]:
            t0 = __import__('time').time()
            de, thr = train_eval(reps, a.held, fxkey, a.epochs, model_cls)
            print(f"    ...{name}/{mname} done in {__import__('time').time()-t0:.0f}s", flush=True)
            print(f"  {name:<9} {mname:<5} thr={thr:.2f}  mean={de.mean():6.0f}  "
                  f"p50={np.percentile(de,50):5.0f}  p90={np.percentile(de,90):5.0f}  "
                  f"max={de.max():5.0f}  (n={len(de)})", flush=True)
            rows.append((name, mname, de))
    # the key question: on COMBO, do the attn models beat the plain TCN? and beat TCN-proxy?
    d = {(n, m): de for n, m, de in rows}
    ref = d[('proxy21', 'TCN')].mean()      # current best design (proxy alone + plain TCN)
    print(f"\n  GO/NO-GO (single split — filter only):")
    print(f"    reference: TCN proxy21 mean = {ref:.0f}")
    for m in ['TCN', 'TCN+attn', 'Attn']:
        print(f"    combo27  {m:<9} mean = {d[('combo27', m)].mean():.0f}"
              f"{'   <-- beats reference' if d[('combo27', m)].mean() < ref else ''}", flush=True)


if __name__ == '__main__':
    main()
