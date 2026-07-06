"""The model + scoring primitives — small, frozen, copied verbatim from drink_study/learn_seg
(TCN, resample, span_from_prob, errs). Self-contained so the experiment needs nothing external.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

HZ = 60.0
SEQ = 256
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class TCN(nn.Module):
    """Per-frame dilated-conv classifier (ch=48, 5 layers). frame mode -> (B,1,T)."""
    def __init__(self, cin, ch=48, layers=5, nout=1, p=0.1):
        super().__init__()
        L = []; c = cin
        for i in range(layers):
            dl = 2 ** i
            L += [nn.Conv1d(c, ch, 3, padding=dl, dilation=dl), nn.ReLU(), nn.Dropout(p)]
            c = ch
        self.body = nn.Sequential(*L)
        self.head = nn.Conv1d(ch, nout, 1)

    def forward(self, x):
        return self.head(self.body(x))


def resample(a, n_out):
    """(T,) or (T,k) linear resample to n_out frames."""
    a = np.asarray(a, float)
    x0 = np.linspace(0, 1, len(a)); x1 = np.linspace(0, 1, n_out)
    if a.ndim == 1:
        return np.interp(x1, x0, a)
    return np.stack([np.interp(x1, x0, a[:, k]) for k in range(a.shape[1])], 1)


def span_from_prob(prob, thr):
    """Longest run of prob>thr -> (s,e) or None."""
    m = prob > thr
    best = None; i = 0; T = len(m)
    while i < T:
        if m[i]:
            j = i
            while j < T and m[j]:
                j += 1
            if best is None or (j - i) > (best[1] - best[0]):
                best = (i, j)
            i = j
        else:
            i += 1
    return best


def errs(tsp, vsp):
    """|duration error| in ms (miss => whole true dwell as error)."""
    td = (tsp[1] - tsp[0]) / HZ * 1000
    if vsp is None:
        return td
    vd = (vsp[1] - vsp[0]) / HZ * 1000
    return abs(vd - td)
