"""Does the MOUTH-based truth vindicate the hybrid on the DISAGREE reps?

For each video rep: pull the paired C3D + sync lag from cache/qtm_align.json, compute the
mouth-based dwell (mouth_dwell) at the mocap rate, map it onto the 60Hz video-track index
(resample + lag), and compare against the OLD speed-proxy truth, the TUNED gate, and the
HYBRID learned segmenter. Prints each span in video frames so you can see whether the
hybrid's wider span is actually closer to the cup-at-mouth truth than the speed proxy was.

    python validate_mouth_vs_hybrid.py 151359 105712 110110   # the 3 DISAGREE reps
    python validate_mouth_vs_hybrid.py                          # all reps we have a pairing for
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import learn_seg as LS
import segment_cup_only as S           # noqa: F401 (via LS)
from mouth_dwell import dwell_truth
import glob

HZ = 60.0
MOCAP_HZ = 100.0
from _paths import CACHE as _C
AL = json.load(open(_C / "qtm_align.json"))
LF = _C / "lopo_fused"


def _align_index():
    """video-stem -> rep dict (c3d, lag, sync_corr, n)."""
    out = {}
    for p in AL.values():
        if isinstance(p, dict) and p.get("ok"):
            for r in p["reps"]:
                out[r["video"]] = r
    return out


def _fused_true(video):
    """(fused, true) 60Hz tracks for a video rep from the lopo_fused cache."""
    hits = [f for f in glob.glob(str(LF / "*.npz"))
            if video in str(np.load(f, allow_pickle=True)["video"])]
    if not hits:
        return None, None
    d = np.load(hits[0], allow_pickle=True)
    return np.asarray(d["fused"], float), np.asarray(d["true"], float)


def _mouth_span_on_track(c3d, lag, n_track):
    """Mouth dwell mapped onto an n_track-long 60Hz video index using the sync `lag`.
    lag is in VIDEO frames (video[lag:] aligns to mocap[0:], per qtm_align)."""
    dw = dwell_truth(c3d)
    if dw.span is None:
        return None, dw
    # mocap-frame span -> seconds -> video frames, then shift by +lag (video leads by lag)
    s = dw.span[0] / dw.rate * HZ + lag
    e = dw.span[1] / dw.rate * HZ + lag
    s, e = int(round(s)), int(round(e))
    s, e = max(0, s), min(n_track, e)
    return ((s, e) if e > s else None), dw


def _hybrid_span(video):
    try:
        import render_dwell_compare as RD
        reps = RD.build(); byv = {r["video"]: r for r in reps}
        cin = reps[0]["fx"].shape[1]
        full = [k for k in byv if video in k or k in video]
        return RD.hybrid_span(reps, byv, cin, full[0]) if full else None
    except Exception as e:
        print(f"  (hybrid skipped: {e})", flush=True)
        return None


def _dur(sp):
    return (sp[1] - sp[0]) / HZ * 1000 if sp else 0.0


def main():
    want = sys.argv[1:]
    idx = _align_index()
    stems = [v for v in idx if (not want) or any(w in v for w in want)]
    if not stems:
        raise SystemExit("no reps matched")
    print(f"comparing {len(stems)} rep(s). spans in VIDEO frames @60Hz; dur in ms.\n", flush=True)
    for video in sorted(stems):
        r = idx[video]
        fused, true = _fused_true(video)
        if fused is None:
            print(f"{video}: no lopo_fused cache; skip"); continue
        n = len(fused)
        sp_speed = LS.geo_span(true)                       # old mocap speed-proxy truth
        sp_tuned = LS.geo_span(fused, **LS.TUN)            # production gate on video track
        sp_mouth, dw = _mouth_span_on_track(r["c3d"], r["lag"], n)
        print(f"=== {video}  (c3d {r['c3d']}, lag {r['lag']}, corr {r['sync_corr']:.2f}) ===", flush=True)
        print(f"  training hybrid fold ...", flush=True)
        sp_hyb = _hybrid_span(video)
        print(f"  MOUTH truth (NEW):  {sp_mouth}   dur {_dur(sp_mouth):.0f}ms", flush=True)
        print(f"  speed-proxy (OLD):  {sp_speed}   dur {_dur(sp_speed):.0f}ms", flush=True)
        print(f"  TUNED gate:         {sp_tuned}   dur {_dur(sp_tuned):.0f}ms", flush=True)
        print(f"  HYBRID:             {sp_hyb}   dur {_dur(sp_hyb):.0f}ms", flush=True)
        if sp_mouth and sp_hyb and sp_speed:
            e_speed = abs(_dur(sp_speed) - _dur(sp_mouth))
            e_hyb = abs(_dur(sp_hyb) - _dur(sp_mouth))
            verdict = "HYBRID closer to mouth-truth" if e_hyb < e_speed else "speed-proxy closer"
            print(f"  vs MOUTH truth: |speed-mouth|={e_speed:.0f}ms  |hybrid-mouth|={e_hyb:.0f}ms"
                  f"  -> {verdict}", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
