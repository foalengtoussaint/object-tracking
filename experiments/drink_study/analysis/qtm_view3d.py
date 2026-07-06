"""Interactive 3D viewer for a QTM cup take — the closest thing to QTM's 3D view
we can do on Linux (the .c3d has ONLY the 4 labeled cup markers; the unlabeled-point
pool and 2D camera rays live in the proprietary .qtm project, not exportable).

Shows the 4 labeled markers (cupdl/dr/ul/ur) animating in 3D over a scrubbable
timeline, the rigid cup quad drawn as edges, and flags the defect frames the
pipeline detected (non-physical step / despiked excursion) so you can SEE the glitch
— a marker teleporting, a swap, or the cup quad deforming.

    python qtm_view3d.py P08_0027          # opens the rerun viewer
    python qtm_view3d.py P10_0029 --plotly # static plotly html instead

Rerun controls: drag=orbit, scroll=zoom, timeline at bottom to scrub/play.
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qtm_c3d import load_trial, CUP_MARKERS

# cup quad edges (which markers are adjacent on the rim/base)
EDGES = [("cupdl", "cupdr"), ("cupur", "cupul"), ("cupdl", "cupul"), ("cupdr", "cupur")]
COLORS = {"cupdl": [255, 80, 80], "cupdr": [80, 255, 80],
          "cupul": [80, 160, 255], "cupur": [255, 220, 60]}


def defect_frames(tr, max_step_mm=50.0):
    """Frames flagged as glitchy: centroid step > threshold (the teleports)."""
    c = tr.centroid(despike=False)
    step = np.linalg.norm(np.diff(c, axis=0), axis=1)
    bad = np.zeros(len(c), bool)
    bad[1:] = step > max_step_mm
    bad[:-1] |= step > max_step_mm
    return bad


def view_rerun(stem: str):
    import rerun as rr
    tr = load_trial(stem)
    m = tr.markers                      # (T, M, 3) mm, possibly NaN
    idx = {l: i for i, l in enumerate(tr.labels)}
    q = tr.gt_quality()
    bad = defect_frames(tr)
    print(f"{stem}: {tr.n_frames} frames @ {tr.rate:.0f}Hz  gt_quality={q['reason']}  "
          f"defect_frames={int(bad.sum())}", flush=True)

    rr.init(f"qtm_{stem}", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    # full trajectory of the centroid as a faint static path for context
    cen = tr.centroid(despike=False)
    good = np.isfinite(cen).all(1)
    rr.log("world/centroid_path", rr.LineStrips3D([cen[good]], colors=[120, 120, 120]), static=True)

    for fr in range(tr.n_frames):
        rr.set_time("frame", sequence=fr)
        pts, cols, labels = [], [], []
        for lab in tr.labels:
            p = m[fr, idx[lab]]
            if np.isfinite(p).all():
                pts.append(p); cols.append(COLORS.get(lab, [200, 200, 200])); labels.append(lab)
        if pts:
            rr.log("world/markers", rr.Points3D(pts, colors=cols, labels=labels, radii=6))
        # cup quad edges
        segs = []
        for a, b in EDGES:
            pa, pb = m[fr, idx[a]], m[fr, idx[b]]
            if np.isfinite(pa).all() and np.isfinite(pb).all():
                segs.append([pa, pb])
        if segs:
            ec = [255, 0, 0] if bad[fr] else [180, 180, 180]   # RED quad on a defect frame
            rr.log("world/cup", rr.LineStrips3D(segs, colors=ec))
        # a big red marker flag on defect frames
        if bad[fr]:
            rr.log("world/DEFECT", rr.Points3D([cen[fr]] if np.isfinite(cen[fr]).all() else [],
                                               colors=[255, 0, 0], radii=20))
        else:
            rr.log("world/DEFECT", rr.Clear(recursive=False))
    print("streamed to rerun viewer — scrub the timeline; red quad = defect frame", flush=True)


def view_plotly(stem: str):
    import plotly.graph_objects as go
    tr = load_trial(stem); m = tr.markers; idx = {l: i for i, l in enumerate(tr.labels)}
    bad = defect_frames(tr)
    frames = []
    for fr in range(0, tr.n_frames, 2):
        data = []
        for lab in tr.labels:
            p = m[fr, idx[lab]]
            if np.isfinite(p).all():
                data.append(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers",
                                         marker=dict(size=6, color=f"rgb({','.join(map(str,COLORS[lab]))})"),
                                         name=lab))
        frames.append(go.Frame(data=data, name=str(fr)))
    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(title=f"{stem} (red frames={int(bad.sum())})",
                      updatemenus=[dict(type="buttons", buttons=[dict(label="play", method="animate", args=[None])])])
    from _paths import CACHE as _C
    out = _C / f"_view3d_{stem}.html"
    fig.write_html(str(out)); print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stem")
    ap.add_argument("--plotly", action="store_true")
    args = ap.parse_args()
    (view_plotly if args.plotly else view_rerun)(args.stem)


if __name__ == "__main__":
    main()
