"""Does a 2-epoch refill student match the ep6 'best'? Score epoch1/epoch2/ep6 on
the FULL held-out metrics (per-cam recall incl. cam_10 + gated 3D precision), so the
'we only need 2 epochs' claim is checked on the numbers that matter, not eyeballed.
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import run
from run_clean3d_fill import per_cam_eval, precision_3d
from agreement import agreement_eval

W = Path("experiments/drink_study/runs/pscale_1_clean3d_refill/weights")
CKPTS = {"epoch1": W / "epoch1.pt", "epoch2": W / "epoch2.pt", "ep6 (best)": W / "best_3df1.pt"}


def main():
    test_dir = run.STAGE / "percam_eval"
    run.stage_clips(run.TEST, 1, run.ALL_CAMS, "right", test_dir)
    print(f"{'ckpt':<12}{'mean recall':>12}{'cam10':>8}{'3D-prec px':>12}{'tri_rate':>10}", flush=True)
    out = {}
    for name, p in CKPTS.items():
        rec = per_cam_eval(p, test_dir)
        cams = sorted(rec, key=lambda k: int(k.replace("cam", "")))
        mr = float(np.mean([rec[c] for c in cams]))
        a = agreement_eval(str(p), run.TEST, reps=1, hand="right", gated=True)
        out[name] = {"mean_recall": round(mr, 3), "cam10": round(rec.get("cam10", 0), 2),
                     "prec_px": a.get("median_reproj_px"), "tri_rate": a.get("tri_rate")}
        print(f"{name:<12}{mr:>12.3f}{rec.get('cam10',0):>8.2f}"
              f"{a.get('median_reproj_px', float('nan')):>12.2f}{a.get('tri_rate', float('nan')):>10.3f}", flush=True)
    Path("experiments/drink_study/cache/verify_2epoch.json").write_text(json.dumps(out, indent=2))
    print("VERIFY_DONE", flush=True)


if __name__ == "__main__":
    main()
