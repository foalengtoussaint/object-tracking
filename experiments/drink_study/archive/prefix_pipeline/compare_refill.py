"""Final 4-way held-out comparison incl. the refill student (best by 3D-F1 = ep6).

Reuses cached precision for raw/drop/reject (heldout_3dprecision.json, unchanged) +
cached per-cam recall JSONs; only the NEW refill student needs scoring. Caches the
refill student's per-clip detections so any future metric re-score is instant.
Writes cache/heldout_3dprecision_refill.json + cache/percam/pscale_1_clean3d_refill_recall.json.
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

CACHE = Path("experiments/drink_study/cache")
BEST = Path("experiments/drink_study/runs/pscale_1_clean3d_refill/weights/best_3df1.pt")


def main():
    test_dir = run.STAGE / "percam_eval"
    run.stage_clips(run.TEST, 1, run.ALL_CAMS, "right", test_dir)

    # refill recall (per_cam_eval caches per-clip presence to PCACHE) + 3D precision
    rec = per_cam_eval(BEST, test_dir)
    (CACHE / "percam" / "pscale_1_clean3d_refill_recall.json").write_text(json.dumps(rec, indent=2))
    a = agreement_eval(str(BEST), run.TEST, reps=1, hand="right", gated=True)
    prec = {"tri_rate": a.get("tri_rate"), "median_px": a.get("median_reproj_px"),
            "cams": a.get("mean_cams_agreeing"), "reps": a.get("reps")}

    # merge with the cached precision for the other 3 (unchanged)
    old = json.loads((CACHE / "heldout_3dprecision.json").read_text())
    old["refill"] = prec
    (CACHE / "heldout_3dprecision_refill.json").write_text(json.dumps(old, indent=2))

    base = json.load(open(CACHE / "percam" / "percam_recall.json"))["pscale_1"]
    no10 = json.load(open(CACHE / "percam" / "pscale_1_no10_recall.json"))
    cl3d = json.load(open(CACHE / "percam" / "pscale_1_clean3d_recall.json"))
    fill = json.load(open(CACHE / "percam" / "pscale_1_clean3d_fill_recall.json"))
    cams = sorted(base, key=lambda k: int(k.replace("cam", "")))
    m = lambda d: float(np.mean([d.get(c, 0) for c in cams]))

    rows = [("raw", base, "raw"), ("drop_cam10", no10, "drop_cam10"),
            ("reject", cl3d, "reject"), ("fill", fill, "fill"), ("refill (NEW)", rec, "refill")]
    print(f"\n{'variant':<14}{'mean recall':>12}{'cam10':>8}{'3D-prec px':>12}{'tri_rate':>10}")
    for name, d, pk in rows:
        p = old.get(pk, {})
        print(f"{name:<14}{m(d):>12.3f}{d.get('cam10',0):>8.2f}"
              f"{p.get('median_px', float('nan')):>12.2f}{p.get('tri_rate', float('nan')):>10.3f}")
    print(f"\ncam10: reject={cl3d.get('cam10',0):.2f} fill={fill.get('cam10',0):.2f} "
          f"refill={rec.get('cam10',0):.2f}  (did reject-then-fill beat reject's 0.51?)")
    print("COMPARE_DONE")


if __name__ == "__main__":
    main()
