"""Re-score the saved per-epoch checkpoints of a finished long_run, keeping the
RAW agreement breakdown so we never have to re-detect to get a different stat.

For each runs/<cfg>/weights/epoch*.pt:
  - eval_gate on the held-out clips -> recall / p_loose / f1
  - agreement_eval_full on the agreement participants -> tri_rate, mean & median
    reproj px, mean cams, AND the full per-frame error list per rep.

Writes:
  - eval_by_epoch.json   : compact per-epoch summary (recall/f1/tri/median_px/...)
  - agreement_by_epoch.json : full per-epoch agreement (per_rep + per_frame)
Then replots. Idempotent: rerun anytime; uses only the saved .pt (no retrain).

  python experiments/drink_study/score_checkpoints.py runs/baseline \
      --holdout <eval_clips_dir> --agr P06 --agr-reps 1
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline_lib import eval_gate          # held-out recall / p_loose
from agreement import agreement_eval_full
from metrics import _f1, replot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path, help="runs/<cfg> dir (or its weights/)")
    ap.add_argument("--holdout", type=Path, required=True,
                    help="held-out eval clips dir (for eval_gate recall/f1)")
    ap.add_argument("--agr", nargs="+", required=True,
                    help="agreement participants, e.g. P06")
    ap.add_argument("--agr-reps", type=int, default=1)
    ap.add_argument("--hand", default="right")
    ap.add_argument("--eval-conf", type=float, default=0.25)
    args = ap.parse_args()

    run_dir = args.run_dir
    if (run_dir / "weights").exists():
        wdir = run_dir / "weights"
    else:                                   # they passed the weights/ dir itself
        wdir, run_dir = run_dir, run_dir.parent

    ckpts: dict[int, Path] = {}
    for p in sorted(wdir.glob("epoch*.pt")):
        digits = "".join(c for c in p.stem if c.isdigit())
        if digits:
            ckpts[int(digits)] = p
    if not ckpts:
        raise SystemExit(f"no epoch*.pt under {wdir}")
    print(f"scoring {len(ckpts)} checkpoints in {run_dir.name}", flush=True)

    eval_by_epoch: dict[str, dict] = {}
    agr_by_epoch: dict[str, dict] = {}
    for ep, p in sorted(ckpts.items()):
        r = eval_gate(str(p), args.holdout, args.eval_conf)
        rec, pl = r.metrics["overall_recall"], r.metrics["overall_p_loose"]
        a = agreement_eval_full(str(p), args.agr, args.agr_reps,
                                classes=None, hand=args.hand)
        agr_by_epoch[str(ep)] = a
        eval_by_epoch[str(ep)] = {
            "recall": rec, "p_loose": pl, "f1": _f1(rec, pl),
            "tri_rate": a.get("tri_rate"),
            "median_px": a.get("median_reproj_px"),
            "mean_px": a.get("mean_reproj_px"),
            "cams": a.get("mean_cams_agreeing"),
        }
        print(f"  ep{ep}: recall={rec:.3f} f1={_f1(rec, pl):.3f} "
              f"tri={a.get('tri_rate')} median_px={a.get('median_reproj_px')} "
              f"mean_px={a.get('mean_reproj_px')}", flush=True)

    (run_dir / "eval_by_epoch.json").write_text(json.dumps(eval_by_epoch, indent=2))
    (run_dir / "agreement_by_epoch.json").write_text(json.dumps(agr_by_epoch, indent=2))
    print(f"wrote eval_by_epoch.json + agreement_by_epoch.json in {run_dir}",
          flush=True)
    replot(run_dir)


if __name__ == "__main__":
    main()
