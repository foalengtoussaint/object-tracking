"""Post-hoc re-score the refill student's per-epoch checkpoints (no retrain).

For each saved epochN.pt computes, all with the SAME eval_gate presence logic:
  - heldout_recall / heldout_f1  : on P06/P19/P23 (2D presence F1, the old stop signal)
  - heldout_prec3d / heldout_f1_3d : PRECISION via OUR 3D filter (fraction of the
        student's detections that land in the >=3-cam consensus); 3D-F1 = harmonic
        mean of heldout_recall and that 3D precision -> the metric we actually care
        about. We RE-PICK best by this.
  - train_f1                     : on P01 reps NOT trained on (reps[2:5]) -> the
        train-participant fit, for the train-vs-heldout divergence (overfit) plot.

Writes runs/pscale_1_clean3d_refill/eval_by_epoch.json (with all of the above) and
prints which epoch is best by 3D-F1 vs by 2D-F1 (they can differ -> tells us if the
2D early-stop picked the wrong checkpoint).
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
from metrics import eval_gate, _f1
from run_clean3d_fill import precision_3d

RUN = Path("experiments/drink_study/runs/pscale_1_clean3d_refill")
CONF = 0.25
# epochs to also score on the TRAIN-held-back set (for the overfit curve). The
# rest are scored held-out only -> just enough to find the 3D-F1-best epoch
# without paying 2x the frames on every checkpoint.
TRAIN_F1_EPOCHS = {0, 2, 5, 8, 11}
# resume: skip epochs already scored in a previous (interrupted) run
PREV = RUN / "eval_by_epoch.json"


def main():
    # held-out (generalization) clips
    heldout = run.STAGE / "percam_eval"
    run.stage_clips(run.TEST, 1, run.ALL_CAMS, "right", heldout)
    # train-participant held-BACK clips: P01 reps the student did NOT train on
    train_eval = run.STAGE / "p01_heldback_eval"
    reps = run.reps_of("P01", "right")
    # stage P01 reps [2:5] (trained on [:2]) across all cams
    train_eval.mkdir(parents=True, exist_ok=True)
    from _paths import CLIPS_ROOT
    import shutil
    for f in train_eval.glob("*.mp4"):
        f.unlink()
    for stem in reps[2:5]:
        for cam in run.ALL_CAMS:
            src = CLIPS_ROOT / "P01" / f"{stem}.{cam}.mp4"
            if src.exists():
                (train_eval / src.name).symlink_to(src.resolve())

    ckpts = {}
    for p in sorted((RUN / "weights").glob("epoch*.pt")):
        d = "".join(c for c in p.stem if c.isdigit())
        if d:
            ckpts[int(d)] = p

    # resume: reuse any epochs already in eval_by_epoch.json
    ee = {}
    if PREV.exists():
        prev = json.loads(PREV.read_text()).get("by_epoch", {})
        ee = {int(k): v for k, v in prev.items()}
        print(f"resuming: {len(ee)} epochs already scored", flush=True)
    todo = [e for e in sorted(ckpts) if e not in ee]
    print(f"scoring {len(todo)} remaining checkpoints (train-F1 only on {sorted(TRAIN_F1_EPOCHS)})", flush=True)

    for ep in todo:
        p = ckpts[ep]
        ho = eval_gate(str(p), heldout, CONF)
        rec, pl = ho.metrics["overall_recall"], ho.metrics["overall_p_loose"]
        prec3d = precision_3d(str(p), heldout)              # consensus-inlier fraction
        row = {"heldout_recall": round(rec, 4), "heldout_p_loose": round(pl, 4),
               "heldout_f1": round(_f1(rec, pl), 4),
               "heldout_prec3d": round(prec3d, 4),
               "heldout_f1_3d": round(_f1(rec, prec3d), 4)}
        if ep in TRAIN_F1_EPOCHS:                           # train-F1 only where needed
            tr = eval_gate(str(p), train_eval, CONF)
            trec, tpl = tr.metrics["overall_recall"], tr.metrics["overall_p_loose"]
            row.update({"train_recall": round(trec, 4), "train_f1": round(_f1(trec, tpl), 4)})
        ee[ep] = row
        print(f"  ep{ep:>2}: heldout 2D-F1={row['heldout_f1']:.3f}  3D-prec={prec3d:.3f}  "
              f"3D-F1={row['heldout_f1_3d']:.3f}  train-F1={row.get('train_f1','-')}", flush=True)
        # checkpoint-safe: write after every epoch so a kill never loses progress
        (RUN / "eval_by_epoch.json").write_text(json.dumps(
            {"by_epoch": {str(k): v for k, v in ee.items()}}, indent=2))

    best_2d = max(ee, key=lambda e: ee[e]["heldout_f1"])
    best_3d = max(ee, key=lambda e: ee[e]["heldout_f1_3d"])
    print(f"\nBEST by 2D-F1: ep{best_2d} ({ee[best_2d]['heldout_f1']:.3f})", flush=True)
    print(f"BEST by 3D-F1: ep{best_3d} ({ee[best_3d]['heldout_f1_3d']:.3f})", flush=True)
    if best_2d != best_3d:
        print(f"  -> they DIFFER: the 2D early-stop would pick ep{best_2d}, but the "
              f"3D-best is ep{best_3d}. Using ep{best_3d} as 'best'.", flush=True)
    else:
        print("  -> same epoch is best by both metrics.", flush=True)

    # promote the 3D-F1-best checkpoint to best_3d.pt (keep ultralytics' best.pt too)
    import shutil
    src = RUN / "weights" / f"epoch{best_3d}.pt"
    shutil.copy(src, RUN / "weights" / "best_3df1.pt")
    ee_out = {"by_epoch": ee, "best_2d_f1_epoch": best_2d, "best_3d_f1_epoch": best_3d}
    (RUN / "eval_by_epoch.json").write_text(json.dumps(ee_out, indent=2))
    print(f"\nwrote {RUN / 'eval_by_epoch.json'}  (best_3df1.pt = epoch{best_3d})", flush=True)
    print("RESCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
