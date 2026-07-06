"""Self-distillation: train a student on the REFILL student's own detections,
through the SAME 3D-gate + reproject-fill filter (just student dets, not teacher).

Why: the refill student detects on ~99% of frames incl. the drink-at-mouth phase
the teacher-based labels skipped (tri_rate 0.989 on P01). Distilling its
3D-consensus-filtered detections gives denser, mouth-inclusive labels. Train 2
epochs, compare to the refill parent.

Reuses run_clean3d_fill's filter/consensus/eval; only the detector + cache key change.
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
import run_clean3d_fill as F
from metrics import train_with_metrics
from agreement import detect_rep

PARENT = "experiments/drink_study/runs/pscale_1_clean3d_refill/weights/best_3df1.pt"
CFG = "pscale_1_selfdistill"
CACHE = F.CACHE


def student_dets(stem: str):
    """Refill student detections on a P01 rep (cached). Single-class cup model so
    classes=None. Same centroid format as teacher_dets."""
    cf = CACHE / f"P01_{stem}__selfdistill_parent__c0.25.json"
    if cf.exists():
        d = json.loads(cf.read_text())
        return {c: [tuple(x) if x else None for x in v] for c, v in d.items()}
    from ultralytics import YOLO
    pdir = F.CLIPS_ROOT / "P01"
    rep = {f"cam_{c}": pdir / f"{stem}.{c}.mp4" for c in range(1, 11)
           if f"cam_{c}" in F.calib and (pdir / f"{stem}.{c}.mp4").exists()}
    print(f"  refill-student detect on {stem} (all cams) ...", flush=True)
    dets = detect_rep(YOLO(PARENT), rep, F.CONF, None, verbose=True)
    cf.write_text(json.dumps(dets))
    return {c: [tuple(x) if x else None for x in v] for c, v in dets.items()}


def main():
    out_root = run.DSROOT / "_labelcache" / "selfdistill_rf"
    pairs, stats, per_cam = _build(out_root)
    print(f"\n[{CFG}] label pool: kept_real={stats['kept_real']} filled={stats['filled']} "
          f"refilled={stats.get('refilled',0)} dropped={stats['dropped']}", flush=True)
    for c in sorted(per_cam, key=lambda k: int(k.split('_')[1])):
        d = per_cam[c]
        print(f"  {c:>7}: real={d['real']:>5} fill={d['fill']:>5} refill={d.get('refill',0):>5} drop={d['drop']:>5}", flush=True)

    data_yaml, n = run.assemble_dataset(CFG, pairs)
    print(f"training on {n} frames", flush=True)
    test_dir = run.STAGE / "percam_eval"
    run.stage_clips(run.TEST, 1, run.ALL_CAMS, "right", test_dir)
    EPOCHS = int(__import__("os").environ.get("SD_EPOCHS", "4"))
    best, _ = train_with_metrics(
        data_yaml, run.STUDENT, str(run.RUNS.resolve()), CFG, test_dir,
        max_epochs=EPOCHS, long_run=True, save_period=1,   # exact N epochs, keep all
        config={"cfg_id": CFG, "parent": "pscale_1_clean3d_refill", "epochs": EPOCHS})

    rec = F.per_cam_eval(best, test_dir)
    prec3d = F.precision_3d(best, test_dir)
    cams = sorted(rec, key=lambda k: int(k.replace("cam", "")))
    m = lambda d: float(np.mean([d.get(c, 0) for c in cams]))
    print("\n=== SELF-DISTILLED student (held-out) ===", flush=True)
    print(f"mean recall={m(rec):.3f}  cam10={rec.get('cam10',0):.2f}  3D-prec={prec3d:.3f}", flush=True)
    (CACHE / "percam" / f"{CFG}_recall.json").write_text(json.dumps(rec, indent=2))
    (CACHE / "selfdistill_result.json").write_text(json.dumps(
        {"mean_recall": m(rec), "cam10": rec.get("cam10", 0), "precision_3d": prec3d,
         "label_stats": stats}, indent=2))
    print("SELFDISTILL_DONE", flush=True)


def _build(out_root):
    import cv2
    from kalman_3d import triangulate_dlt, project
    pairs = []
    stats = {"kept_real": 0, "filled": 0, "refilled": 0, "dropped": 0}
    per_cam = {f"cam_{c}": {"real": 0, "fill": 0, "refill": 0, "drop": 0} for c in range(1, 11)}
    done = out_root / ".done"
    if done.exists():
        meta = json.loads(done.read_text())
        pairs = [(out_root / r, out_root / r.replace(".jpg", ".txt")) for r in meta["pairs"]]
        print(f"[{CFG}] reusing labelcache ({len(pairs)} pairs)", flush=True)
        return pairs, meta["stats"], meta["per_cam"]
    for stem in F.REPS:
        dets = student_dets(stem); dets = {c: v for c, v in dets.items() if c in F.calib}
        cams = sorted(dets, key=lambda k: int(k.split('_')[1])); n = min(len(v) for v in dets.values())
        caps = {c: cv2.VideoCapture(str(F.CLIPS_ROOT / "P01" / f"{stem}.{int(c.split('_')[1])}.mp4")) for c in cams}
        for fr in range(n):
            obs = {c: dets[c][fr] for c in cams if dets[c][fr] is not None}
            X, kept = F.consensus(obs)
            if X is None:
                for c in obs:
                    per_cam[c]["drop"] += 1; stats["dropped"] += 1
                continue
            for c in cams:
                c0, r = F.apparent_radius_px(c, X)
                if c in obs and c in kept:
                    z = dets[c][fr]; line = F._rect_poly_line(z[0], z[1], r); kind = "real"
                elif c in obs:
                    line = F.bbox_label(c0, r); kind = "refill"
                else:
                    line = F.bbox_label(c0, r); kind = "fill"
                if line is None:
                    per_cam[c]["drop"] += 1; stats["dropped"] += 1; continue
                cap = caps[c]; cap.set(cv2.CAP_PROP_POS_FRAMES, fr); ok, frame = cap.read()
                if not ok:
                    continue
                cdir = out_root / f"{stem}.{int(c.split('_')[1])}"; cdir.mkdir(parents=True, exist_ok=True)
                sf = f"{stem}.{int(c.split('_')[1])}_f{fr:05d}"
                ip, lp = cdir / f"{sf}.jpg", cdir / f"{sf}.txt"
                cv2.imwrite(str(ip), frame); lp.write_text(line)
                pairs.append((ip, lp)); per_cam[c][kind] += 1
                stats[{"real": "kept_real", "fill": "filled", "refill": "refilled"}[kind]] += 1
        for cap in caps.values():
            cap.release()
    done.write_text(json.dumps({"stats": stats, "per_cam": per_cam,
                                "pairs": [str(ip.relative_to(out_root)) for ip, _ in pairs]}))
    return pairs, stats, per_cam


if __name__ == "__main__":
    main()
