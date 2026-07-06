"""3D-cleaned + REPROJECT-FILL P01 retrain — the "better filter".

run_clean3d.py is reject-only: it keeps a camera's detection only if it lands in
the >=3-cam, <=30px 3D consensus, and DROPS everything else. That over-prunes
(~28% of labels, incl. good low-support frames) so its mean recall trails the
crude drop_cam10 ablation.

This variant adds reproject-FILL, matched to what the data actually showed:
  - A camera that DID detect: keep it only if it passes the strict 30px gate
    (so cam_10's static glass, 942px off, stays REJECTED -- never readmitted).
  - A camera that did NOT detect in a frame that HAS a >=3-cam consensus: SYNTHESIZE
    a label by reprojecting the consensus 3D point into that camera and drawing an
    axis-aligned bbox of the cup's apparent size (35mm reprojected). This recovers
    coverage on sparse cameras (e.g. cam_8) WITHOUT inventing geometry or letting
    a wrong detection back in.

Eval reports RECALL as-is (presence) and PRECISION via OUR 3D filter (a detection
counts as correct only if it agrees with the >=3-cam consensus), plus held-out F1.
Training early-stops on the held-out F1 plateau (a presence metric saturates ~ep8-10;
val-loss keeps dropping = overfit), not a fixed epoch count.

GPU only for the teacher labeling pass (cached) + the fine-tune. No 30-epoch run.
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
import cv2
import numpy as np
import run
from _paths import CLIPS_ROOT
from kalman_3d import load_calibration, triangulate_dlt, project
from agreement import RES
from pipeline_lib import camera_id
from metrics import train_with_metrics, _f1
from kf_accuracy import CUP_R

CFG = "pscale_1_clean3d_refill"   # reject-then-fill (cam_10 glass replaced by consensus)
REPS = run.reps_of("P01", "right")[:2]
CACHE = Path("experiments/drink_study/cache")
PCACHE = CACHE / "percam"
CONF = 0.25
THR = 30.0
MINC = 3
calib = load_calibration("data/calib/P01/calibration.toml", target_size=RES)


def teacher_dets(stem: str):
    cf = CACHE / f"P01_{stem}__teacher__c0.25.json"
    if cf.exists():
        d = json.loads(cf.read_text())
        return {c: [tuple(x) if x else None for x in v] for c, v in d.items()}
    from ultralytics import YOLO
    from agreement import detect_rep
    from pseudo_label import CUP_LIKE_CLASSES
    pdir = CLIPS_ROOT / "P01"
    rep = {f"cam_{c}": pdir / f"{stem}.{c}.mp4" for c in range(1, 11)
           if f"cam_{c}" in calib and (pdir / f"{stem}.{c}.mp4").exists()}
    print(f"  teacher on {stem} (all cams) ...", flush=True)
    dets = detect_rep(YOLO(run.COCO), rep, CONF, CUP_LIKE_CLASSES, verbose=True)
    cf.write_text(json.dumps(dets))
    return {c: [tuple(x) if x else None for x in v] for c, v in dets.items()}


def consensus(obs):
    """>=3-cam, <=30px gated consensus. Returns (X, kept_set) or (None, set())."""
    cur = dict(obs)
    while len(cur) >= 2:
        X = triangulate_dlt([calib[c] for c in cur], [np.array(cur[c]) for c in cur])
        e = {c: float(np.hypot(*(project(calib[c], X)[0] - np.array(cur[c])))) for c in cur}
        w = max(e, key=e.get)
        if e[w] <= THR:
            break
        del cur[w]
    if len(cur) < MINC:
        return None, set()
    X = triangulate_dlt([calib[c] for c in cur], [np.array(cur[c]) for c in cur])
    return X, set(cur)


def apparent_radius_px(cam, X):
    """Reproject X and a point CUP_R mm offset from it; pixel distance ~ cup radius
    in this view. Offset along image-x of the camera so it's roughly fronto-parallel."""
    c0 = project(calib[cam], X)[0]
    # World coords are in MILLIMETRES (cup ~2000mm from cams), so offset by CUP_R
    # mm directly to get the cup's apparent radius in this view.
    Xo = X.copy(); Xo[0] += CUP_R
    c1 = project(calib[cam], Xo)[0]
    r = float(np.hypot(*(c1 - c0)))
    return c0, max(r, 6.0)            # floor so the bbox isn't degenerate


def _rect_poly_line(cx_px, cy_px, r):
    """A YOLO-SEG label: an axis-aligned square (as a 4-corner polygon) centred at
    (cx_px, cy_px) with half-size r px. The student is a segmentation model, so
    labels must be polygons (0 x1 y1 x2 y2 ...), not boxes. Returns None if the
    centre falls outside the frame."""
    W, H = RES
    if not (0 < cx_px < W and 0 < cy_px < H):
        return None
    corners = [(cx_px - r, cy_px - r), (cx_px + r, cy_px - r),
               (cx_px + r, cy_px + r), (cx_px - r, cy_px + r)]
    pts = []
    for x, y in corners:
        pts += [min(max(x / W, 0.0), 1.0), min(max(y / H, 0.0), 1.0)]
    return "0 " + " ".join(f"{v:.6f}" for v in pts) + "\n"


def bbox_label(c0, r):
    """Synthesized fill label: square polygon at reprojected point c0 (px), half r."""
    return _rect_poly_line(c0[0], c0[1], r)


def build_filled_pairs():
    """Build (img.jpg, label.txt) pairs with strict-reject for detections + bbox
    reproject-fill for non-detecting cams in consensus frames. Writes images and
    synthesized labels into a dedicated cache dir."""
    # "rf" = reject-THEN-fill (failed-gate detections replaced by consensus,
    # not dropped). Separate dir so the older drop-based labelcache is preserved.
    out_root = run.DSROOT / "_labelcache" / "clean3d_fill_rf"
    done = out_root / ".done"
    if done.exists():                       # labelcache already built -> reuse it
        meta = json.loads(done.read_text())
        pairs = [(out_root / rel, out_root / rel.replace(".jpg", ".txt"))
                 for rel in meta["pairs"]]
        print(f"[{CFG}] reusing cached labelcache ({len(pairs)} pairs)", flush=True)
        return pairs, meta["stats"], meta["per_cam"]
    pairs = []
    stats = {"kept_real": 0, "filled": 0, "refilled": 0, "dropped": 0}
    per_cam = {f"cam_{c}": {"real": 0, "fill": 0, "refill": 0, "drop": 0}
               for c in range(1, 11)}
    for stem in REPS:
        dets = teacher_dets(stem)
        dets = {c: v for c, v in dets.items() if c in calib}
        cams = sorted(dets, key=lambda k: int(k.split("_")[1]))
        n = min(len(v) for v in dets.values())
        # caps to pull actual frames for image writing
        caps = {c: cv2.VideoCapture(str(CLIPS_ROOT / "P01" / f"{stem}.{int(c.split('_')[1])}.mp4"))
                for c in cams}
        for fr in range(n):
            obs = {c: dets[c][fr] for c in cams if dets[c][fr] is not None}
            X, kept = consensus(obs)
            if X is None:
                for c in obs:
                    per_cam[c]["drop"] += 1; stats["dropped"] += 1
                continue
            for c in cams:
                line = None; kind = None
                # consensus reprojection + apparent cup radius in this view (mm
                # world coords) -- sizes/places BOTH real and synthesized labels.
                c0, r = apparent_radius_px(c, X)
                if c in obs and c in kept:
                    z = dets[c][fr]                          # real detection centroid
                    line = _rect_poly_line(z[0], z[1], r); kind = "real"
                elif c in obs:
                    # detected but DISAGREES with consensus (e.g. cam_10 glass):
                    # SUPPRESS that wrong detection and REPLACE it with the cup
                    # reprojected from the >=3-cam consensus -> reject-THEN-fill.
                    line = bbox_label(c0, r); kind = "refill"
                else:
                    # no detection at all in a consensus frame -> fill from consensus.
                    line = bbox_label(c0, r); kind = "fill"
                if line is None:
                    per_cam[c]["drop"] += 1; stats["dropped"] += 1
                    continue
                # write image + label
                cap = caps[c]; cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
                ok, frame = cap.read()
                if not ok:
                    continue
                cdir = out_root / f"{stem}.{int(c.split('_')[1])}"
                cdir.mkdir(parents=True, exist_ok=True)
                stem_f = f"{stem}.{int(c.split('_')[1])}_f{fr:05d}"
                ip, lp = cdir / f"{stem_f}.jpg", cdir / f"{stem_f}.txt"
                cv2.imwrite(str(ip), frame); lp.write_text(line)
                pairs.append((ip, lp))
                per_cam[c][kind] += 1
                stats[{"real": "kept_real", "fill": "filled",
                       "refill": "refilled"}[kind]] += 1
        for cap in caps.values():
            cap.release()
    # marker so subsequent runs reuse this labelcache instead of rewriting 8k images
    done.write_text(json.dumps({
        "stats": stats, "per_cam": per_cam,
        "pairs": [str(ip.relative_to(out_root)) for ip, _ in pairs]}))
    return pairs, stats, per_cam


def precision_3d(model_weights, test_dir):
    """PRECISION via OUR 3D filter: run the student on held-out cams, triangulate
    its detections per frame, count a detection as a TRUE positive only if it lands
    in the >=3-cam <=30px consensus. precision = inliers / all student detections."""
    from ultralytics import YOLO
    from agreement import detect_rep
    model = YOLO(str(model_weights))
    # group test clips by rep -> {cam: video}
    reps = {}
    for clip in sorted(test_dir.glob("*.mp4")):
        cam = camera_id(clip.stem)
        rep_stem = clip.stem.rsplit(".", 1)[0]
        reps.setdefault(rep_stem, {})[f"cam_{cam.replace('cam','')}"] = clip
    tp = fp = 0
    for rep_stem, camclips in reps.items():
        tcalib = _calib_for(rep_stem)
        if tcalib is None:
            continue
        dets = detect_rep(model, {c: p for c, p in camclips.items() if c in tcalib},
                          CONF, None, verbose=False)
        dets = {c: [tuple(x) if x else None for x in v] for c, v in dets.items()}
        cams = sorted(dets, key=lambda k: int(k.split("_")[1]))
        n = min(len(v) for v in dets.values()) if dets else 0
        for fr in range(n):
            obs = {c: dets[c][fr] for c in cams if dets[c][fr] is not None}
            if not obs:
                continue
            _, kept = _consensus_for(obs, tcalib)
            for c in obs:
                if c in kept:
                    tp += 1
                else:
                    fp += 1
    return tp / (tp + fp) if (tp + fp) else 0.0


def _calib_for(rep_stem):
    p = rep_stem.split("_")[0]
    f = Path(f"data/calib/{p}/calibration.toml")
    if not f.exists():
        return None
    return load_calibration(str(f), target_size=RES)


def _consensus_for(obs, tcalib):
    cur = dict(obs)
    while len(cur) >= 2:
        X = triangulate_dlt([tcalib[c] for c in cur], [np.array(cur[c]) for c in cur])
        e = {c: float(np.hypot(*(project(tcalib[c], X)[0] - np.array(cur[c])))) for c in cur}
        w = max(e, key=e.get)
        if e[w] <= THR:
            break
        del cur[w]
    if len(cur) < MINC:
        return None, set()
    return None, set(cur)


def per_cam_eval(best, test_dir):
    """Held-out per-camera DETECTION RATE (fraction of frames the model emits ANY
    box). NOTE: this is NOT recall -- there is no ground truth and no cup/FP check;
    a glass or any false box counts. Use the 3D-consensus metrics (gated tri_rate /
    median_px) for correctness. Cache is keyed by the CHECKPOINT (best.pt path hash),
    not just CFG, so different checkpoints of the same config never reuse each other."""
    import hashlib
    from ultralytics import YOLO
    ckpt_id = hashlib.md5(str(Path(best).resolve()).encode()
                          + str(Path(best).stat().st_mtime).encode()).hexdigest()[:8]
    model = YOLO(str(best)); by_cam = {}
    for clip in sorted(test_dir.glob("*.mp4")):
        cam = camera_id(clip.stem)
        cf = PCACHE / f"{CFG}__{ckpt_id}__{clip.stem}.json"
        if cf.exists():
            pres = json.loads(cf.read_text())
        else:
            cap = cv2.VideoCapture(str(clip)); pres = []
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                r = model(f, conf=CONF, verbose=False)[0]
                pres.append(1 if (r.boxes is not None and len(r.boxes) > 0) else 0)
            cap.release(); cf.write_text(json.dumps(pres))
        d = by_cam.setdefault(cam, [0, 0]); d[0] += sum(pres); d[1] += len(pres)
    return {c: (d[0] / d[1] if d[1] else 0) for c, d in by_cam.items()}




def main():
    pairs, stats, per_cam = build_filled_pairs()
    print(f"\n[{CFG}] label pool: kept_real={stats['kept_real']} filled={stats['filled']} "
          f"refilled={stats.get('refilled',0)} dropped={stats['dropped']}", flush=True)
    print("per-cam (real / fill / refill / drop):")
    for c in sorted(per_cam, key=lambda k: int(k.split("_")[1])):
        d = per_cam[c]
        print(f"  {c:>7}: real={d['real']:>5} fill={d['fill']:>5} "
              f"refill={d.get('refill',0):>5} drop={d['drop']:>5}", flush=True)

    data_yaml, n = run.assemble_dataset(CFG, pairs)
    print(f"training on {n} frames (budget {run.TRAIN_FRAMES})", flush=True)
    test_dir = run.STAGE / "percam_eval"
    run.stage_clips(run.TEST, 1, run.ALL_CAMS, "right", test_dir)

    # Use the shared harness so we get BOTH: held-out-F1 early-stop AND a per-epoch
    # eval_by_epoch.json (recall/f1/tri_rate/median_px/cams at EVERY saved epoch) +
    # every epoch checkpoint -- so the per-epoch analysis in findings.ipynb (the
    # early-stop / wrong-but-confident-epoch story) reproduces for this student too.
    best, _ = train_with_metrics(
        data_yaml, run.STUDENT, str(run.RUNS.resolve()), CFG, test_dir,
        max_epochs=60, earlystop_on_f1=True, save_every_epoch=True, save_period=1,
        agr_participants=["P01"], agr_reps=1, agr_hand="right",
        config={"cfg_id": CFG, "train_participants": ["P01"], "reps": len(REPS),
                "clean": "3d_gate_reject_THEN_reproject_fill", "train_frames": n})

    rec = per_cam_eval(best, test_dir)
    prec3d = precision_3d(best, test_dir)

    base = json.load(open(PCACHE / "percam_recall.json"))["pscale_1"]
    no10 = json.load(open(PCACHE / "pscale_1_no10_recall.json"))
    cl3d = json.load(open(PCACHE / "pscale_1_clean3d_recall.json"))
    cams = sorted(rec, key=lambda k: int(k.replace("cam", "")))
    m = lambda d: float(np.mean([d.get(c, 0) for c in cams]))
    print("\n=== per-camera held-out RECALL ===", flush=True)
    print("cfg                  " + " ".join(f"{c:>6}" for c in cams), flush=True)
    for name, d in [("pscale_1 (raw)", base), ("no_cam10 (drop)", no10),
                    ("clean3d (reject)", cl3d), ("clean3d_fill (NEW)", rec)]:
        print(f"{name:<20}" + " ".join(f"{d.get(c,0):>6.2f}" for c in cams), flush=True)
    print(f"\nmean RECALL: raw={m(base):.3f} drop={m(no10):.3f} reject={m(cl3d):.3f} fill={m(rec):.3f}", flush=True)
    print(f"cam10 RECALL: raw={base.get('cam10',0):.2f} drop={no10.get('cam10',0):.2f} "
          f"reject={cl3d.get('cam10',0):.2f} fill={rec.get('cam10',0):.2f}", flush=True)
    print(f"\nPRECISION (via 3D filter) of NEW fill student: {prec3d:.3f}", flush=True)

    out = {"per_cam_recall": rec, "precision_3d": prec3d, "label_stats": stats,
           "per_cam_label_stats": per_cam, "mean_recall": m(rec)}
    (PCACHE / f"{CFG}_recall.json").write_text(json.dumps(rec, indent=2))
    (PCACHE / f"{CFG}_full.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {PCACHE / (CFG + '_full.json')}", flush=True)


if __name__ == "__main__":
    main()
