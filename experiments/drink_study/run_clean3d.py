"""3D-cleaned (rejection-only) P01 retrain. For each pscale_1 rep, run the teacher
on all 10 cams (cached), 3D-inlier-gate per frame (>=3 agreeing cams, thr=30px),
and for each camera KEEP only the frames whose detection is a 3D inlier — drop the
glass (cam_10) and bracelet (cam_4) outliers. Build cleaned labels from the
existing per-clip cache (same polygons, just filtered), train P01 on them, eval
per-camera vs pscale_1 (raw) and pscale_1_no10 (drop-cam10 ablation).
"""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cv2
import numpy as np
import run
from kalman_3d import load_calibration, triangulate_dlt, project
from agreement import RES
from pipeline_lib import camera_id
from metrics import train_with_metrics

CFG = "pscale_1_clean3d"
REPS = run.reps_of("P01", "right")[:2]
CACHE = Path("experiments/drink_study/cache")
PCACHE = CACHE / "percam"
CONF = 0.25
THR = 30.0
calib = load_calibration("data/calib/P01/calibration.toml", target_size=RES)


def teacher_dets(stem: str):
    cf = CACHE / f"P01_{stem}__teacher__c0.25.json"
    if cf.exists():
        d = json.loads(cf.read_text())
        return {c: [tuple(x) if x else None for x in v] for c, v in d.items()}
    from ultralytics import YOLO
    from agreement import detect_rep
    from pseudo_label import CUP_LIKE_CLASSES
    pdir = Path("/home/imove/Documents/clips/P01")
    rep = {f"cam_{c}": pdir / f"{stem}.{c}.mp4" for c in range(1, 11)
           if f"cam_{c}" in calib and (pdir / f"{stem}.{c}.mp4").exists()}
    print(f"  teacher on {stem} (all cams) ...", flush=True)
    dets = detect_rep(YOLO(run.COCO), rep, CONF, CUP_LIKE_CLASSES, verbose=True)
    cf.write_text(json.dumps(dets))
    return {c: [tuple(x) if x else None for x in v] for c, v in dets.items()}


def inlier_frames(dets) -> dict[str, set[int]]:
    """For each camera, the set of frame indices whose detection is a 3D inlier."""
    cams = sorted(dets, key=lambda k: int(k.split("_")[1]))
    n = min(len(v) for v in dets.values())
    keep = {c: set() for c in cams}
    for fr in range(n):
        obs = {c: dets[c][fr] for c in cams if dets[c][fr] is not None}
        if len(obs) < 3:
            continue
        cur = dict(obs)
        while len(cur) >= 2:
            X = triangulate_dlt([calib[c] for c in cur], [np.array(cur[c]) for c in cur])
            e = {c: float(np.hypot(*(project(calib[c], X)[0] - np.array(cur[c])))) for c in cur}
            w = max(e, key=e.get)
            if e[w] <= THR:
                break
            del cur[w]
        if len(cur) < 3:
            continue
        X = triangulate_dlt([calib[c] for c in cur], [np.array(cur[c]) for c in cur])
        for c in cur:
            if float(np.hypot(*(project(calib[c], X)[0] - np.array(cur[c])))) <= THR:
                keep[c].add(fr)
    return keep


def build_clean_pairs():
    """Cleaned (img,label) pairs: from the per-clip cache, keep only frames whose
    camera detection passed the 3D gate. Cache .txt filenames encode frame idx _fNNNNN."""
    pairs, kept, dropped = [], 0, 0
    for stem in REPS:
        dets = teacher_dets(stem)
        keep = inlier_frames(dets)
        for cam in range(1, 11):
            clip_stem = f"{stem}.{cam}"
            cdir = run.LABELCACHE / "kf" / clip_stem
            if not (cdir / ".done").exists():
                continue
            ck = f"cam_{cam}"
            keepset = keep.get(ck, set())
            for jpg in sorted(cdir.glob("*.jpg")):
                fi = int(jpg.stem.rsplit("_f", 1)[1])
                if fi in keepset:
                    pairs.append((jpg, cdir / (jpg.stem + ".txt"))); kept += 1
                else:
                    dropped += 1
    print(f"3D-clean: kept {kept} labels, dropped {dropped} "
          f"({100*dropped/(kept+dropped):.0f}% rejected)", flush=True)
    return pairs


def per_cam_eval(best, test_dir):
    from ultralytics import YOLO
    model = YOLO(str(best)); by_cam = {}
    for clip in sorted(test_dir.glob("*.mp4")):
        cam = camera_id(clip.stem)
        cf = PCACHE / f"{CFG}__{clip.stem}.json"
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
    return {c: (d[0]/d[1] if d[1] else 0) for c, d in by_cam.items()}


def main():
    pairs = build_clean_pairs()
    data_yaml, n = run.assemble_dataset(CFG, pairs)
    print(f"[{CFG}] cleaned pool={len(pairs)} -> training on {n} frames", flush=True)
    test_dir = run.STAGE / "percam_eval"
    run.stage_clips(run.TEST, 1, run.ALL_CAMS, "right", test_dir)
    best, _ = train_with_metrics(
        data_yaml, run.STUDENT, str(run.RUNS.resolve()), CFG, test_dir,
        long_run=False, max_epochs=20,
        config={"cfg_id": CFG, "train_participants": ["P01"], "reps": 2,
                "clean": "3d_gate_reject", "train_frames": n})
    rec = per_cam_eval(best, test_dir)

    base = json.load(open(PCACHE / "percam_recall.json"))["pscale_1"]
    no10 = json.load(open(PCACHE / "pscale_1_no10_recall.json"))
    cams = sorted(rec, key=lambda k: int(k.replace("cam", "")))
    print("\n=== per-camera held-out recall ===", flush=True)
    hdr = "cfg                " + " ".join(f"{c:>6}" for c in cams)
    print(hdr, flush=True)
    print("pscale_1 (raw)     " + " ".join(f"{base.get(c,0):>6.2f}" for c in cams), flush=True)
    print("no_cam10 (drop)    " + " ".join(f"{no10.get(c,0):>6.2f}" for c in cams), flush=True)
    print("clean3d (gate all) " + " ".join(f"{rec.get(c,0):>6.2f}" for c in cams), flush=True)
    m = lambda d: np.mean([d.get(c, 0) for c in cams])
    print(f"\nmean held-out recall: raw={m(base):.3f}  drop_cam10={m(no10):.3f}  clean3d={m(rec):.3f}", flush=True)
    print(f"cam10: raw={base.get('cam10',0):.2f} drop={no10.get('cam10',0):.2f} clean3d={rec.get('cam10',0):.2f}", flush=True)
    print(f"cam4:  raw={base.get('cam4',0):.2f} drop={no10.get('cam4',0):.2f} clean3d={rec.get('cam4',0):.2f}", flush=True)
    (PCACHE / f"{CFG}_recall.json").write_text(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
