"""Diagnostics, gates, and helpers for the gated auto-pipeline (pipeline.py).

Every stage of the pipeline ends in a *gate*: a cheap diagnostic that returns a
GateResult with one of three dispositions:

    PASS       -> auto-proceed
    FLAG_AUTO  -> a problem worth noting; take the recommended default + log it
    FLAG_FORK  -> a genuine decision; halt and ask the user to choose

This module also holds the reusable helpers the driver calls between the
existing scripts (sampling, seed extraction, val splitting, loss-plateau
training, label-review video, coverage + eval reports). It deliberately reuses
build_dataset / label_clip / filter_detections / YOLO rather than re-implementing.
"""
from __future__ import annotations

import json
import random
import re
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from kalman import filter_detections


# --------------------------------------------------------------------------- #
# Thresholds (all overridable via --config <json>)
# --------------------------------------------------------------------------- #
DEFAULTS = {
    "probe_frames_per_clip": 3,     # frames sampled per clip in the auto-detect probe
    "auto_recall_min": 0.50,        # min probe recall to recommend the AUTO path
    "auto_phantom_max": 0.30,       # max KF phantom rate to still call AUTO "consistent"
    "seed_frames_per_clip": 10,     # frames extracted per clip for SAM seeding
    "val_frac": 0.15,               # random-frame holdout fraction
    "min_labels_per_cam": 30,       # below this a camera is "weak" (FLAG_AUTO)
    "blind_cam_limit": 2,           # > this many zero-label cameras -> patch FORK
    "max_phantom_rate": 0.15,       # teacher phantom rate above this -> FLAG_AUTO
    "plateau_patience": 12,         # epochs of no val-loss gain before stopping
    "plateau_delta": 0.01,
    "plateau_min_epoch": 30,        # seed (small) data needs many epochs; dense
    "max_epochs": 120,              # data plateau-stops on its own well before this

    "imgsz": 640,
    "batch": 16,
    "coco_weights": "data/pretrained/yolo26x-seg.pt",
    "world_weights": "yolov8x-worldv2.pt",
    "student_weights": "data/pretrained/yolo26n-seg.pt",
    "probe_conf": 0.05,
    "label_conf": 0.10,
    "eval_conf": 0.25,
}

PASS, FLAG_AUTO, FLAG_FORK = "PASS", "FLAG_AUTO", "FLAG_FORK"


@dataclass
class GateResult:
    stage: str
    status: str                      # PASS | FLAG_AUTO | FLAG_FORK
    summary: str = ""
    metrics: dict = field(default_factory=dict)
    recommendation: str = ""
    options: list[str] = field(default_factory=list)   # for FLAG_FORK
    artifacts: list[str] = field(default_factory=list)  # files to look at

    def as_record(self, decision: str | None = None) -> dict:
        d = asdict(self)
        d["decision"] = decision
        return d


# --------------------------------------------------------------------------- #
# Gate report printing + fork prompt
# --------------------------------------------------------------------------- #
def print_gate(r: GateResult) -> None:
    bar = "=" * 64
    print(f"\n{bar}\n[GATE: {r.stage}]  {r.status}\n{bar}")
    if r.summary:
        print(r.summary)
    for k, v in r.metrics.items():
        print(f"  {k}: {v}")
    for a in r.artifacts:
        print(f"  look: {a}")
    if r.recommendation:
        print(f"  recommendation: {r.recommendation}")


def ask_fork(r: GateResult) -> int:
    """Print a FORK gate's options and return the chosen 1-based index."""
    print_gate(r)
    print("\n  DECISION REQUIRED:")
    for i, opt in enumerate(r.options, 1):
        print(f"    [{i}] {opt}")
    while True:
        raw = input("  choose> ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(r.options):
            return int(raw)
        print("  invalid choice, try again.")


# --------------------------------------------------------------------------- #
# Manifest (audit log + resume)
# --------------------------------------------------------------------------- #
def manifest_path(run_dir: Path) -> Path:
    return run_dir / "manifest.json"


def load_manifest(run_dir: Path) -> dict:
    p = manifest_path(run_dir)
    if p.exists():
        return json.loads(p.read_text())
    return {"stages": {}, "config": {}, "artifacts": {}}


def save_manifest(run_dir: Path, m: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path(run_dir).write_text(json.dumps(m, indent=2))


def record_stage(run_dir: Path, m: dict, stage: str, result: GateResult,
                 decision: str | None = None, artifacts: dict | None = None) -> None:
    m["stages"][stage] = result.as_record(decision)
    if artifacts:
        m["artifacts"].update({k: str(v) for k, v in artifacts.items()})
    save_manifest(run_dir, m)


def stage_done(m: dict, stage: str) -> bool:
    return m["stages"].get(stage, {}).get("status") is not None


# --------------------------------------------------------------------------- #
# Camera id heuristic (group labels/clips by camera)
# --------------------------------------------------------------------------- #
def camera_id(stem: str) -> str:
    """Best-effort camera key from a clip/label stem.

    Handles cup-style ('cam_1_<ts>') and ARAT-style ('..._<ts>.3') naming;
    falls back to the whole stem when neither matches.
    """
    m = re.search(r"cam[_-]?(\d+)", stem, re.IGNORECASE)
    if m:
        return f"cam{int(m.group(1))}"
    m = re.search(r"\.(\d+)(?:_f\d+)?$", stem)   # ARAT camera suffix, opt. _f<idx>
    if m:
        return f"cam{int(m.group(1))}"
    return stem


def label_to_camera(label_name: str) -> str:
    stem = re.sub(r"_f\d+$", "", Path(label_name).stem)
    return camera_id(stem)


# --------------------------------------------------------------------------- #
# Stage 0: auto-detectability probe
# --------------------------------------------------------------------------- #
def _sample_frames(clips_dir: Path, n_per_clip: int):
    clips = sorted(clips_dir.glob("*.mp4"))
    for clip in clips:
        cap = cv2.VideoCapture(str(clip))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        idxs = [int(x) for x in np.linspace(n * 0.15, n * 0.85, n_per_clip)]
        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, f = cap.read()
            if ret:
                yield clip, fi, f
        cap.release()


def _run_detector(model, frames, conf, classes=None, set_classes=None):
    """Return (recall, per_frame_dets) over a list of frames."""
    if set_classes is not None:
        model.set_classes(set_classes)
    per_frame, det = [], 0
    for f in frames:
        r = model(f, conf=conf, classes=classes, verbose=False)[0]
        dets = []
        if r.boxes is not None and len(r.boxes) > 0:
            det += 1
            for b in r.boxes.xyxy.cpu().numpy():
                cx, cy = float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2)
                dets.append({"cx": cx, "cy": cy, "conf": 1.0,
                             "bbox": [float(x) for x in b]})
        per_frame.append(dets)
    return (det / len(frames) if frames else 0.0), per_frame


def probe_auto_detect(clips_dir: Path, object_name: str, out_dir: Path,
                      thr: dict) -> tuple[GateResult, dict]:
    """Sample frames, try COCO + YOLO-World, recommend AUTO vs SAM.

    Returns (GateResult[FLAG_FORK], extra) where extra carries the recommended
    source and, for AUTO, suggested teacher settings.
    """
    from ultralytics import YOLO, SAM  # noqa: F401  (SAM import kept symmetric)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = list(_sample_frames(clips_dir, thr["probe_frames_per_clip"]))
    frames = [f for _, _, f in samples]

    coco = YOLO(thr["coco_weights"])
    coco_recall, _ = _run_detector(coco, frames, thr["probe_conf"], classes=None)

    world_recall = 0.0
    if Path(thr["world_weights"]).exists() and object_name:
        world = YOLO(thr["world_weights"])
        world_recall, _ = _run_detector(world, frames, thr["probe_conf"],
                                        set_classes=[object_name])

    best = max(coco_recall, world_recall)
    auto_ok = best >= thr["auto_recall_min"]
    rec_src = "auto" if auto_ok else "sam"

    metrics = {"coco_recall": round(coco_recall, 3),
               "world_recall": round(world_recall, 3),
               "frames_sampled": len(frames)}
    summary = (f"Auto-detectability probe for '{object_name}': "
               f"COCO recall {coco_recall:.2f}, YOLO-World recall {world_recall:.2f}.")
    r = GateResult(
        stage="0_probe", status=FLAG_FORK, summary=summary, metrics=metrics,
        recommendation=("SAM seeding (object not reliably auto-detected)"
                        if rec_src == "sam"
                        else "AUTO teacher (object reliably detected by YOLO)"),
        options=["SAM seed (recommended)" if rec_src == "sam" else "SAM seed",
                 "Use AUTO detect teacher" if rec_src == "auto"
                 else "Force AUTO detect teacher",
                 "Abort"])
    return r, {"recommended": rec_src, "coco_recall": coco_recall,
               "world_recall": world_recall}


# --------------------------------------------------------------------------- #
# SAM seed-frame extraction
# --------------------------------------------------------------------------- #
def extract_seed_frames(clips_dir: Path, dataset: Path, frames_per_clip: int) -> int:
    img_dir = dataset / "images" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for clip in sorted(clips_dir.glob("*.mp4")):
        cap = cv2.VideoCapture(str(clip))
        tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        for fi in [int(x) for x in np.linspace(tot * 0.12, tot * 0.9, frames_per_clip)]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, f = cap.read()
            if ret:
                cv2.imwrite(str(img_dir / f"{clip.stem}_f{fi:05d}.jpg"), f)
                n += 1
        cap.release()
    return n


# --------------------------------------------------------------------------- #
# Stage 2: label QA
# --------------------------------------------------------------------------- #
def qa_labels(dataset: Path) -> GateResult:
    lbl_dir = dataset / "labels" / "train"
    labels = sorted(lbl_dir.glob("*.txt"))
    degenerate, per_cam = [], {}
    for lp in labels:
        toks = lp.read_text().split()
        npts = (len(toks) - 1) // 2
        cam = label_to_camera(lp.name)
        per_cam[cam] = per_cam.get(cam, 0) + 1
        if npts < 3:
            degenerate.append(lp.name)
    for name in degenerate:                       # drop degenerate label + image
        (lbl_dir / name).unlink(missing_ok=True)
        (dataset / "images" / "train" / (Path(name).stem + ".jpg")).unlink(missing_ok=True)
    status = FLAG_AUTO if degenerate else PASS
    return GateResult(
        stage="2_label_qa", status=status,
        summary=(f"{len(labels)} labels; dropped {len(degenerate)} degenerate (<3 pts)."
                 if degenerate else f"{len(labels)} labels, all valid."),
        metrics={"per_camera": per_cam, "dropped": len(degenerate)})


# --------------------------------------------------------------------------- #
# Random-frame val split
# --------------------------------------------------------------------------- #
def make_val_split(dataset: Path, frac: float, seed: int = 0) -> int:
    """Move a random `frac` of train image+label pairs into images/val+labels/val
    and point data.yaml's val at it. Idempotent: no-op if a val split exists."""
    img_tr, lbl_tr = dataset / "images/train", dataset / "labels/train"
    img_va, lbl_va = dataset / "images/val", dataset / "labels/val"
    if img_va.exists() and any(img_va.glob("*.jpg")):
        return len(list(img_va.glob("*.jpg")))
    img_va.mkdir(parents=True, exist_ok=True)
    lbl_va.mkdir(parents=True, exist_ok=True)
    stems = [p.stem for p in sorted(img_tr.glob("*.jpg"))
             if (lbl_tr / (p.stem + ".txt")).exists()]
    random.Random(seed).shuffle(stems)
    k = max(1, int(len(stems) * frac))
    for s in stems[:k]:
        shutil.move(str(img_tr / f"{s}.jpg"), str(img_va / f"{s}.jpg"))
        shutil.move(str(lbl_tr / f"{s}.txt"), str(lbl_va / f"{s}.txt"))
    y = yaml.safe_load((dataset / "data.yaml").read_text())
    y["path"] = str(dataset.resolve())   # robust to copied/moved datasets
    y["val"] = "images/val"
    (dataset / "data.yaml").write_text(yaml.safe_dump(y, sort_keys=False))
    return k


# --------------------------------------------------------------------------- #
# Loss-plateau training (ultralytics callback stops on val-loss plateau)
# --------------------------------------------------------------------------- #
def epoch_policy(n_images: int) -> tuple[int, int, int]:
    """(max_epochs, min_epoch, patience) scaled to dataset size.

    Small seed sets have cheap epochs and keep improving for a long time, so we
    let them run long; big dense sets plateau within ~15-20 epochs.
    """
    if n_images < 300:
        return 200, 60, 20
    if n_images < 2000:
        return 120, 25, 12
    return 80, 12, 8


def loss_plateau_train(data: Path, weights: str, project: str, name: str,
                       thr: dict) -> Path:
    from ultralytics import YOLO
    data = Path(data)
    n_train = len(list((data.parent / "images" / "train").glob("*.jpg")))
    max_epochs, min_epoch, patience = epoch_policy(n_train)
    print(f"[epochs] {n_train} train images -> max={max_epochs} "
          f"min={min_epoch} patience={patience}")
    model = YOLO(weights)
    st = {"best": 1e9, "best_ep": 0}

    def on_fit_epoch_end(trainer):
        ep = int(getattr(trainer, "epoch", 0)) + 1
        met = getattr(trainer, "metrics", {}) or {}
        vloss = sum(float(v) for k, v in met.items()
                    if k.startswith("val/") and k.endswith("_loss"))
        if vloss and vloss < st["best"] - thr["plateau_delta"]:
            st["best"], st["best_ep"] = vloss, ep
        if ep >= min_epoch and (ep - st["best_ep"]) >= patience:
            print(f"[plateau] val-loss no >{thr['plateau_delta']} gain for "
                  f"{ep - st['best_ep']} epochs; stopping at epoch {ep}.")
            trainer.stop = True

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.train(data=str(data), epochs=max_epochs, imgsz=thr["imgsz"],
                batch=thr["batch"], project=project, name=name)
    return Path(project) / name / "weights" / "best.pt"


def converged(project: str, name: str) -> tuple[bool, dict]:
    """Read results.csv; report whether val loss fell meaningfully."""
    csv = Path(project) / name / "results.csv"
    if not csv.exists():
        return False, {}
    import csv as _csv
    rows = list(_csv.DictReader(csv.open()))
    if len(rows) < 2:
        return False, {}
    def vloss(row):
        return sum(float(v) for k, v in row.items()
                   if k.strip().startswith("val/") and k.strip().endswith("loss"))
    first, last, best = vloss(rows[0]), vloss(rows[-1]), min(vloss(r) for r in rows)
    drop = first - best
    return drop > 0.1, {"val_loss_first": round(first, 3),
                        "val_loss_best": round(best, 3),
                        "epochs": len(rows)}


# --------------------------------------------------------------------------- #
# Stage 4: coverage report (over build_dataset's per_clip + KF stats)
# --------------------------------------------------------------------------- #
def coverage_report(per_clip: dict, kf_phantom_rate: float, thr: dict) -> GateResult:
    per_cam: dict[str, int] = {}
    for stem, cnt in per_clip.items():
        cam = camera_id(stem)
        per_cam[cam] = per_cam.get(cam, 0) + cnt
    blind = [c for c, n in per_cam.items() if n == 0]
    weak = [c for c, n in per_cam.items() if 0 < n < thr["min_labels_per_cam"]]
    metrics = {"per_camera": per_cam, "blind_cams": blind, "weak_cams": weak,
               "teacher_phantom_rate": round(kf_phantom_rate, 3),
               "total_labels": sum(per_cam.values())}
    if len(blind) > thr["blind_cam_limit"]:
        return GateResult(
            stage="4_coverage", status=FLAG_FORK,
            summary=f"{len(blind)} cameras got zero labels: {blind}.",
            metrics=metrics,
            recommendation="patch blind cameras with a few SAM seeds before finetune #2",
            options=["Patch blind cams via SAM seeds", "Proceed anyway"])
    status = FLAG_AUTO if (blind or weak or kf_phantom_rate > thr["max_phantom_rate"]) else PASS
    return GateResult(
        stage="4_coverage", status=status,
        summary=(f"{sum(per_cam.values())} labels across {len(per_cam)} cameras; "
                 f"blind={blind} weak={weak} phantom_rate={kf_phantom_rate:.2f}."),
        metrics=metrics)


# --------------------------------------------------------------------------- #
# Label-review video (faithful: draws the saved labels the model will train on)
# --------------------------------------------------------------------------- #
def render_label_video(dataset: Path, out_mp4: Path, max_w: int = 960,
                       fps: int = 20) -> Path:
    """Play every labeled (train+val) frame in order with its saved polygon
    drawn, then transcode to H.264. This is exactly the data finetune sees."""
    pairs = []
    for split in ("train", "val"):
        for img in sorted((dataset / "images" / split).glob("*.jpg")):
            lbl = dataset / "labels" / split / (img.stem + ".txt")
            if lbl.exists():
                pairs.append((img, lbl))
    pairs.sort(key=lambda p: p[0].name)
    if not pairs:
        raise SystemExit(f"no labeled frames in {dataset}")
    sample = cv2.imread(str(pairs[0][0]))
    h0, w0 = sample.shape[:2]
    scale = min(1.0, max_w / w0)
    W, H = int(w0 * scale), int(h0 * scale)
    raw = out_mp4.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for img_p, lbl_p in pairs:
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        hh, ww = img.shape[:2]
        toks = lbl_p.read_text().split()
        pts = np.array([float(x) for x in toks[1:]]).reshape(-1, 2)
        pts[:, 0] *= ww; pts[:, 1] *= hh
        poly = pts.astype(np.int32)
        ov = img.copy(); cv2.fillPoly(ov, [poly], (80, 220, 120))
        img = cv2.addWeighted(ov, 0.35, img, 0.65, 0)
        cv2.polylines(img, [poly], True, (60, 255, 90), 2)
        cv2.putText(img, img_p.stem, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
        vw.write(cv2.resize(img, (W, H)))
    vw.release()
    _to_h264(raw, out_mp4)
    raw.unlink(missing_ok=True)
    return out_mp4


def _to_h264(src: Path, dst: Path) -> None:
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-i", str(src), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "23", str(dst)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    else:
        shutil.copy(src, dst)


# --------------------------------------------------------------------------- #
# Stage 6: eval gate (recall + KF precision proxies on full clips)
# --------------------------------------------------------------------------- #
def eval_gate(weights: str, clips_dir: Path, conf: float,
              low_precision_thr: float = 0.7) -> GateResult:
    from ultralytics import YOLO
    model = YOLO(weights)
    by_cam: dict[str, dict] = {}
    for clip in sorted(clips_dir.glob("*.mp4")):
        cam = camera_id(clip.stem)
        cap = cv2.VideoCapture(str(clip))
        per_frame = []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            r = model(f, conf=conf, verbose=False)[0]
            dets = []
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes.xyxy.cpu().numpy():
                    cx, cy = float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2)
                    dets.append({"cx": cx, "cy": cy, "conf": 1.0})
            per_frame.append(dets)
        cap.release()
        filt = filter_detections(per_frame)
        F = len(per_frame); detF = sum(1 for p in per_frame if p)
        tot = sum(len(p) for p in per_frame)
        phan = sum(ff.phantom for ff in filt); tele = sum(ff.tele for ff in filt)
        d = by_cam.setdefault(cam, {"F": 0, "detF": 0, "tot": 0, "phan": 0, "tele": 0})
        d["F"] += F; d["detF"] += detF; d["tot"] += tot
        d["phan"] += phan; d["tele"] += tele

    table, low = {}, []
    agg = {"F": 0, "detF": 0, "tot": 0, "phan": 0, "tele": 0}
    for cam, d in sorted(by_cam.items()):
        for k in agg:
            agg[k] += d[k]
        rec = d["detF"] / d["F"] if d["F"] else 0
        pl = (d["tot"] - d["phan"] - d["tele"]) / d["tot"] if d["tot"] else 0
        table[cam] = {"recall": round(rec, 3), "p_loose": round(pl, 3)}
        if d["tot"] and pl < low_precision_thr:
            low.append(cam)
    o_rec = agg["detF"] / agg["F"] if agg["F"] else 0
    o_pl = (agg["tot"] - agg["phan"] - agg["tele"]) / agg["tot"] if agg["tot"] else 0
    status = FLAG_AUTO if low else PASS
    return GateResult(
        stage="6_eval", status=status,
        summary=(f"overall recall {o_rec:.3f}, P_loose {o_pl:.3f}; "
                 f"low-precision cams: {low or 'none'}."),
        metrics={"overall_recall": round(o_rec, 3),
                 "overall_p_loose": round(o_pl, 3),
                 "per_camera": table, "low_precision_cams": low})
