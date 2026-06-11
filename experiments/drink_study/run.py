"""Config-driven harness for the drinking-task study.

Each config runs the spine (COCO cup teacher -> KF dense self-label -> student)
on a selected set of clips and captures full training metrics via metrics.py.
Experiments differ only in the clip selector / flags; results land under
experiments/drink_study/runs/<cfg_id>/ with curves + held-out eval.

The experiment SWEEP is deferred. For now only the `baseline` config is wired to
run, to validate metric capture. Selectors for the deferred experiments (KF
ablation, clip/participant scaling, camera transfer) are provided as helpers.

Usage:
    python experiments/drink_study/run.py --config baseline --long-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # for metrics.py

import random
import shutil

import cv2
import yaml
from ultralytics import YOLO

from pseudo_label import label_clip, CUP_LIKE_CLASSES
from kalman import GATE, MAX_MISS
from metrics import train_with_metrics

CLIPS_ROOT = Path("/home/imove/Documents/clips")
STUDY = ROOT / "experiments" / "drink_study"
RUNS = STUDY / "runs"
STAGE = ROOT / "data" / "clips" / "drink_study"      # symlink staging (data artifacts)
DSROOT = ROOT / "data" / "datasets" / "drink_study"
LABELCACHE = DSROOT / "_labelcache"   # per-clip labels, labeled ONCE, shared by all configs
COCO = "data/pretrained/yolo26x-seg.pt"
STUDENT = "data/pretrained/yolo26n-seg.pt"

# Fixed controls
TEST = ["P06", "P19", "P23"]
TRAIN_POOL = ["P01", "P02", "P03", "P04", "P05", "P08", "P09", "P10"]
ALL_CAMS = list(range(1, 11))

# STANDARD: every student trains on this many frames (randomly sampled from the
# config's clip pool), regardless of how many clips/people/cameras it draws from.
TRAIN_FRAMES = 3000
VAL_FRAMES = 450
SAMPLE_SEED = 0


def reps_of(participant: str, hand: str) -> list[str]:
    """Sorted unique rep stems (timestamps) for a participant's drinking, one hand."""
    pdir = CLIPS_ROOT / participant
    stems = {f.name.rsplit(".", 2)[0]
             for f in pdir.glob(f"{participant}_drinking_{hand}_*.mp4")}
    return sorted(stems)


def stage_clips(participants, reps, cameras, hand, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    for x in out_dir.glob("*.mp4"):
        x.unlink()
    n = 0
    for p in participants:
        for stem in reps_of(p, hand)[:reps]:
            for cam in cameras:
                src = CLIPS_ROOT / p / f"{stem}.{cam}.mp4"
                if src.exists():
                    (out_dir / src.name).symlink_to(src)
                    n += 1
    return n


def clip_files(participants, reps, cameras, hand) -> list[Path]:
    """The selected clip paths (one per camera per rep per participant)."""
    out = []
    for p in participants:
        for stem in reps_of(p, hand)[:reps]:
            for cam in cameras:
                src = CLIPS_ROOT / p / f"{stem}.{cam}.mp4"
                if src.exists():
                    out.append(src)
    return out


def label_clip_cached(teacher, clip: Path, use_kf: bool = True,
                      conf: float = 0.10) -> list[tuple[Path, Path]]:
    """Label ONE clip once and cache (frame.jpg, label.txt) under LABELCACHE.
    Shared across every config that uses this clip, so the slow COCO teacher
    runs on each unique clip exactly once. Returns the (img, label) pairs."""
    kf_tag = "kf" if use_kf else "nokf"
    cache = LABELCACHE / kf_tag / clip.stem
    if (cache / ".done").exists():
        return [(p, cache / (p.stem + ".txt")) for p in sorted(cache.glob("*.jpg"))]
    cache.mkdir(parents=True, exist_ok=True)
    accepted, _ = label_clip(teacher, clip, conf=conf, classes=CUP_LIKE_CLASSES,
                             gate=GATE, max_miss=MAX_MISS, use_kf=use_kf)
    cap = cv2.VideoCapture(str(clip))
    fi, out = 0, []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi in accepted:
            stem = f"{clip.stem}_f{fi:05d}"
            ip, lp = cache / f"{stem}.jpg", cache / f"{stem}.txt"
            cv2.imwrite(str(ip), frame)
            poly = accepted[fi]
            lp.write_text("0 " + " ".join(f"{x:.6f} {y:.6f}" for x, y in poly) + "\n")
            out.append((ip, lp))
        fi += 1
    cap.release()
    (cache / ".done").write_text("ok")
    return out


def assemble_dataset(cfg_id: str, pairs: list[tuple[Path, Path]],
                     n_train: int = TRAIN_FRAMES, n_val: int = VAL_FRAMES) -> tuple[Path, int]:
    """Randomly sample n_train(+n_val) frames from the labeled pool and build the
    config's dataset via symlinks into the per-clip cache (keeps all labels on
    disk). Every config thus trains on the SAME number of frames."""
    ds = DSROOT / cfg_id
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            d = ds / sub / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
    pool = list(pairs)
    random.Random(SAMPLE_SEED).shuffle(pool)
    n_train = min(n_train, len(pool))
    train = pool[:n_train]
    val = pool[n_train:n_train + n_val]
    for split, sel in (("train", train), ("val", val)):
        for ip, lp in sel:
            (ds / "images" / split / ip.name).symlink_to(ip.resolve())
            (ds / "labels" / split / lp.name).symlink_to(lp.resolve())
    y = {"path": str(ds.resolve()), "train": "images/train",
         "val": "images/val" if val else "images/train", "nc": 1, "names": ["cup"]}
    (ds / "data.yaml").write_text(yaml.safe_dump(y, sort_keys=False))
    return ds / "data.yaml", len(train)


def run_spine(cfg_id: str, *, train_participants, reps, train_cameras=ALL_CAMS,
              hand="right", use_kf=True, eval_participants=TEST, eval_reps=1,
              long_run=False, max_epochs=20, save_period=10,
              student_weights=STUDENT, agreement=True, teacher=None) -> Path:
    eval_clips = STAGE / cfg_id / "eval"
    n_ev = stage_clips(eval_participants, eval_reps, ALL_CAMS, hand, eval_clips)

    # Label each unique clip once (cached + shared), then sample a FIXED budget.
    clips = clip_files(train_participants, reps, train_cameras, hand)
    print(f"[{cfg_id}] {len(clips)} train clips; labeling (cached) ...", flush=True)
    if teacher is None:
        teacher = YOLO(COCO)
    pairs = []
    for i, clip in enumerate(clips, 1):
        pairs += label_clip_cached(teacher, clip, use_kf=use_kf)
        if i % 10 == 0 or i == len(clips):
            print(f"  labeled {i}/{len(clips)} clips, pool={len(pairs)} frames", flush=True)
    if not pairs:
        raise SystemExit(f"[{cfg_id}] teacher produced 0 labels; aborting.")
    data_yaml, n_used = assemble_dataset(cfg_id, pairs)
    print(f"[{cfg_id}] pool={len(pairs)} -> training on {n_used} frames "
          f"(budget {TRAIN_FRAMES})", flush=True)

    config = {"cfg_id": cfg_id, "train_participants": train_participants,
              "reps": reps, "train_cameras": train_cameras, "hand": hand,
              "use_kf": use_kf, "eval_participants": eval_participants,
              "eval_reps": eval_reps, "long_run": long_run,
              "student_weights": student_weights, "train_frames": n_used,
              "label_pool": len(pairs), "train_clips": len(clips), "eval_clips": n_ev}

    # In long_run mode, also score inter-camera agreement per checkpoint so we
    # can chart all 4 held-out metrics (recall/precision/F1/agreement) vs loss.
    agr_kw = {}
    if long_run and agreement:
        agr_kw = {"agr_participants": eval_participants, "agr_reps": eval_reps,
                  "agr_hand": hand}

    best, _ = train_with_metrics(
        data_yaml, student_weights, str(RUNS.resolve()), cfg_id, eval_clips,
        long_run=long_run, max_epochs=max_epochs, save_period=save_period,
        config=config, **agr_kw)

    # inter-camera agreement on the final model (held-out participants' calib).
    agr = {}
    if agreement:
        from agreement import agreement_eval
        print(f"[{cfg_id}] computing inter-camera agreement ...", flush=True)
        agr = agreement_eval(str(best), eval_participants, eval_reps, classes=None,
                             hand=hand, verbose=True)
        (RUNS / cfg_id / "agreement.json").write_text(json.dumps(agr, indent=2))
        print(f"[{cfg_id}] agreement: tri_rate={agr.get('tri_rate')} "
              f"cams={agr.get('mean_cams_agreeing')} "
              f"median_px={agr.get('median_reproj_px')}", flush=True)

    idx_path = STUDY / "results.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    idx[cfg_id] = {**config, "best_weights": str(best),
                   "agreement": {k: v for k, v in agr.items() if k != "per_rep"}}
    idx_path.write_text(json.dumps(idx, indent=2))
    print(f"[{cfg_id}] done -> {best}")
    return best


# ---- config registry --------------------------------------------------------
# Every config trains on TRAIN_FRAMES (=3000) frames; configs differ only in
# WHICH frames (more/fewer people, reps, cameras) — never in volume.
SWEEP_TRAIN = ["P01", "P02", "P03", "P04"]
SWEEP_REPS = 2          # enough label pool per participant to sample 3000 from


def cfg_baseline(long_run: bool):
    # verification config; save_period=1 -> held-out eval every epoch.
    return run_spine("baseline", train_participants=["P01"], reps=2,
                     eval_participants=["P06"], eval_reps=1,
                     long_run=long_run, max_epochs=30, save_period=1)


# ---- E3: participant scaling — same 3000 frames, drawn from N people -------- #
# Tests whether person-DIVERSITY helps at fixed training volume.
def cfg_pscale(n: int):
    return run_spine(f"pscale_{n}", train_participants=SWEEP_TRAIN[:n], reps=SWEEP_REPS,
                     eval_participants=TEST, eval_reps=1)


# ---- E5: clip scaling — same 3000 frames, drawn from N reps of one person --- #
# Tests whether temporal/pose DIVERSITY helps at fixed training volume.
def cfg_cscale(reps: int):
    return run_spine(f"cscale_{reps}", train_participants=["P01"], reps=reps,
                     eval_participants=TEST, eval_reps=1)


# ---- (b) student-capacity sweep: does the teacher precision gap close? ------ #
def cfg_size(size: str):
    weights = STUDENT if size == "n" else f"data/pretrained/yolo26{size}-seg.pt"
    return run_spine(f"size_{size}", train_participants=SWEEP_TRAIN, reps=SWEEP_REPS,
                     eval_participants=TEST, eval_reps=1, student_weights=weights)


def _sweep(jobs):
    import traceback
    for i, (name, fn) in enumerate(jobs, 1):
        print(f"\n########## [{i}/{len(jobs)}] {name} ##########", flush=True)
        try:
            fn()
        except Exception:
            print(f"!!!!! {name} FAILED (continuing) !!!!!", flush=True)
            traceback.print_exc()
    print("\n########## SWEEP COMPLETE ##########", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="baseline",
                    help="baseline | now (pscale+cscale) | pscale | cscale | "
                         "sizes | size_n/s/m")
    ap.add_argument("--long-run", action="store_true",
                    help="disable early-stop + checkpoint periodically (degradation curve)")
    args = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    if args.config == "baseline":
        cfg_baseline(args.long_run)
    elif args.config == "pscale":
        _sweep([(f"pscale_{n}", (lambda n=n: cfg_pscale(n))) for n in (1, 2, 4)])
    elif args.config == "cscale":
        _sweep([(f"cscale_{r}", (lambda r=r: cfg_cscale(r))) for r in (1, 3, 5, 10)])
    elif args.config == "now":          # core experiments at fixed 3000 frames
        _sweep([(f"pscale_{n}", (lambda n=n: cfg_pscale(n))) for n in (1, 2, 4)]
               + [(f"cscale_{r}", (lambda r=r: cfg_cscale(r))) for r in (1, 3, 5, 10)])
    elif args.config == "sizes":
        _sweep([(f"size_{s}", (lambda s=s: cfg_size(s))) for s in ("n", "s", "m")])
    elif args.config in ("size_n", "size_s", "size_m"):
        cfg_size(args.config.split("_")[1])
    else:
        raise SystemExit(f"unknown config {args.config}")


if __name__ == "__main__":
    main()
