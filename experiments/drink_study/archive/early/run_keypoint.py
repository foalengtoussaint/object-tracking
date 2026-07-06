"""Centroid-only experiment: train a YOLO POSE (keypoint) student to predict just
the cup centroid, reusing the refill labelcache. Tests whether a simpler target
(point, not box/mask) reduces the on-body false positives seen on held-out P23.

CAVEAT: base model is yolo11n-pose (no yolo26-pose exists), so this is NOT a clean
head-only swap vs the seg students -- architecture differs too. Read accordingly.

Labels: convert each cached refill polygon -> pose line "0 cx cy w h  cx cy 2"
(bbox from polygon extent + ONE keypoint = the centroid, visibility=2). Reuses the
same images. Train on P01, eval detection-rate + gated 3D precision on P01 & P23.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import yaml
import run

REFILL_CACHE = run.DSROOT / "_labelcache" / "clean3d_fill_rf"
KP_CACHE = run.DSROOT / "_labelcache" / "keypoint_rf"
CFG = "pscale_1_keypoint"
BASE = "yolo11n-pose.pt"


def poly_to_pose(line: str) -> str | None:
    """'0 x1 y1 x2 y2 ...' polygon -> '0 cx cy w h  cx cy 2' (bbox + 1 keypoint)."""
    p = line.split()
    xs = [float(p[i]) for i in range(1, len(p), 2)]
    ys = [float(p[i]) for i in range(2, len(p), 2)]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    w, h = max(max(xs) - min(xs), 0.01), max(max(ys) - min(ys), 0.01)
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {cx:.6f} {cy:.6f} 2\n"


def build_pose_labels():
    """Convert refill polygon labels -> pose labels; symlink the same images."""
    done = KP_CACHE / ".done"
    pairs = []
    if done.exists():
        for jp in KP_CACHE.rglob("*.jpg"):
            pairs.append((jp, jp.with_suffix(".txt")))
        print(f"[{CFG}] reusing keypoint labelcache ({len(pairs)} pairs)", flush=True)
        return pairs
    n = 0
    for txt in REFILL_CACHE.rglob("*.txt"):
        rel = txt.relative_to(REFILL_CACHE)
        outdir = KP_CACHE / rel.parent
        outdir.mkdir(parents=True, exist_ok=True)
        pose = poly_to_pose(txt.read_text().strip())
        if pose is None:
            continue
        (outdir / txt.name).write_text(pose)
        jp_src = txt.with_suffix(".jpg")
        jp_dst = outdir / jp_src.name
        if not jp_dst.exists():
            jp_dst.symlink_to(jp_src.resolve())
        pairs.append((jp_dst, outdir / txt.name)); n += 1
        if n % 2000 == 0:
            print(f"  converted {n} labels ...", flush=True)
    done.write_text("ok")
    print(f"[{CFG}] built {len(pairs)} keypoint labels", flush=True)
    return pairs


def assemble_pose_dataset(pairs):
    import random, shutil
    ds = run.DSROOT / CFG
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            d = ds / sub / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
    pool = list(pairs); random.Random(0).shuffle(pool)
    ntr = min(run.TRAIN_FRAMES, len(pool))
    train, val = pool[:ntr], pool[ntr:ntr + run.VAL_FRAMES]
    for split, sel in (("train", train), ("val", val)):
        for ip, lp in sel:
            (ds / "images" / split / ip.name).symlink_to(ip.resolve())
            (ds / "labels" / split / lp.name).symlink_to(lp.resolve())
    y = {"path": str(ds.resolve()), "train": "images/train",
         "val": "images/val" if val else "images/train",
         "kpt_shape": [1, 3], "nc": 1, "names": ["cup"]}
    (ds / "data.yaml").write_text(yaml.safe_dump(y, sort_keys=False))
    return ds / "data.yaml", len(train)


def main():
    pairs = build_pose_labels()
    data_yaml, n = assemble_pose_dataset(pairs)
    print(f"training pose on {n} frames", flush=True)
    from ultralytics import YOLO
    model = YOLO(BASE)
    model.train(data=str(data_yaml), epochs=4, imgsz=640, batch=16,
                project=str(run.RUNS.resolve()), name=CFG, exist_ok=True, save_period=1)
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"DONE trained -> {best}", flush=True)
    print("KP_DONE", flush=True)


if __name__ == "__main__":
    main()
