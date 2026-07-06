"""Cache cup-centroid detections for a given model on EVERY drinking rep, ALL 10
cameras, using NVDEC (GPU) decode + streaming batched GPU inference.

Unlike cache_all_dets.py this:
  * takes any --model checkpoint + a --tag (separate cache dir, pscale_4 untouched)
  * decodes via gpu_decode (NVDEC) instead of CPU cv2 -> decode no longer the bottleneck
  * streams frames in batches (no full-clip buffering, no multiprocessing -> no OOM)
  * caches every camera clip that exists (fixes the P10/P03 5-cam gap)

Output: cache/student_dets_<tag>/{pid}_{stem}__<tag>__c{conf}.json
Resumable (skips cached). Verbose with per-rep ETA.

    python experiments/drink_study/cache_dets_model.py \
        --model /home/imove/drink_study_models/pscale_1_clean3d_fill/best.pt \
        --tag clean3d_fill
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import argparse, json, os, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import gpu_decode
from _paths import CLIPS_ROOT

BASE_RE = re.compile(r"^(P\d+)")
CONF = 0.25
CUP_CLASS = [0]


def list_reps(clip_dir: Path) -> list[str]:
    return sorted({f.name.rsplit(".", 2)[0]
                   for f in clip_dir.glob("*drinking_*_*.mp4")})


def detect_clip(model, path, conf, classes, batch):
    """Stream a clip via NVDEC; per-frame max-conf box center (or None)."""
    out, buf = [], []

    def flush():
        if not buf:
            return
        for r in model(buf, conf=conf, classes=classes, verbose=False):
            if r.boxes is not None and len(r.boxes) > 0:
                j = int(r.boxes.conf.argmax())
                b = r.boxes.xyxy[j].cpu().numpy()
                out.append((float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2)))
            else:
                out.append(None)
        buf.clear()

    for img in gpu_decode.frames(path):
        buf.append(img)
        if len(buf) >= batch:
            flush()
    flush()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--participants", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not CLIPS_ROOT.exists():
        raise SystemExit(f"clips root missing: {CLIPS_ROOT} (set OT_CLIPS_ROOT)")
    want = set(args.participants.split(",")) if args.participants else None
    outdir = Path("experiments/drink_study/cache") / f"student_dets_{args.tag}"
    outdir.mkdir(parents=True, exist_ok=True)

    # Dedup by (pid, stem): duplicate participant dirs like "P03 (1)" hold the
    # SAME stems as "P03" but with fewer cameras -> same cache key. Keep the dir
    # with the MOST cameras so we never let a 5-cam copy clobber the 10-cam one
    # (this collision is exactly what left P03/P10 at 5 cams in the old cache).
    best: dict[tuple[str, str], tuple[int, Path]] = {}
    for d in sorted(CLIPS_ROOT.iterdir()):
        if not d.is_dir():
            continue
        m = BASE_RE.match(d.name)
        if not m or (want and m.group(1) not in want):
            continue
        pid = m.group(1)
        for stem in list_reps(d):
            ncams = sum((d / f"{stem}.{c}.mp4").exists() for c in range(1, 11))
            key = (pid, stem)
            if key not in best or ncams > best[key][0]:
                best[key] = (ncams, d)
    jobs = sorted((pid, d, stem) for (pid, stem), (_, d) in best.items())

    cached = {f.name for f in outdir.glob(f"*__{args.tag}__c*.json")}
    todo = [(p, d, s) for (p, d, s) in jobs
            if f"{p}_{s}__{args.tag}__c{args.conf}.json" not in cached]
    print(f"[{args.tag}] decode={'GPU/NVDEC' if gpu_decode.gpu_available() else 'CPU'}  "
          f"jobs {len(jobs)} | cached {len(jobs)-len(todo)} | to run {len(todo)}", flush=True)
    if args.dry_run:
        return
    if not todo:
        print(f"[{args.tag}] nothing to do.", flush=True)
        print("CACHE_MODEL_DONE", flush=True)
        return

    from ultralytics import YOLO
    model = YOLO(args.model)
    print(f"[{args.tag}] model {args.model}  classes={model.names}", flush=True)

    t0 = time.time()
    for k, (pid, d, stem) in enumerate(todo, 1):
        cams = {c: d / f"{stem}.{c}.mp4" for c in range(1, 11)
                if (d / f"{stem}.{c}.mp4").exists()}
        el = time.time() - t0
        eta = (el / (k - 1) * (len(todo) - k + 1)) if k > 1 else 0
        print(f"[{args.tag}][{k}/{len(todo)}] {pid} {stem} ({len(cams)} cams)  "
              f"elapsed {el/60:.1f}m eta {eta/60:.1f}m", flush=True)
        dets = {}
        for c, path in sorted(cams.items()):
            dets[f"cam_{c}"] = detect_clip(model, path, args.conf, CUP_CLASS, args.batch)
        cf = outdir / f"{pid}_{stem}__{args.tag}__c{args.conf}.json"
        cf.write_text(json.dumps(dets))
        print(f"    -> {cf.name}  ({sum(sum(x is not None for x in v) for v in dets.values())} dets)",
              flush=True)
    print(f"[{args.tag}] ALL DONE: {len(todo)} reps in {(time.time()-t0)/60:.1f}m", flush=True)
    print("CACHE_MODEL_DONE", flush=True)


if __name__ == "__main__":
    main()
