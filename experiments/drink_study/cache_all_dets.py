"""Batch-cache pscale_4 student cup-centroid detections on EVERY drinking_right rep,
all participants, all 10 cams -- the same model/format the replay (viz_replay.py) and
kf_accuracy.py consume, so everything downstream reuses these JSONs with no GPU.

- One cache file per rep: cache/student_dets/{P}_{stem}__pscale_4__c0.25.json
- Reuses kf_accuracy.student_dets (skips reps already cached -> resumable).
- Participant dirs like "P03 (1)" are the SAME participant, other trials: we map the
  clip dir to its base participant id (P03) for the cache key, but enumerate stems from
  the actual dir so distinct timestamped reps are all captured.
- 2D only: triangulation/consensus needs calibration.toml (only P01/P06/P19/P23 have it).
  Detections for the rest are cached for later once calib arrives.

VERBOSE by design (long run): prints per-rep and per-cam progress with flush=True.

    OT_CLIPS_ROOT=/path/to/clips python experiments/drink_study/cache_all_dets.py
    ... --participants P01,P06          # subset
    ... --dry-run                       # list what WOULD run, no GPU
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("OMP_NUM_THREADS", "8")          # avoid futex-spin hang
os.environ.setdefault("OPENCV_FOR_THREADS_NUM", "4")

import kf_accuracy as ka
from _paths import CLIPS_ROOT

BASE_RE = re.compile(r"^(P\d+)")                        # "P03 (1)" -> "P03"


# --- parallel decode: workers decode clips on the CPU, main infers on the GPU ---
# Decode is single-threaded per file and is THE bottleneck (CPU H.264, not the GPU).
# We hand each clip to a worker process that decodes it into chunks of JPEG-light
# numpy frames and streams them back; the main process owns the one GPU model and
# runs batched inference on each chunk as it arrives. Frames cross the process
# boundary as uint8 arrays (a chunk of 64 1080p frames ~ 380MB -> keep chunk small).

def _decode_count(path):
    """Worker: decode a clip -> total frame count only (cheap probe, not used for dets)."""
    import cv2
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(__import__("cv2").CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def _decode_clip(job):
    """Worker: decode ONE clip fully -> (rep_idx, cam, [frame,...]).

    Bounded by the pool size (only `workers` clips in flight at once). A 1080p
    ~500-frame clip is ~3GB; with workers<=6 and 20GB free that stays in budget.
    """
    import cv2
    rep_idx, cam, path = job
    cv2.setNumThreads(1)                                # one core per worker; parallelism is across files
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return rep_idx, cam, frames


def _infer_frames(model, frames, conf, classes, batch):
    out = []
    for i in range(0, len(frames), batch):
        for r in model(frames[i:i + batch], conf=conf, classes=classes, verbose=False):
            if r.boxes is not None and len(r.boxes) > 0:
                j = int(r.boxes.conf.argmax())
                b = r.boxes.xyxy[j].cpu().numpy()
                out.append((float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2)))
            else:
                out.append(None)
    return out


def base_pid(dirname: str) -> str | None:
    m = BASE_RE.match(dirname)
    return m.group(1) if m else None


def list_reps(clip_dir: Path) -> list[str]:
    """Distinct drinking_right rep stems in a clip dir (any hand)."""
    return sorted({f.name.rsplit(".", 2)[0]
                   for f in clip_dir.glob("*drinking_*_*.mp4")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", default=None,
                    help="comma list of base ids to restrict to (default: all)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--batch", type=int, default=64,
                    help="frames per GPU inference batch (0 = per-frame, slow)")
    ap.add_argument("--workers", type=int, default=5,
                    help="parallel CPU decode workers (1 = serial decode). RAM-bounded.")
    args = ap.parse_args()

    if not CLIPS_ROOT.exists():
        raise SystemExit(f"clips root missing: {CLIPS_ROOT} (set OT_CLIPS_ROOT)")
    want = set(args.participants.split(",")) if args.participants else None

    # collect (base_pid, clip_dir, stem) jobs across all (including "(1)") dirs
    jobs: list[tuple[str, Path, str]] = []
    for d in sorted(CLIPS_ROOT.iterdir()):
        if not d.is_dir():
            continue
        pid = base_pid(d.name)
        if pid is None or (want and pid not in want):
            continue
        for stem in list_reps(d):
            jobs.append((pid, d, stem))

    cached = {f.name for f in ka.DETCACHE.glob("*__pscale_4__c*.json")} if ka.DETCACHE.exists() else set()
    todo = [(pid, d, s) for (pid, d, s) in jobs
            if f"{pid}_{s}__pscale_4__c{ka.CONF}.json" not in cached]
    print(f"jobs: {len(jobs)} reps total | already cached: {len(jobs)-len(todo)} | to run: {len(todo)}",
          flush=True)
    if args.dry_run:
        for pid, d, s in todo:
            print(f"  WOULD run [{pid}] {s}  (dir={d.name})", flush=True)
        return
    if not todo:
        print("nothing to do -- all cached.", flush=True)
        return

    from agreement import detect_rep, detect_rep_batched
    from ultralytics import YOLO
    model = YOLO(ka.STUDENT)                            # load ONCE on the GPU, reuse across reps
    ka.DETCACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.workers <= 1:
        # --- serial decode (batched inference) ---
        print(f"inference: batched x{args.batch}, serial decode", flush=True)
        for k, (pid, d, stem) in enumerate(todo, 1):
            cf = ka.DETCACHE / f"{pid}_{stem}__pscale_4__c{ka.CONF}.json"
            rep = {f"cam_{c}": d / f"{stem}.{c}.mp4" for c in range(1, 11)
                   if (d / f"{stem}.{c}.mp4").exists()}
            el = time.time() - t0
            eta = (el / (k - 1) * (len(todo) - k + 1)) if k > 1 else 0
            print(f"[{k}/{len(todo)}] {pid} {stem}  ({len(rep)} cams)  "
                  f"elapsed {el/60:.1f}m  eta {eta/60:.1f}m", flush=True)
            dets = detect_rep_batched(model, rep, ka.CONF, ka.CUP_CLASS,
                                      verbose=True, batch=args.batch)
            cf.write_text(json.dumps(dets))
            print(f"    -> cached {cf.name}", flush=True)
        print(f"ALL DONE: cached {len(todo)} reps in {(time.time()-t0)/60:.1f}m", flush=True)
        print("CACHE_ALL_DONE", flush=True)
        return

    # --- parallel CPU decode + single-GPU batched inference ---
    import multiprocessing as mp
    # flat decode-job list across all todo reps; remember each rep's cam set + path
    rep_meta = {}      # rep_idx -> (pid, stem, cf, ncams)
    jobs_flat = []
    for ri, (pid, d, stem) in enumerate(todo):
        cf = ka.DETCACHE / f"{pid}_{stem}__pscale_4__c{ka.CONF}.json"
        cams = [(c, d / f"{stem}.{c}.mp4") for c in range(1, 11)
                if (d / f"{stem}.{c}.mp4").exists()]
        rep_meta[ri] = (pid, stem, cf, len(cams))
        for c, path in cams:
            jobs_flat.append((ri, f"cam_{c}", path))
    pending = {ri: {} for ri in rep_meta}              # rep_idx -> {cam: dets}
    done_reps = 0
    print(f"inference: batched x{args.batch}, {args.workers} parallel decode workers; "
          f"{len(jobs_flat)} clips across {len(todo)} reps", flush=True)
    ctx = mp.get_context("spawn")                       # avoid forking CUDA state
    with ctx.Pool(args.workers, maxtasksperchild=8) as pool:
        for ri, cam, frames in pool.imap_unordered(_decode_clip, jobs_flat):
            dets = _infer_frames(model, frames, ka.CONF, ka.CUP_CLASS, args.batch)
            del frames                                  # free the decoded clip immediately
            pending[ri][cam] = dets
            pid, stem, cf, ncams = rep_meta[ri]
            if len(pending[ri]) == ncams:               # rep complete -> write cache
                cf.write_text(json.dumps(pending.pop(ri)))
                done_reps += 1
                el = time.time() - t0
                eta = el / done_reps * (len(todo) - done_reps)
                print(f"[{done_reps}/{len(todo)}] cached {pid} {stem} ({ncams} cams)  "
                      f"elapsed {el/60:.1f}m  eta {eta/60:.1f}m", flush=True)
    print(f"ALL DONE: cached {done_reps} reps in {(time.time()-t0)/60:.1f}m", flush=True)
    print("CACHE_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
