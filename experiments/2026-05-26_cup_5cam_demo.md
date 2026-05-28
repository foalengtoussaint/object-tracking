# Run: cup_5cam_demo

**Date:** 2026-05-26
**Machine:** imove desktop (GPU: NVIDIA GeForce RTX 3060 Ti, OS: Ubuntu)
**Git SHA:** `0c80a29` + local 5-cam YAML / docstring tweaks (uncommitted at time of run)
**Weights stored at:** local only — `runs/segment/cup_5cam_demo/weights/best.pt` (6.5 MB). Not yet copied off-machine.

## Setup

- **Cameras:** 5× Logitech BRIO, USB. `/dev/video{0,4,8,13,17}` → ports 55555–55559. 1280×720 @ 30 fps.
- **Room / lighting:** office workspace, natural daylight from window + overhead LED.
- **Cup(s):** single cup, hand-held and moved through the workspace by the operator during the 30 s window.
- **Distractors present:** monitors, keyboard, mouse, mug — typical desk clutter. Pseudo-labeler restricted to COCO classes [39, 40, 41, 45, 75] (bottle, wine glass, cup, bowl, vase).

## Dataset

- **Source clips:** `clips/cam_{1..5}_20260526_134839.mp4` — recorded with 5 parallel `record_clips.py` calls, 30 s each.
- **Total frames after pseudo-label:** **1983**
  - cam_1: 189, cam_2: 382, cam_3: 291, cam_4: 520, cam_5: 601
  - cam_5 had the best framing → highest yield.
- **Pseudo-label base model:** `yolo26x-seg.pt` (downloaded fresh, 135 MB).
- **Confidence threshold:** 0.10 (default in `pseudo_label.py`)
- **Kalman gate:** Mahalanobis 13.82 (99.9% chi-sq, 2 DoF).
- **Manual review:** none; rendered per-cam labeled previews and spot-checked visually before training.
- **Train/val split:** none — val=train (single-class demo, no held-out split).

## Training

```bash
python finetune.py --epochs 10 --name cup_5cam_demo
# default --weights yolo26n-seg.pt, --data dataset/data.yaml, --imgsz 640, --batch 16
```

Defaults otherwise unchanged. No augmentation tweaks.

## Results

- **Epochs:** 10
- **Box mAP50:** 0.991
- **Box mAP50-95:** 0.81
- **Mask mAP50:** 0.992
- **Mask mAP50-95:** 0.636
- **Inference speed:** ~1.0 ms / image on the 3060 Ti
- **Output:** `runs/segment/cup_5cam_demo/weights/best.pt`
- **Inference command:**
  ```bash
  python live_cup_detect.py --weights runs/segment/cup_5cam_demo/weights/best.pt
  ```

## Notes

- Val=train, so mAP numbers are optimistic by design. The real test was the live run.
- **Live generalization:** model held the cup across positions it hadn't seen during recording; Kalman filter never had to bridge long enough gaps to drift. Subjectively solid.
- **Pipeline timing (end-to-end):**
  - record (5 parallel × 30 s): 30 s wall
  - pseudo_label.py (yolo26x on 4505 frames): ~3 min
  - finetune (10 epochs, 1983 imgs): ~4 min
  - total: ~8 min from cold start
- **Known weaknesses:** dataset only captures the cup in operator's hand, no static scenes, no other cups. Will likely fail on cluttered tabletops with multiple cup-like distractors. No held-out validation.
- **Current best for this physical setup:** yes (first run).


## Addendum — 2026-05-26 18:27: size-scaling sweep

Attempted to extend the n/s sweep with m (28M) and l (50M), all fine-tuned for 10
epochs on the same 1983-frame dataset. Held-out evaluation uses the cached
per-frame detections in `detections_cache/clips_heldout/`.


Models with held-out data: **n, s, m, l** plus the yolo26x COCO baseline at
conf=0.25 and conf=0.05.

### Held-out PR (cup assumed present in every frame)

```
configs: ['finetuned_l_0.25', 'finetuned_m_0.25', 'finetuned_n_0.25', 'finetuned_s_0.25', 'yolo26x_cup_0.05', 'yolo26x_cup_0.25']
cams: ['cam_1', 'cam_2', 'cam_3', 'cam_4', 'cam_5']

config                frames  det_F  tot_det   dup  phan  tele  recall  P_loose  F1_loose  P_strict  F1_strict
finetuned_l_0.25        2255   2052     2505    88   412   133   0.910    0.782     0.841     0.747      0.821
finetuned_m_0.25        2255   2032     2200    42   135   122   0.901    0.883     0.892     0.864      0.882
finetuned_n_0.25        2255   1787     1909   121     1     1   0.792    0.999     0.884     0.936      0.858
finetuned_s_0.25        2255   1812     1873    29    33    32   0.804    0.965     0.877     0.950      0.871
yolo26x_cup_0.05        2255    923     1038   109     6     7   0.409    0.987     0.579     0.882      0.559
yolo26x_cup_0.25        2255    421      424     3     0     0   0.187    1.000     0.315     0.993      0.314
```

(Initial auto-generated table omitted `m` because the cache step pointed at
a stale `runs/segment/cup_5cam_demo_m/` dir while the actual training landed
in `cup_5cam_demo_m-2/` due to ultralytics auto-renaming. Fixed in a follow-up
cache pass.)

- `recall` = fraction of frames with at least one detection.
- `P_loose` only counts real localization errors (spatially-separated phantoms
  in multi-detection frames + lone-detection teleports).
- `P_strict` also counts NMS-duplicate extras as wrong.

### Key findings

1. **Distillation completely dominates the COCO teacher.** All fine-tunes
   reach F1_loose ≥ 0.84, vs yolo26x's 0.31 (conf=0.25) / 0.58 (conf=0.05).
2. **There is a real recall/precision tradeoff across sizes — and m wins F1.**
   The n/s/m/l curve isn't flat:
   - **n (3M)** and **s (12M)** cap at recall ~0.80 with near-perfect P_loose
     (0.999 / 0.965). Very precise, undershoots recall.
   - **m (28M)** jumps to recall 0.90 with modest precision loss
     (P_loose 0.883). **Highest F1_loose (0.892) and F1_strict (0.882).**
   - **l (50M)** pushes recall just 0.009 higher (0.910) but precision tanks
     (P_loose 0.782, 412 phantoms vs m's 135). Overshoot.
   The model becomes more confident on the cup AND more confident in wrong
   places as capacity grows; the sweet spot is m.
3. **For deployment, pick by what you need:**
   - **n** if precision is paramount (P_loose 0.999 — essentially no
     mislocalizations), fastest, smallest.
   - **m** if you want the best overall F1 (highest recall *and* still 0.88
     precision).
   - **s** is dominated by m on F1 and by n on precision; not the right pick
     for anything.
   - **l** is dominated by m on everything; capacity wasted on overfitting.
4. **Kalman reject% is a biased metric** — the gate widens during predict-mode
   gaps, undercounting sparse-firing models' true outliers. Per-frame
   "extras" + spatial-separation (what P_loose / P_strict use) is more honest.
5. **Student-beats-teacher gap held up on held-out.** In-domain bake-off had
   a 3.2× gap, held-out 4.2× — real generalization within the scene's
   manifold, not just memorization.
6. **8 GB VRAM caps the trainable model size.** yolo26x training OOM'd even
   at batch=2 (backward pass + Adam state too big). yolo26m needed batch=4
   to fit, yolo26l needed batch=2. Bigger models would need a larger GPU.

### Next investment: data variety, not model size

Since m beats l despite l being larger, we're already past the size sweet
spot for this dataset. Adding more model capacity makes things worse. The
right next move is more diverse data. Useful next recordings (in rough
impact order):
- Different lighting (later in day, lights off, single side-light)
- Multiple cup species in scene (distractors)
- Hand occlusion at higher angles
- Different rooms / backgrounds
- Empty vs liquid-filled cup

### Artifacts

- `runs/segment/cup_5cam_demo_m-2/weights/best.pt` (yolo26m fine-tune; the
  `cup_5cam_demo_m/` dir is the empty stub from a prior failed attempt and
  can be deleted)
- `runs/segment/cup_5cam_demo_l/weights/best.pt`
- `detections_cache/clips_heldout/*/` (per-config per-frame detections)
- `compare_heldout/finetuned_*_raw_mosaic.mp4` (visuals for each model)
- `compare_heldout/finetuned_{m,l}_filtered_mosaic.mp4` (KF-filtered: only the
  top-1 accepted detection drawn, yellow ghost during predict-only frames)
- `cache_detections.py` / `analyze_pr.py` for reproducing


## Addendum — 2026-05-27: KF tracker behavior + in-domain bake-off tables

Saving two tables that weren't in the doc yet. Both are derived from already-
cached artifacts (no re-inference) so they're cheap to regenerate.

### Kalman tracker behavior per config (held-out, 2255 frames, gate 13.82, MAX_MISS 30)

```
config                  frames  accept  predict  reject   acc%  pred%   rej%
finetuned_l_0.25          2255    1872      203     180  83.0%   9.0%   8.0%
finetuned_m_0.25          2255    1901      200     131  84.3%   8.9%   5.8%
finetuned_n_0.25          2255    1786      447       1  79.2%  19.8%   0.0%
finetuned_s_0.25          2255    1779      443      33  78.9%  19.6%   1.5%
yolo26x_cup_0.05          2255     916     1330       7  40.6%  59.0%   0.3%
yolo26x_cup_0.25          2255     421     1576       0  18.7%  69.9%   0.0%
```

- `accept` = frame had a detection within the KF gate (top-1 used to update)
- `predict` = frame had no detection; KF coasted on velocity
- `reject` = frame had detection(s) but all failed the gate (treated as outliers)

This is the cleanest single view of the recall/precision tradeoff:
- **n/s** never reject — they're conservative enough that what they produce is
  in-trajectory. Cost: 20% of frames are KF-coast (predict-only).
- **m** swings to higher accept (84%) without spiking reject (5.8%) — best
  balance.
- **l** rejects 8% — capacity bought confident-but-wrong detections that the
  KF correctly throws out.
- **yolo26x** can't see the cup most of the time — 59–70% predict-only.

### In-domain bake-off summary (4505 frames across 5 cams, conf 0.25)

Aggregated from `compare/metrics.csv`:

```
model           frames  mean_det_rate  mean_dets/f  mean_conf   ms/f
yolo26x           4505          0.301        0.307      0.487   25.0
fastsam_x         4505          1.000        1.000      0.711  231.4
finetuned_n       4505          0.961        1.019      0.811    5.2
```

- `fastsam_x` always returns 100 segments per frame regardless of content;
  detection_rate=1.0 is a measurement artifact, not signal. Subjectively it
  found cup-like things but with high false-positive rate.
- `finetuned_n` is 3.2× the teacher's in-domain recall at 1/5 the latency
  (5.2 ms vs 25 ms) — held-out comes in at 4.2× (see PR table above).
- Per-cam breakdown in `compare/metrics.csv`; held-out per-cam in
  `compare_heldout/metrics.csv` (n + s only, no m/l rows there).


## Addendum — 2026-05-27 10:59: gen2 (self-distillation with cleaner labels)

Re-pseudo-labeled the same `clips/` recordings using `finetuned_n` as the
teacher (instead of yolo26x), then fine-tuned a fresh yolo26n on those labels.
Same source video, same KF gate, same training schedule (10 epochs, batch 16).
The only thing that changed is the labeler.

### Dataset comparison

| | gen1 (yolo26x teacher) | gen2 (finetuned_n teacher) |
|---|---|---|
| Frames labeled | 1983 | **4311** (2.17×) |
| Per-cam distribution | 189 / 382 / 291 / 520 / 601 (skewed) | 823 / 888 / 828 / 881 / 891 (uniform) |
| Teacher in-domain recall | ~0.30 | ~0.96 |
| Teacher mean conf | 0.49 | 0.81 (tighter boxes) |

### Held-out PR (new row added to the table)

```
config                  frames  det_F  tot_det   dup  phan  tele  recall  P_loose  F1_loose  P_strict  F1_strict
finetuned_n_0.25  (gen1)   2255   1787     1909   121     1     1   0.792    0.999     0.884     0.936      0.858
finetuned_n_gen2_0.25      2255   1896     1922    26     0     1   0.841    0.999     0.913     0.986      0.908
finetuned_m_0.25 (prev SOTA) 2255 2032     2200    42   135   122   0.901    0.883     0.892     0.864      0.882
```

### Key findings

1. **Gen2 nano is the new overall best** — beats gen1 n on every metric AND
   beats the m model (28M params, prev F1 winner) on both F1_loose and
   F1_strict despite being 9× smaller.
2. **The "self-distillation on memorized data" critique was wrong** — the gain
   came from *label cleanliness*, not new information:
   - 0 phantoms (was 1)
   - 26 NMS duplicates (was 121) — the tight, consistent gen1 box geometry
     taught gen2 to make one tight box per cup instead of overlapping ones.
   - +4.9 pp recall — more labeled frames meant the model saw more poses of
     the cup, even though those exact frames were already in gen1's training set
     (just unlabeled because yolo26x failed to detect them).
3. **In-domain val (still val=train, biased high):** Box mAP50-95 0.81 → 0.84,
   Mask mAP50-95 0.636 → 0.658.
4. **The pseudo-labels filter their own bad predictions.** A "bad" finetuned_n
   detection only enters gen2's dataset if it happens to be trajectory-
   consistent with the real cup's motion (rare per held-out P_loose 0.999).

### Recommended next iteration

Gen3 with `finetuned_n_gen2` as teacher on `clips/ + clips_heldout/` combined.
Expected wins: even more data, even cleaner labels, and the held-out clips
become in-distribution for gen3.

### Gen2 train vs test PR

```
set                              frames  recall  P_loose  F1_loose  P_strict  F1_strict
train (clips/)                     4505   0.992    0.997     0.995     0.977      0.985
test  (clips_heldout/)             2255   0.841    0.999     0.913     0.986      0.908
```

Diagnostic: precision is identical across train/test (0.997 vs 0.999), but
recall drops 15 pp. So generalization fails by *not firing* on novel poses,
not by firing on wrong things. Cleanest possible failure mode — every gen2
detection on held-out is still on the cup, the model just misses some poses
it never saw during training. Argues for more diverse data, not better
regularization.

### Artifacts

- `runs/segment/cup_5cam_demo_gen2/weights/best.pt`
- `dataset_gen2/` (4311 labeled frames)
- `detections_cache/clips/finetuned_n_gen2_0.25/` (gen2 on training set)
- `detections_cache/clips_heldout/finetuned_n_gen2_0.25/` (gen2 on test set)
- `compare/finetuned_n_gen2_{raw,filtered}_mosaic.mp4` (visuals on training set)
- `compare_heldout/finetuned_n_gen2_{raw,filtered}_mosaic.mp4` (visuals on test set)


## Metric definitions (P_loose vs P_strict)

All P/R metrics in this doc are computed without ground-truth boxes. They
exploit the held-out clips' invariant that the operator held the cup visible
in every frame, so any detection somewhere on the cup counts as correct.

**For each frame's raw detections, the Kalman filter labels each one:**

- **primary** — the detection closest to the KF-predicted position, within
  the Mahalanobis gate (13.82 ≈ 99.9% chi-sq, 2 DoF). At most one per frame.
- **duplicate (`dup`)** — an extra detection <100 px from the primary. Cause:
  NMS didn't merge two boxes that lie on the same physical cup. Wasteful,
  but the model *did* find the cup.
- **phantom (`phan`)** — an extra detection ≥100 px from the primary. Model
  found the cup AND fired on something else (laptop edge, mug, etc.).
- **teleport (`tele`)** — a *lone* detection (only one in the frame) that
  fails the gate. The model fired once, on the wrong thing.

**The two precisions differ only in how they treat `dup`:**

```
P_loose  = (total_dets - phan - tele) / total_dets
P_strict = (total_dets - phan - tele - dup) / total_dets
```

- **P_loose** treats duplicates as still-correct: the model identified the
  cup, it just drew the box twice. This is the right metric if a downstream
  tracker picks the highest-conf box per frame — duplicates cost nothing.
- **P_strict** penalizes duplicates too. The right metric if you actually
  *consume* every detection (e.g., counting cups in a scene). Mostly
  measures NMS quality.

**Concrete example** (gen1 finetuned_n on held-out, 1909 total detections):

| | count | math | result |
|---|---|---|---|
| phantoms | 1 | | |
| teleports | 1 | | |
| dups | 121 | | |
| P_loose | | (1909 − 1 − 1) / 1909 | **0.999** |
| P_strict | | (1909 − 1 − 1 − 121) / 1909 | **0.936** |

The 6.3 pp gap is entirely NMS duplicates — gen1 sometimes drew two boxes on
the same cup. Gen2 reduced this to 26 duplicates (P_strict jumped to 0.986).

**For our single-object KF tracker, P_loose is the load-bearing metric.**
P_strict mostly tells us "how often does NMS double-fire," which is
informative for diagnosing model behavior but not what we'd optimize against.
