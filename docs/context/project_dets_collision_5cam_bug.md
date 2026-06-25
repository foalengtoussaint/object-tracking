---
name: project_dets_collision_5cam_bug
description: "drink_study 5-cam detection-cache gap (P03/P10) was a duplicate-dir cache-filename COLLISION, not missing footage; cache_all_dets.py still has the bug; cache_dets_model.py fixed via max-camera dedup"
metadata:
  type: project
---

The "P03/P10 only have 5-camera detections" gap (see [[project_pscale4_full_det_cache]])
was **not** missing clips — it was a cache-filename collision.

Clips live in `P03` AND `P03 (1)` (and `P10`/`P10 (1)`). The `(1)` dirs hold the
**same rep stems** but with only 5 cameras. Both dirs map to base pid `P03`, and the
cache key is `{pid}_{stem}__...` — so the 5-cam `(1)` copy and the 10-cam main copy
write the **same filename**. Dirs are processed in sorted order (`P03` before
`P03 (1)`), so the 10-cam version is written first and the **5-cam version overwrites
it**. Net: P03/P10 ended up 5-cam everywhere they had a `(1)` dir.

This is why P10 looked like the "worst-tracked participant" (low tri_rate, cup-lost-at-
mouth failures) — it was triangulating from half the cameras the whole time. The 10-cam
clips were always present in the main dir.

- **`experiments/drink_study/cache_all_dets.py` still has this bug** — re-running it would
  recreate the 5-cam gap. Apply the same dedup fix there.
- **Fix (in `cache_dets_model.py`):** dedup by `(pid, stem)` keeping the dir with the MOST
  cameras, so a partial copy can never clobber the full one. Verified P10→all 38 reps
  10-cam, P03→29 at 10-cam + 3 genuinely-5-cam (drinking_left reps only in the dup dir).
- **P12 (5-cam) and P21 (5-cam) are GENUINE** 5-camera sessions (only 5 ever recorded),
  not collisions — leave them.

Related: [[project_kf_accuracy_budget]] (≥3 well-spread cams needed — 5 vs 10 cams matters).
