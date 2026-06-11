---
name: feedback_keep_all_experiment_data
description: "Never auto-delete experiment data — keep all checkpoints, datasets, caches, metrics"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: bdb7162a-6c96-4391-8f59-078e4c60445f
---

For the drink_study (and experiments generally), **keep all data always** — never auto-delete artifacts to save space. The user was burned once when per-epoch `epochN.pt` checkpoints were deleted after scoring, then a new metric (inter-camera agreement) needed them.

**Why:** new metrics/analyses get added later and often need the original per-epoch weights / datasets / raw detections; regenerating is slow (re-labeling, re-training, re-running the slow teacher).

**How to apply:**
- Per-epoch checkpoints: keep (`train_with_metrics(keep_checkpoints=True)` is the default; don't flip it off).
- Datasets: cached via a `.complete` marker and reused, never re-labeled/wiped once built. `build_dataset`'s `shutil.rmtree` only runs on an unmarked (interrupted/fresh) build.
- Detection results: cached to `experiments/drink_study/cache/` so the slow teacher isn't re-run.
- Each config writes to its own `cfg_id` dir (`runs/<cfg_id>/`) — no cross-overwrite; metrics.json / eval_by_epoch.json / agreement.json / curves all persisted.
- If disk pressure ever forces cleanup, ask first and only purge after confirming every desired metric has been computed.
Relates to [[project_auto_pipeline]] and the drink_study harness.
