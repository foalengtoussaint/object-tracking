---
name: session-20260618-cache-and-outliers
description: "Handoff for the 2026-06-18 session — full pscale_4 det cache, 3D trajectory cache, trajectory-outlier analysis, P23 characterization, calibration gap"
metadata:
  type: project
---

# Session handoff — 2026-06-18 (detection cache, 3D trajectories, outliers)

Narrative continuity for a fresh Claude Code instance on another machine. Conclusions
are also in the individual memory notes; this is the blow-by-blow + open tasks.

## What was done
1. **Offline Rerun replay from cache** (`viz_replay.py`): replayed P01 then P19 from
   the cached pscale_4 detections (no GPU/video). Had to `pip install rerun-sdk==0.32.2`
   into the `.venv` (wasn't installed on this machine). P19 served at
   `http://127.0.0.1:9090/?url=rerun%2Bhttp%3A%2F%2F127.0.0.1%3A9876%2Fproxy`.

2. **Full pscale_4 detection cache** — see [[project_pscale4_full_det_cache]].
   `cache_all_dets.py` ran the pscale_4 student on **370 distinct drinking_right reps,
   all 23 participants** (P01–P24, no P22), all 10 cams → `cache/student_dets/`.
   ~12.5h serial, batched (`agreement.cup_centroids_batched`, batch=64, ~2.5x vs
   per-frame, verified identical: None-pattern exact, centroids <0.06px). Clips live at
   `/home/imove-laptop-01/object_tracking_data/clips` (set `OT_CLIPS_ROOT`).

3. **3D trajectory cache** — `cache_track3d.py` ran consensus→causal KF→RTS on the
   **67 calibrated reps** (P01/P06/P19/P23) → `cache/track3d/{P}_{stem}__pscale_4.json`
   (per-frame consensus/kf/rts xyz + kept-cams + median_px) + `_summary.json`. ~3.4 min,
   no GPU. Per-participant: tri_rate 0.93–0.95, median_px 4.3–4.9px.

4. **Trajectory-outlier analysis** (3 scripts):
   - `plot_track3d_outliers.py`: 4 scalar features (path_len, lift, vol, peak_speed),
     robust-z (MAD). Found 0 motion outliers, 4 quality outliers (P01 early reps low
     tri_rate; P23 high px).
   - `plot_track3d_shape.py`: SHAPE outliers, paths normalized to first cup position
     (`traj - traj[0]`); arc-length resample→PCA(2D)+Mahalanobis AND self-contained
     numpy DTW (no dep). Cross-check.
   - `plot_p23_character.py`: characterized P23.

5. **P23 finding** (the headline): DTW flagged ~14/19 P23 reps, but PCA flagged none —
   that pattern = a GROUP difference, not individual anomalies. DTW's "mean distance to
   all others" conflates "rare" with "member of a minority cluster". P23 is a **faster,
   shallower drink**: lift 226mm vs 301 (z=-3.2), duration 6.4s vs 9.0 (z=-2.0), peak
   speed 1102 vs 767 mm/s (z=+1.8), similar path length. Box+jitter shows whole-group
   shift, not fliers. Caveat: P23 is also loosest-px (z=+1.0) so peak_speed partly
   measurement noise from fast motion; lift/duration are robust.

6. **Committed to master** (02de420, NOT pushed as of handoff): the 2 caches, 5 scripts,
   3 plots (`track3d_outliers.png`, `track3d_shape.png`, `p23_character.png`),
   `agreement.py` batched fns, memory notes. Logs + unrelated untracked (teleimager/,
   gigapose/, percam/) deliberately excluded.

## Gotchas hit this session
- **Parallel-decode OOM'd VSCode**: `cache_all_dets.py --workers>1` (spawn) had each
  worker re-import torch+CUDA and buffer 1080p clips in 20GB RAM → OOM killed the VSCode
  extension host. ABANDONED that path; serial batched is the safe one. GPU decode (NVDEC)
  not available in `.venv` (torchvision 0.27 dropped VideoReader; no decord/ffmpeg-cuvid).
- cv2.VideoCapture = CPU software H.264 decode, single-threaded per file = the real
  bottleneck (GPU sat at 5% per-frame). Batching got GPU to ~56%.

## Open tasks / next steps
- **Calibrate the other ~19 participants** (BLOCKED): the 4 calib TOMLs are *distinct
  per participant* (rig re-calibrated each session — extrinsics differ), so one can't be
  reused. Need each session's Charuco capture footage (`run_calibration.py`) or the TOMLs
  themselves — both on the iMOVE DATA SSD that is NOT mounted on this machine right now.
  Until then the other participants have only 2D detections cached, no 3D tracks.
- **Within-participant outlier rerun**: score each rep against ITS OWN participant's reps
  (within-group DTW) to find true individual anomalies, removing the P23-minority artifact.
- **Verify P23 visually**: render P23 vs P19 consensus video side by side to confirm the
  fast-shallow drink is real motion not jitter (needs clips, on disk).
- Standing drink_study threads: confident-wrong guard, fix agreement metric — see
  [[project_failure_modes_confident_wrong.md]]; agreement ≠ correctness still applies to
  all these 3D tracks.

## Reminder: agreement ≠ correctness
tri_rate / median_px / consensus measure that cameras AGREE on a 3D point, not that it's
the cup. A confidently-wrong self-consistent on-body track scores well. Holds for all
track3d outputs.
