---
name: project_qtm_cup_mocap
description: "Qualisys cup mocap — 772 labeled C3D drinking trials = sub-mm 6-DoF cup ground truth, cached on Linux; loader + inventory built"
metadata: 
  node_type: memory
  type: project
  originSessionId: 25fd7658-7e75-4292-a8ad-32fbb99160c0
---

Optical motion capture for the drink study is **cup ground truth**, not body pose.
Source: Windows QTM partition `/dev/nvme1n1p3` (NTFS, mount read-only at /mnt/win3),
`Users/ccedg/Documents/Qualisys/Data/_drinking_all/` (also per-participant Pxx/ dirs
+ `_AIM_training/`). The 4 cup markers come from QTM's AIM auto-labeling.

**Copied to** `experiments/drink_study/cache/qtm_c3d/` (772 `.c3d`, 43 MB) so it's
independent of the Windows mount. No reboot to Windows needed once mounted+copied.

**Contents (all 772, uniform):** markers `cupdl, cupdr, cupul, cupur` (down/up ×
left/right corners of the cup), in **mm**, QTM lab frame. 4 rigid markers ⇒ full
6-DoF cup pose (cup proven rigid: dl–dr edge 25.6 mm, std 0.39 mm). Cup-only — NO
arm/body markers, so it's cup GT for the YOLO→triangulate→consensus-KF tracks, not a
pose GT for MeTRAbs.

**Gotchas the loader carries per-trial (do NOT assume):**
- Two frame rates: **698 @ 100 Hz, 74 @ 120 Hz** — read `rate` per file.
- Mostly gap-filled to 0 missing, but **7 trials >2% missing** (worst P060019 24.9%,
  P13_0033 21%, P210027 18%). These degrade speed curves; re-clean in QTM if needed.
- Participants P02–P24 present; **no P01, no P22**. The 30 `DTL/DTR####` files are
  left/right sets with no participant id in the name.

**Trajectories verified physically sane** (`cache/qtm_traj_sanity.png`): trapezoid
height profile (rest→lift→drink dwell plateau→return) + classic **two-peak speed
curve** with the dwell at the speed minimum — exactly the signal [[project_cup_only_segmentation]]
and [[project_cup_pose_fusion]] segment on. ~350–400 mm lift, ~900–1100 mm/s peak.

**Tooling:** `experiments/drink_study/qtm_c3d.py` — `load_trial(stem)` → `CupTrial`
(`.markers (T,M,3)`, `.centroid()`, `.pose()`→(R,t)), `list_trials()`, `inventory()`.
`_paths.py` adds `QTM_C3D` (override `OT_QTM_C3D`). Inventory JSON:
`cache/qtm_inventory.json`. Needs `ezc3d` (pip-installed into the `idrink` env).

**SOLVED — C3D↔video mapping is a deterministic POSITIONAL join** (`qtm_video_map.py`,
`build_map()` → `cache/qtm_video_map.json`). Mocap = one QTM take per sip, C3D
run-numbered ascending = chronological. Video annotation `clean_data.xlsx`
(`~/Documents/annotation_tool/`, 60 fps) lists every sip as its own row but the
**"task name" column is SPARSE/forward-filled** — a task name appears once, then
blank-task rows belong to it until the next named task. So the drinking rows (in
sheet order = chronological, with start/end frame + side + valid) pair 1:1 with
sorted C3D where counts match: `sorted_c3d[k] ↔ drink_row[k]`. NO content matching
needed. Verified by per-sip duration correlation video-vs-C3D **r=0.995 mean**
(several exactly 1.000) → k-th video sip IS k-th C3D take.

**Matched 22/23, 716 sip-pairs.** 18 are exact count-match positional 1:1
(P06-P20,P16,P17,P23,P24 + **P03 via "P03 (1)"** sheet, SHEET_OVERRIDE). The other
4 were **RESCUED by monotonic duration alignment** (`monotonic_align()` — both
sides strictly time-ordered so higher run# = later; DP matches duration sequences
allowing skips for dropped/extra sips, accept if dur-RMS ≤ 0.4s):
  - P21 RMS 0.02s — video=first 26, C3D 27-40 are extra mocap (clean tail). HIGH.
  - P04 RMS 0.04s — 30/30 C3D used, 3 video sips dropped (idx 6,9,13). HIGH.
  - P05 RMS 0.17s — one extra C3D take (idx 11) skipped. HIGH.
  - **P02 RMS 0.34s — LOW CONF**: 23 pairs, 11 C3D skipped scattered (warm-ups/
    retakes?), correctly skipped the valid=0 video sip. Don't trust for tight val.
Confidence tier in reason string: RMS≤0.2 "high", else "low". **Unmatched: only P01
(no mocap).** valid-flag re-test did NOT rescue (use ALL drinking rows). Empty dup
sheets P07(1)/P15(1) ignored (base sheet used).

**ALIGNMENT + VALIDATION DONE** (`qtm_align.py` → `cache/qtm_align.json`). Per rep:
resample both to 60 Hz, temporal sync by cup-speed cross-correlation, then Kabsch
(rotation+translation, NO scale — both already mm) QTM-lab→W0. Uses BOTH sides from
the `track3d_clean3d_refill` cache (351 L + 355 R; plain `track3d/` is right-only).
Pairs each side's video reps ↔ same-side C3D (dur-corr ~1.0 confirms).

**RESULT: video cup tracking matches optical GT to ~3mm where the cup is trackable.**
ROBUST (IRLS/Huber) per-rep Kabsch inlier RMS **median 2.9mm (n=654 valid reps),
p90 6.3mm**; per-participant median 2.8mm, best P09 1.5mm. (Plain least-squares read
14mm — but LS lets a localized bad segment drag the whole transform; user wanted the
fit BIASED toward the agreeing frames: "adding low errors matters, removing high
errors matters less".) The robust fit locks onto rest/transport/clear-view frames and
down-weights discrepant segments. `rep_max` survivors (P08 90mm, P13 72mm) = reps
with a real localized outlier segment the fit correctly refused to absorb — a genuine
phase-localized failure (e.g. P12-left-14: video puts cup ~140mm too HIGH in z during
the occluded drink apex; rest/transport agree → robust 7mm, LS 98mm). So inlier-RMS =
tracking fidelity where trackable; the down-weighted fraction = where it fails.
`kabsch(robust=True)` default.

**Per-rep is described by TWO numbers, not a median** (median hid localized failures —
P24 looked like a 4.7mm star but 29% of every rep fails at the apex):
  * `inlier_rms_mm` = robust-fit residual on agreeing frames = fidelity where trackable
  * `frac_fail` = fraction of trajectory beyond FAIL_MM(50) = coverage of failure
`classify_rep()` → clean / localized / broken (FIDELITY_OK_MM=15, FRAC_FAIL_OK=0.10):
  * clean: tight everywhere. localized: tight fidelity but a chunk off (apex/occlusion).
  * broken: off ~everywhere (bad track or residual GT).
**COHORT: 71% clean, 28% localized, 1% broken (6/654).** Clean median inlier ~3mm.
LOCALIZED clusters in **P24 (32/36, frac_fail .29), P16 (28/33 .18), P12 (30/32 .16),
P19 (23/34 .15), P15 (16/31)** — participant-systematic apex/occlusion z-bias (cup too
HIGH near mouth), NOT random → a real confident-wrong pattern for
[[project_failure_modes_confident_wrong]], now GT-backed. BROKEN: P13×3, P08×1, P02×2.
NOTE frac at >20mm is partly transport-phase sync jitter, not error — use >50mm.

**Stack ablation vs GT (same per-rep transform, n=654, validates the FULL pipeline is
applied: ≥3-cam gated consensus `ka.gated_consensus` → consensus-anchored KF →RTS):**
| stage | inlier_med | p90 | frac_fail | clean% |
| raw DLT triangulation (no gate/filter) | 3.4mm | 26.6 | 0.17 | 33% |
| gated+KF (causal/live) | 3.2mm | 7.0 | 0.04 | 63% |
| gated+KF+RTS (smoothed) | 2.9mm | 6.3 | 0.00 | 71% |
KEY: median fidelity ~3mm at ALL stages — when all cams see the cup, even raw DLT is
3mm; the gate+filter don't improve the typical frame, they kill the BAD-FRAME TAIL
(p90 27→7mm, clean 33→63→71%). Gate = the big win; RTS = modest polish. The aligner
scores the RTS field by default (`video_track`); kf/raw computed via re-triangulation
from `student_dets_clean3d_refill`. Localized/broken failures survive all 3 stages
(upstream of the gate — several cams agree on a wrong apex point).

**QTM RE-CLEAN ROUND 1 (user fixed takes on Windows → `Qualisys/Data/_to_inspect/`,
4 subfolders).** Originals backed up to `cache/qtm_c3d_orig_backup/` before overwrite.
17 takes were genuinely edited + copied in (verified byte-diff, all now pass
gt_quality): **12 P14 dead-takes RECOVERED** (were lift=0 = unlabeled cup, now real
drinks → P14 bad_gt 12→0, 23→35 reps), P08_0027 + P13_0029/0030/0041 (broken→clean),
DTL0007. **Result: broken 6→2, valid reps 654→666, clean 72%.** STILL PENDING: the 13
marker-swap takes (cat 01) came back UNEDITED/identical (P060026 byte-identical —
relabel not done; still rejected, not hurting validation); P02_0014/0015 unedited (the
last 2 broken); P02_0028/P19_0046 still flat (maybe truly aborted). Inspect any take in
3D without QTM: `qtm_view3d.py <stem>` (rerun viewer; red quad = defect frame).
Windows NTFS = `nvme0n1p3` (drives RE-ENUMERATE across reboots: was nvme1n1p3 — re-check
`lsblk -f` each time); mount RO `sudo mount -o ro /dev/nvme0n1p3 /mnt/win3`. Don't write
NTFS (hibernation/Fast-Startup corruption risk); Windows reads ext4 fine the other way.

**WHERE localized failures fall (use OMC/GT segmentation, NOT video-track seg —
video-seg is circular: when tracking fails the video phase boundary also shifts, which
under-counted the apex 33% vs true 55%).** With ground-truth cup phases: drinking/dwell
**55%**, forward_transport 26%, back_transport 19%, rest 0%. So failures ARE
apex-concentrated (worst at mouth/dwell) bleeding into adjacent transport; tracker is
clean when cup on table, degrades when lifted/occluded. Confirms apex-occlusion
hypothesis. (rel-pos median 0.43 = mid-rep.)

**Next:** finish marker-swap relabels (13 cat-01 takes still defective); investigate the
P24/P16/P19 apex-occlusion failure (which camera goes blind at the mouth?) — check
per-phase camera coverage for localized participants; fuse into
[[project_cup_pose_fusion]] for phase-timing validation.

**Final methodology (settled after 3 user catches, each a real failure mode):**
1. Pair rep↔sip (positional/duration, r≈1.0).
2. **Reject bad GT from the MOCAP ALONE, before any sync** (`CupTrial.gt_quality()`):
   non-physical step >50mm/frame (QTM teleport/hard gap-fill step — real drink
   <~10mm/fr) OR dead take (lift<120mm, cup never moves). 25 reps rejected.
   `CupTrial.centroid(despike=True)` also NaN-excises short return-excursions (QTM
   latches a reflection then jumps back; 5/772 trials e.g. P10_0029 43-frame block).
3. Temporal sync by cup-speed xcorr (peak corr = confidence).
4. **Per-rep Kabsch** (rot+trans, no scale, mm). Per-rep is CORRECT because we
   validate trajectory SHAPE not absolute placement (arbitrary per rep) — each rep's
   own transform is factored out, residual = pure shape error. Do NOT pool into one
   transform (reads 45-70mm from per-rep sync offsets + session drift).

**Key proof the gate is right: after up-front GT rejection, `bad_sync=0` everywhere.**
Every low-sync-correlation rep was bad GROUND TRUTH (16/34 had >50mm mocap steps, the
rest dead/borderline), never bad tracking. So high residual = bad GT, not a tracking
failure; the per-rep residual on CLEAN reps is the true accuracy + a confident-wrong
detector (10-cam, 3px reproj yet >100mm off) for [[project_failure_modes_confident_wrong]].
The earlier "left-side worse" conclusion was WRONG — it was dead/defective takes
clustering in some left blocks (P14 rejected 12). Constants: `MIN_GT_LIFT_MM=120`,
`MIN_SYNC_CORR=0.7` (clean takes sync ~0.99; relaxed since defects pre-filtered).

Compare renderer: `render_qtm_compare.py PID side rep` overlays video track (green)
vs GT mocap reprojected (cyan) + live err on the real camera frame →
`cache/qtm_compare_videos/`; picks the camera with max green-cyan separation (most
telling). P21 unalignable (0 video tracks triangulated).

**Next:** fuse GT into [[project_cup_pose_fusion]] / [[project_cup_only_segmentation]]
for phase-timing validation against the true cup trajectory.
