---
name: project-imove-fast-biomech-pipeline
description: "How the iMOVE \"fast biomechanical model\" is actually run — it's the ISR/DataJoint imove_fast_* pipeline in ~/Documents/iMOVE, driven by PipelineOrchestrator on a DB-ingested MultiCameraRecording, NOT a raw-file CLI."
metadata: 
  node_type: memory
  type: project
  originSessionId: 25d11f20-b722-49a2-b616-7d5262e468ad
---

The "fast biomechanical model" is NOT in the object_tracking repo. It's the iMOVE
ISR container stack at `~/Documents/iMOVE/DEV/`. Inputs are iMove-SPZ **session dirs**
(`~/Documents/iMOVE/DATA/sub-*/<timestamp>/` with `_recording_meta.json` v2 +
`calibration/<base>.json`), ingested into a **DataJoint DB**, then processed by
DataJoint `.populate()` — there is no raw-mp4 CLI.

## The chain (DataJoint schemas, all coexist with the canonical pose_pipeline)
`Recording` → `MultiCameraRecording` → `CalibratedRecording` →
`BottomUpBridgingFast` (schema `imove_fast_pose`; MeTRAbs bml_movi_87, cross-cam batched,
num_aug=2/1, stride 1/2 — "fast" = low aug) → `PersonKeypointFast` (triangulate, auto
person-pick) → `KinematicReconstructionFast` (schema `imove_fast_kinematic`; IK → biomech
skeleton/meshes, terminal). Optional parallel branches: hands (WholeBody/Hand), drink
(`imove_fast_object`: CupTrackingFast + DrinkTaskSegmentationFast + MurphyMeasuresFast).

## How it's actually driven (the recipe from "a few videos previously")
iMove-SPZ composes a pipeline-orch YAML and runs it via `PipelineOrchestrator(yaml_path).run(restriction=...)`:
- YAML generator: `iMOVE/DEV/iMove-SPZ/src/fast_pipeline_yaml.py::compose_fast_pipeline_yaml(stride,aug,hands,drink)`.
  Method PKs: (stride,aug)→BottomUpBridgingFastMethodLookup {(1,2):1,(2,2):2,(1,1):3,(2,1):4};
  KinematicReconstructionFastSettingsLookup="2"; entry `Recording filter "participant_id LIKE '%'"`.
- Runner: `imove_extensions/fast_preloader.py` — preloads MeTRAbs once (~41s), then
  `PipelineOrchestrator(yaml).run(restriction=<DJ filter>, use_subprocess=False, execution_mode="breadth_first")`.
  Restriction picks WHICH recording(s). Spawned by iMove-SPZ at sync-time (`postprocess_settings.py`).
- Executes inside `isr/prod:stable` (image built locally, 42.7GB). Aliases in
  `isr-containers/docker_aliases.sh` (`run_prod_session_pipeline` etc.) — must be `source`d with full env.

## State on THIS box (2026-07-08)
- Image `isr/prod:stable` present. GPU RTX 3060 Ti 8GB (fast path fits; canonical num_aug=10 risky).
- DataJoint DB volume exists+populated: `isr-supplementary/mySQL_mount` (4.3GB), schemas
  `imove_fast_{pose,kinematic,object}`, `imove_object_tracking`, `imove_calibration_archive`,
  `multicamera_tracking`, `pose_pipeline`, etc. DB container = compose service `db` (datajoint/mysql:8.0).
- Prior real inputs: `~/Documents/iMOVE/DATA/sub-02, sub-03, sub-test*` (have `_recording_meta.json`).
- **BLOCKER: DB cred drift** — running db rejects `.env` creds (DJ_USER=root + 29-char DJ_PASS →
  Access denied); `MYSQL_ROOT_PASS` empty in `.env`. Volume was initialized with a different root
  pw than current `.env`. Fix before any query: recover pw / reset root on the existing volume.

## ACTUAL runnable path for drink_study clips (2026-07-08 — SOLVED, no DataJoint ingest)
The fast biomech model was run on drink_study clips via a DB-BYPASS script, NOT the orch/populate path:
- Driver: `experiments/drink_study/run_pose_batch.sh` (untracked) → stages each trial's 10 cam clips
  from `/home/imove/Documents/clips/<P>/<P>_drinking_<side>_<ts>.<cam>.mp4` + `data/calib/<P>/calibration.toml`
  into `iMOVE/DEV/isr-supplementary/DATA/ot_pose_scratch/{clips,calib}` → `docker exec isr-containers-dev-1
  python run_pose.py ...` → copies `biomech_<trial>.npz` into `experiments/drink_study/cache/`.
- `run_pose.py` (in the DATA scratch, container path `/home/vscode/workspace/DATA/ot_pose_scratch/`):
  5 stages = MeTRAbs bml_movi_87 (metrabs_multicam_batched, num_aug=2) → triangulate to W0 →
  IK → phases → Murphy. Connects to `root@db:3306` at startup (needs the compose `db` up).
- Output cache: `cache/biomech_*.npz` (keypoints3d (T,87,4)[x,y,z,conf] + side). Consumed by fuse_phases.py.
- COST: ~6 min/rep steady-state on RTX 3060 Ti (MeTRAbs ~558s dominates; IK 41s; rest seconds).
  Cold start +~1min model load. 721 total drinking reps; ~70h for all → OPT levers (parked for supervisor):
  (A) detect-once+crop [biggest, MeTRAbs re-runs full-1080p detector every frame], (B) downscale input,
  (C) num_aug=1 + frame_stride=2 (both already supported args in bridging_fast.metrabs_multicam_batched).
- BUGS FIXED this session: (1) DB root pw — volume pw is `pose` (compose strips the inline `#comment`
  correctly; my bash `cut` didn't → false "access denied"). Reset root to `pose` via `mysqld --init-file`.
  (2) clip-glob mismatch: run_pose.py stripped the leading `<P>_` (old lists used double-P `P07_P07_...`);
  patched run_pose.py + run_pose_batch.sh to match full `<P>_drinking_...` names. Idempotent: skips
  existing npz → crash-safe/resumable.
- Calib present for 22/23 (all but P21). 2026-07-08 run: breadth-first 6/participant subset =132 reps
  (`/tmp/subset_reps.txt`), ~13h, finishes ~6am; chain script waits for P01 test then batches the rest.

## (superseded) DataJoint path — heavier, not used

drink_study clips are from the SEPARATE 4/5/10-cam BRIO rig, not iMove-SPZ sessions → likely need an
ingest shim (Session/Recording/SingleCameraVideo + matching Calibration from the BRIO TOML) unless
they're already in the DB. Can't confirm until the cred blocker is cleared. See [[calibration_source]]
[[live_rig_cam_mapping]].
