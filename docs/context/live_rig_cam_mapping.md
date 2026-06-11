---
name: live_rig_cam_mapping
description: Live 4-BRIO rig — USB re-enumeration shuffles /dev/video order; verify cam->profile mapping before trusting 3D
metadata: 
  node_type: memory
  type: project
  originSessionId: 842cd438-acac-4b32-a71a-c4ab5b6ad54e
---

The local live rig is **4 Logitech BRIO** webcams (cam_4 absent). Each BRIO exposes ~4 `/dev/videoN` nodes, so the 4 cams sit on video **0, 4, 8, 12**. `recording/teleimager/cam_config_server.yaml` maps: cam_1→video0, cam_2→video8, cam_3→video4, cam_5→video12 (ports 55555/56/57/59). cam_5's device is **video12, not video16** (video16 doesn't exist).

**Gotcha (2026-06-02):** USB re-enumeration shuffles which physical BRIO lands on which `/dev/videoN`, so calibration profiles (matched to streams by NAME in `live_track.py`/`load_calibration`) can end up on the wrong camera. Symptom: live 3D track flip-flops between two inconsistent positions; `live_track.py --check` shows reprojection error in the hundreds–thousands of px. It was **cam_2↔cam_3 swapped**; fixed by swapping their `video_id` in the config (cam_2→8, cam_3→4). After fix, `--check` is ~1–6px.

**Before trusting live 3D, always run the calibration check** with the Charuco/ArUco board flat and visible in 2+ cams: `python live_track.py --check --cal data/calibration_local.toml` (want <5px). To find a relabel fix fast, `/tmp/perm_check.py` brute-forces profile→stream assignment and reports the best. Calibration to use for the live rig is `data/calibration_local.toml` (names cam_1/2/3/5 with underscores, matching stream names) — NOT the 5-cam `data/calibration.toml`.

Live cup demo: `live_track.py --weights data/runs/segment/cup_5cam_demo_gen2/weights/best.pt --cal data/calibration_local.toml --classes 0` — note the cup model is **single-class (class 0 'my_cup')**, so `--classes cup-like` (COCO 39/40/41/45/75) detects NOTHING; must use `--classes 0`. Related: [[calibration_source]], [[cube_sam_distill]].