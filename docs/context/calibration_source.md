---
name: calibration-source
description: "Where the 5-camera BRIO Charuco calibration for object_tracking lives, format, and unit/scaling notes"
metadata: 
  node_type: memory
  type: reference
  originSessionId: cc721c98-1dfd-4c4a-ac34-07f4d2873c00
---

The 5-camera calibration matching the object_tracking ZMQ rig (cam_1..cam_5) is in the iMOVE DATA tree, not in this repo:

`/home/imove/Documents/iMOVE/DATA/sub-test/20260428-152404/trials/calibration_20260428_152404_calibration.toml`

(The `.filtered.1-2-3.toml`, `.filtered.4-5-6.toml`, etc. siblings are subsets of this same shoot — share the same reprojection error of 0.257 px. Use the unfiltered one when you need all 5 cameras together.)

Format: **aniposelib TOML** (Charuco). Each `[cam_N]` block has `matrix` (3x3 K), `distortions` (5-vec OpenCV), `rotation` (3-vec Rodrigues rvec), `translation` (3-vec tvec), `size` (calib image WxH).

Camera mapping:
- TOML `[cam_0]` → object_tracking `cam_1` (TOML `name` field is `"1"`)
- `cam_1` → `cam_2`, `cam_2` → `cam_3`, `cam_3` → `cam_4`, `cam_4` → `cam_5`
- i.e. `cam_<i>` in the TOML maps to `cam_<i+1>` in [recording/teleimager/cam_config_server.yaml](recording/teleimager/cam_config_server.yaml). Old memory mentioned cam_left/center/right naming + an uncertain mapping — that was the 3-camera era; the 5-camera TOML uses indexed `name` fields that align 1:1.

Important details (carry over from the 3-cam predecessor):
- (R, t) is **world → camera** (OpenCV convention), same form as `cv2.solvePnP` output.
- Translation units are **millimeters**. iMOVE's `realtime_pose.py` divides by 1000 for meters.
- Calibration was done at **1920x1080**; ZMQ streams are **1280x720** (16:9 → 16:9, pure 2/3 scale). Rescale intrinsics: `K' = diag(2/3, 2/3, 1) @ K`. Distortion coefficients stay the same (they're functions of normalized image coords).
- World origin is the Charuco board frame — not where the tracked object will sit. For 3D Kalman state, either recenter on the triangulated object centroid at init, or just work directly in world coords and let the KF cover the offset.

Caveats before using:
- These cams are the iMOVE BRIO rig. The object_tracking pipeline records from the same machine via the same teleimager ZMQ ports, so it's almost certainly the same physical rig — but worth a sanity check by overlaying a projected Charuco corner on a live frame before trusting 3D positions for pseudo-labels.
- Existing `data/clips/train/cam_N_*.mp4` were recorded by independent ZMQ subscribers ([recording/record_clips.py](recording/record_clips.py)) — no shared timestamping. Multi-cam fusion needs frame-synced clips, so existing ones may need to be re-recorded with a single-process synced recorder.

Related: [[kalman-3d-design]] (TBD) — the 3D Kalman work that will consume this calibration.
