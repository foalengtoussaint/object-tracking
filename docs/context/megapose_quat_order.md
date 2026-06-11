---
name: megapose-quat-order
description: "MegaPose serializes quaternions as xyzw (Eigen convention), not wxyz"
metadata: 
  node_type: memory
  type: project
  originSessionId: cc721c98-1dfd-4c4a-ac34-07f4d2873c00
---

MegaPose's `object_data.json` output contains poses as `TWO: [[qx, qy, qz, qw], [tx, ty, tz]]` — **xyzw order** (Eigen / pinocchio's `Quaternion.coeffs()` returns x first, w last). Don't assume scalar-first wxyz.

**Why:** Burned ~30 min debugging "MegaPose rotation off by 50°" when the predictions were actually fine — my conversion treated the quat as wxyz and produced a tilted rendering. Verified in [pose_6d/megapose6d/src/megapose/lib3d/transform.py:78-83](pose_6d/megapose6d/src/megapose/lib3d/transform.py) and [scene_dataset.py:67-68](pose_6d/megapose6d/src/megapose/datasets/scene_dataset.py) — `transform_to_list` calls `T.quaternion.coeffs().tolist()` which is xyzw.

**How to apply:** When loading TWO from MegaPose JSON, unpack as `x, y, z, w = quat`, not `w, x, y, z`. Translation field is meters (multiply by 1000 if your mesh is in mm). Pose is object→camera in OpenCV convention.
