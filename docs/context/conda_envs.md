---
name: conda-envs
description: Which conda env to use for which part of the object-tracking pipeline
metadata: 
  node_type: memory
  type: reference
  originSessionId: 62571851-fddb-428a-9781-9cae276ead5b
---

Two conda envs are involved in this project. They are NOT interchangeable:

- **`teleimager`** (Python 3.10) — runs the Tele Imager camera server (`teleimager-server`) that publishes the 3 BRIO streams over ZMQ on ports 55555/55556/55557. Has only pyzmq, opencv, numpy. No torch/open3d.
- **`object_tracking`** (Python 3.10) — has everything heavy: open3d 0.19, ultralytics 8.4, torch 2.7+cu118 (CUDA available), scikit-image, filterpy, deep-sort-realtime. Plus pyzmq + toml (added 2026-05-21). **This is the env for any ZMQ client / YOLO / visual-hull / triangulation script.**

So the runtime split is:
```
# terminal 1 — keep the camera server alive
conda activate teleimager
teleimager-server

# terminal 2 — everything that consumes frames
conda activate object_tracking
python capture_triplet.py captures/
python visual_hull.py captures/
```

Other envs (`objects`, `arat`, `idrink`) exist but are not for this project — don't reuse them by mistake.
