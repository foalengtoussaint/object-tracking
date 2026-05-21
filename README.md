# Multi-View Object Tracking

3D visual hull reconstruction of objects from three calibrated cameras. A YOLO segmentation model carves a voxel grid from three synchronized views to produce a point cloud and mesh.

## System overview

```
[Camera server (Unitree/teleimager)]
        │  ZMQ streams (ports 55555-55557)
        ▼
[This machine]
  multi_view_client.py   – live preview
  capture_triplet.py     – save synchronized frames
  record_calibration.py  – record videos for calibration
  run_calibration.py     – run Charuco bundle calibration
  visual_hull.py         – 3D reconstruction from frames
  find_cam_mapping.py    – auto-detect stream↔calibration mapping
```

## Setup

### 1. Clone (with submodule)

```bash
git clone --recurse-submodules https://github.com/foalengtoussaint/object-tracking.git
cd object-tracking
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init
```

### 2. Install Python dependencies

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

`aniposelib` may require additional system packages on Linux:

```bash
sudo apt install libgl1 libglib2.0-0
```

### 3. Set up the camera server

The camera server runs on the robot (or any machine with three USB cameras) and streams JPEG frames over ZMQ. Follow the setup instructions in [teleimager/README.md](teleimager/README.md):

```bash
cd teleimager
pip install -e .
# Edit cam_config_server.yaml with your camera indices, then:
python -m teleimager.server
```

The server publishes on ports **55555** (left), **55556** (center), **55557** (right) by default. Update the `PORTS` dict in the scripts if you use different ports.

### 4. YOLO weights

The YOLO weights (`yolo11n-seg.pt`) are downloaded automatically by `ultralytics` the first time you run a script that needs them. No manual step required.

## Workflow

### View live streams

```bash
python multi_view_client.py
# Press q to quit
```

### Capture a synchronized frame triplet

```bash
python capture_triplet.py                # saves to captures/
python capture_triplet.py --yolo         # start with YOLO cup overlay on
# Controls: SPACE/c = capture,  y = toggle YOLO overlay,  q = quit
```

### Calibrate the cameras

**Step 1 – Record Charuco footage** (move the board through the whole workspace):

```bash
python record_calibration.py
# Controls: SPACE = start/stop recording,  q = quit
# Aim for ~30 s with the board visible in at least one camera at all times.
# Output: calib_recordings/<timestamp>/cam-1.mp4, cam-2.mp4, cam-3.mp4
```

**Step 2 – Run bundle calibration**:

```bash
python run_calibration.py calib_recordings/<timestamp>/
# Writes calibration.toml next to the videos (takes a few minutes)
```

**Step 3 – Find the correct stream↔calibration mapping** (needed once per hardware setup):

```bash
# First capture a triplet with a cup clearly visible in all three views
python capture_triplet.py captures/

python find_cam_mapping.py captures/ --calib calib_recordings/<timestamp>/calibration.toml
# Prints the best CAM_MAPPING — copy it into visual_hull.py
```

### Reconstruct a 3D model

```bash
python visual_hull.py captures/ --calib path/to/calibration.toml --out cup_model/
# Outputs: cup_model/cup_points.ply, cup_model/cup_mesh.ply, cup_model/masks/
```

Open the `.ply` files in [MeshLab](https://www.meshlab.net/) or any 3D viewer.

## Configuration

| File | Key constants |
|---|---|
| `visual_hull.py` | `CALIB_TOML`, `CAM_MAPPING`, `VOLUME_SIDE_MM`, `VOXEL_SIZE_MM`, `MIN_VIEWS_INSIDE` |
| `capture_triplet.py` / `record_calibration.py` | `PORTS`, `IMAGE_SIZE` |
| `run_calibration.py` | `BOARD` (Charuco board dimensions) |

Translation units throughout are **millimeters**, matching the iMOVE calibration convention.
