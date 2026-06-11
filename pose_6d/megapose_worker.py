"""MegaPose ZMQ worker.

Loads MegaPose RGB model + cup mesh ONCE at startup, serves pose requests
over a ZMQ REP socket. Each call ~0.5-1s once model is warm.

Wire format (pickle):
  request:  {"rgb": HxWx3 uint8 np.ndarray, "K": (3,3) np.ndarray, "bbox": [x1,y1,x2,y2]}
  reply:    {"quat_xyzw": [qx,qy,qz,qw], "t_m": [tx,ty,tz] (meters), "inf_s": float}
  on error: {"error": str}

Run inside the `megapose` conda env:
    conda activate megapose
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python pose_6d/megapose_worker.py
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import torch
import zmq

from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import ObjectData
from megapose.inference.types import ObservationTensor
from megapose.inference.utils import make_detections_from_object_data
from megapose.lib3d.transform import Transform
from megapose.utils.load_model import NAMED_MODELS, load_named_model

EXAMPLE_DIR = Path(__file__).parent / "megapose6d" / "local_data" / "examples" / "my-cup"
MESHES_DIR = EXAMPLE_DIR / "meshes"
MODEL_NAME = "megapose-1.0-RGB-multi-hypothesis"
LABEL = "my-cup"
BIND_URI = "tcp://127.0.0.1:5560"


def make_object_dataset(meshes_dir: Path) -> RigidObjectDataset:
    rigid_objects = []
    for obj_dir in meshes_dir.iterdir():
        mesh = next((p for p in obj_dir.glob("*") if p.suffix in {".obj", ".ply"}), None)
        if mesh is None:
            continue
        rigid_objects.append(RigidObject(label=obj_dir.name, mesh_path=mesh, mesh_units="mm"))
    return RigidObjectDataset(rigid_objects)


def main() -> None:
    print(f"megapose-worker: loading model {MODEL_NAME}...", flush=True)
    t0 = time.time()
    model_info = NAMED_MODELS[MODEL_NAME]
    object_dataset = make_object_dataset(MESHES_DIR)
    pose_estimator = load_named_model(MODEL_NAME, object_dataset).cuda()
    print(f"megapose-worker: loaded in {time.time()-t0:.1f}s, "
          f"label={LABEL}, mesh={MESHES_DIR/LABEL}", flush=True)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(BIND_URI)
    print(f"megapose-worker: listening on {BIND_URI}", flush=True)

    while True:
        try:
            req = pickle.loads(sock.recv())
        except KeyboardInterrupt:
            print("\nmegapose-worker: shutdown")
            break

        t0 = time.time()
        try:
            rgb = req["rgb"]
            K = np.asarray(req["K"], float)
            bbox = req["bbox"]

            observation = ObservationTensor.from_numpy(rgb, None, K).cuda()
            obj_data = [ObjectData(label=LABEL,
                                   bbox_modal=np.asarray(bbox, dtype=np.float32))]
            detections = make_detections_from_object_data(obj_data).cuda()

            with torch.no_grad():
                output, _ = pose_estimator.run_inference_pipeline(
                    observation, detections=detections,
                    **model_info["inference_parameters"])

            pose_4x4 = output.poses[0].cpu().numpy()
            T = Transform(pose_4x4)
            reply = {
                "quat_xyzw": T.quaternion.coeffs().tolist(),  # x,y,z,w
                "t_m": T.translation.tolist(),                # meters
                "inf_s": time.time() - t0,
            }
        except Exception as e:
            reply = {"error": f"{type(e).__name__}: {e}"}

        sock.send(pickle.dumps(reply))
        if "error" in reply:
            print(f"  ERR {reply['error']}", flush=True)
        else:
            print(f"  {reply['inf_s']*1000:.0f}ms  t={reply['t_m']}", flush=True)


if __name__ == "__main__":
    main()
