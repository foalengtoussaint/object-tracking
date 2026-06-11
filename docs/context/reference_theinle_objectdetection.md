---
name: reference_theinle_objectdetection
description: theinle/ObjectDetection repo — Unity+SwissDINO one-shot onboarding; source for our swissdino_lib port
metadata: 
  node_type: memory
  type: reference
  originSessionId: 6dbe7eaf-84a9-438a-96e2-3dbb2b1ed064
---

Private GitHub `theinle/ObjectDetection` (user has access via `foalengtoussaint` gh account). Up-to-date branch is **`dev/swissdino`**, not `master`. It's a Unity (Quest/MetaXR) + Python one-shot object onboarding system — single-camera, training-free, headset-facing. NOT multi-camera/3D like ours.

Core method (in `AdditionalScripts/SwissDINO/engine.py`, `segmentation_utils.py`): DINOv2 frozen features → mean of normalized patch features inside a mask = **prototype vector**; cosine similarity map vs prototype → adaptive percentile **threshold** → connected components → pick best-matching component (PerSAM argmax fallback). Onboarding uses Track-Anything (SAM+XMem) in WSL to turn 1 click into a propagated mask.

**License: SwissDINO code is CC-BY-NC-SA 4.0 (Samsung)** — non-commercial + share-alike. We are clean-room reimplementing the (trivial) prototype/threshold/CC math from the method, NOT copying their files. DINOv2 backbone itself is Apache-2.0.

We're porting this as `swissdino_lib.py` for: faster object onboarding (replace 30-click cold start), training-free fallback detector, and a DINOv2-cosine appearance gate for the 3D label filter — see [[project_swissdino_port]].
