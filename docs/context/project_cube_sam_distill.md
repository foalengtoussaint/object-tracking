---
name: project_cube_sam_distill
description: SAM-seed → dense KF distillation pipeline works for non-COCO ARAT cube; generalizes to held-out participant
metadata: 
  node_type: memory
  type: project
  originSessionId: 842cd438-acac-4b32-a71a-c4ab5b6ad54e
---

Validated (2026-06-02) a pipeline for detecting non-COCO ARAT objects, starting with the wooden **cube** (GRASP_Cube task), P01 → held-out P02.

Pipeline: **30 manual SAM clicks** (browser tool `sam_label_server.py`, MobileSAM point-prompt, 3 cams × 10 frames) → train weak seed student (yolo26n-seg) → use seed student as teacher to **KF-pseudo-label all 40 P01 cube clips** (`pseudo_label.py --classes 0`, 6350 frames) → train final "dense" student.

Key results:
- Seed student (30 imgs, cams 3/7/9) already generalized to 5 of 7 unseen P01 cameras; **blind on cam6/cam10** (distant/oblique, cube tiny).
- Dense student on **held-out participant P02**, 10cm cube, all 10 cams: overall recall **0.59→0.87** vs seed. **Self-healed cam6 (0.01→0.48) and cam10 (0.00→0.66)** despite those cams contributing ZERO pseudo-labels — learned cube appearance from 8 other views.
- Teacher generalized across cube SIZES for free (trained on 10cm, labeled 7.5cm/5cm well; 2.5cm mostly missed — too small).
- Consistent with cup pipeline: near-zero false positives, KF rejects ~0; model "detects cube or nothing".

Artifacts: seed weights `data/runs/segment/cube_p01_seed/weights/best.pt`; dense weights `data/runs/segment/cube_p01_dense/weights/best.pt` (class `cube`); dense dataset `data/datasets/cube_p01_dense/`. YOLO-World was rejected as teacher (only ~4% recall, lifted-pose only) — see why manual SAM seeds chosen. Tooling: `sam_label_server.py` (port 5008). Next candidates: other non-COCO ARAT objects (stone, tube, washer) via same loop. Related: [[megapose_quat_order]] for downstream 6D pose.
