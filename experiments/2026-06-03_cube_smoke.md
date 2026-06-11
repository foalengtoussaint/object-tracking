# cube_smoke — cube (auto-pipeline)

- date: 2026-06-03
- final weights: `/home/imove/Documents/object_tracking/data/runs/pipeline/cube_smoke/segment/student2/weights/best.pt`
- final eval: recall 0.515, P_loose 0.999

## Per-camera eval

| cam | recall | P_loose |
|---|---|---|
| cam10 | 0.063 | 1.0 |
| cam3 | 1.0 | 0.998 |
| cam6 | 0.0 | 0 |
| cam7 | 0.997 | 1.0 |

## Decision log

- **2_label_qa** [PASS]: auto-proceed — 30 labels, all valid.
- **3a_review_round1** [FLAG_FORK]: labels look good — train — Review the pseudo-labels before training. The masks should track the object across the clip.
- **3_finetune1** [PASS]: auto-proceed — training converged.
- **4_coverage** [FLAG_FORK]: Proceed anyway — 2 cameras got zero labels: ['cam10', 'cam6'].
- **5a_review_dense** [FLAG_FORK]: labels look good — train — Review the pseudo-labels before training. The masks should track the object across the clip.
- **5_finetune2** [PASS]: auto-proceed — training converged.
- **6_eval** [PASS]: auto-proceed — overall recall 0.515, P_loose 0.999; low-precision cams: none.