# Run: <short-name>

**Date:** YYYY-MM-DD
**Machine:** <hostname> (GPU: <model>, OS: <distro>)
**Git SHA:** `<output of: git rev-parse --short HEAD>`
**Weights stored at:** <path on external drive / NAS / Drive folder — NOT in this repo>

## Setup

- **Cameras:** how many, model, mounting/positions
- **Room / lighting:** describe in one sentence
- **Cup(s):** color, size, material, any markings
- **Distractors present:** other cups, mugs, glasses in scene?

## Dataset

- **Source clips:** `clips/<glob>` — record_clips.py session(s) from <date>
- **Total frames after pseudo-label:** N
- **Pseudo-label base model:** `yolo11n-seg.pt` (or fine-tune from prior run `<run-name>`)
- **Confidence threshold used:** 0.X
- **Manual review:** none / spot-checked / fully reviewed
- **Train/val split:** auto / manual

## Training

```bash
python finetune.py \
    --weights yolo11n-seg.pt \
    --data dataset/data.yaml \
    --epochs 50 \
    --imgsz 640 \
    --batch 16 \
    --name <run-name>
```

Anything non-default about hyperparams or dataset prep — note here.

## Results

- **Best epoch:** N
- **mAP50 (val):** 0.X
- **mAP50-95 (val):** 0.X
- **Output:** `runs/segment/<run-name>/weights/best.pt`
- **Inference command:**
  ```bash
  python live_cup_detect.py --weights <path-to-best.pt>
  ```

## Notes

- What worked, what didn't, regressions vs. previous runs.
- Conditions where this model is expected to fail (lighting, cup type, etc.)
- Whether this run is "current best" for this physical setup.
