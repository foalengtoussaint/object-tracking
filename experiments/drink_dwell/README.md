# drink_dwell — the drink-dwell result, one legible stage at a time

Self-contained rebuild of the proxy21 drink-dwell pipeline (headline: **proxy21 ~77 ms** vs
**base17 ~118 ms** duration error against mocap ground truth). Split out of `drink_study`
because the original was a 96-line tangle no one — human or model — could debug a single stage
of. Here every stage is a small named function you can run on ONE trial and inspect.

## Run it

```bash
python experiments/drink_dwell/run.py        # -> cache/results.json  (the headline numbers)
python experiments/drink_dwell/summary.py    # -> slides/dwell_summary.png  (base17 vs proxy21: CDF + paired + table)
python experiments/drink_dwell/plot.py       # -> slides/worst_proxy21_grid.png  (worst reps)
```

**base17 vs proxy21:** identical inputs (13 kinematics + 4 occlusion); proxy21 just ADDS the
4 head-distance channels. So base17 is the video-only ceiling and the base17→proxy21 gap is
exactly what cup→head distance buys.

## The pipeline, stage by stage (each file runs standalone for a smoke test)

| file | stage | in → out | inspect |
|---|---|---|---|
| `mocap.py` | load cup + head mocap; **the signal** | C3D → `cup_to_head()` (T,) mm | `python mocap.py P02_0030` |
| `truth.py` | drink-dwell truth (van Andel) | cup→head dist → dwell span | `python truth.py P02_0030` |
| `features.py` | build proxy21 / base17 | caches → (T,21) / (T,17) | `python features.py` |
| `model.py` | TCN + scoring primitives | — | (imported) |
| `run.py` | LOPO train/score | reps → `results.json` | the entry point |
| `plot.py` | worst-reps graph | `results.json` → grid PNG | — |

**The 21 channels (proxy21):**
- 13 **kinematics** (`features.kinematics`): filtered speed/disp + RAW 3D velocity & displacement
  in the rep-local basis (direction, not just magnitude).
- 4 **occlusion** (`features.occlusion`): present, median reprojection px, #cameras, occluded flag.
- 4 **head distance** (`features.head_distance`): **tracked** cup → **mocap** head-centroid
  (dist, approach velocity, normalised, present). The head is the only mocap stand-in — a proxy
  for the future *video* head landmark; the cup is the real noisy track. This is why it's
  deployment-realistic, and why proxy21 (77) is worse than the mocap-cup oracle (34) but honest.

base17 = kinematics + occlusion (video-only ceiling). The 4 head channels are the whole lever.

## Truth (van Andel), in `truth.py` as 5 visible steps
smooth cup→head → apex (floor) → rest (walk out to 70% of max) → threshold = apex + 15%·(rest−apex)
→ longest run below. One head point, no mouth proxy → transfers to a 1-landmark biomech head model.

## Data (shared with drink_study — copied CODE, shared DATA, no duplicated caches)
Reads from `../drink_study/cache/`:
- `lopo_fused/<rep>.npz` — fused 3D cup track + rep-local basis/rest + truth (per rep)
- `track3d_clean3d_refill/<rep>.json` — per-frame detection health (occlusion channels)
- `qtm_align.json` — which mocap C3D each rep paired to + sync lag
- `qtm_c3d_cleaned/*.c3d` — the cup + head mocap

No GPU is used for detection/tracking (those caches are reused); `run.py` only trains the tiny
per-frame TCN (CPU is fine, ~3 min for 21 LOPO folds × 2 models).

## Why this replaces the drink_study version
`drink_study/analysis/learn_seg_mouth.py` + `plot_worst.py` produced the same number through a
tangle where the cup→head distance was implicit across three interleaved concerns (mocap load,
Kabsch, resample) — the reason a "spike" in that distance took a whole session to localise. Here
the distance is `mocap.CupTrial.cup_to_head()`, one function. Those two old scripts are retired to
`drink_study/archive/` so there is exactly one live copy (no drift). Verified to reproduce the
same 666-rep set and the same proxy21/base17 numbers.
