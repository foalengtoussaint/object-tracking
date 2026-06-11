---
name: project_kf_accuracy_budget
description: "drink_study: 3D KF tracks the cup (<cup-radius 35mm err) up to ~±20px detection jitter and needs >=3 well-spread cams; sigma>=40px or only 2 cams -> catastrophic (>1m, locks onto glass). Holds across 4 participants w/ pscale_4 STUDENT (not just teacher/P01)"
metadata: 
  node_type: memory
  type: project
  originSessionId: bdb7162a-6c96-4391-8f59-078e4c60445f
---

Offline sensitivity sweep (`experiments/drink_study/kf_accuracy.py`), **pscale_4 STUDENT** (single-class cup, conf 0.25 — the model that runs at inference, not the teacher) on **4 calibrated participants P01/P06/P19/P23, 1 trial each**. Per rep: reference = RTS-smoothed >=3-cam gated consensus of the student's own dets (100% cov on all 4); track error = 3D dist of a fresh runtime EKF estimate to that reference. Student dets cached in `cache/student_dets/`; aggregate in `cache/kf_accuracy_multi.json`.

**(a) per-detection precision budget** (all cams, add N(0,sigma) px jitter; median over participants):

| sigma_px | median_mm (agg) | p25–p75 |
|---|---|---|
| 0  | 3.0  | 2.8–3.9 |
| 10 | 6.8  | 6.5–7.7 |
| 20 | 16.8 | 15.9–89.7 |
| 40 | 582  | 399–703 |
| 80 | 966  | 877–1055 |

Sharp cliff: usable (median < cup radius 35mm) up to **~±20px (1σ)** on all 4 people; at sigma=40px the EKF Mahalanobis gate (GATE_2D=13.82, meas_noise_px=8) admits garbage → >0.5m, and on P01 relocks onto the cam_10 glass.

**Per-frame TAIL (median over reps of each rep's within-rep percentile) is the sharp version — the tail fails before the median:** at sigma=20px median=16.8mm but **worst-10%=30.6mm, worst-1%=44.8mm** → the worst-1% frame already busts the 35mm cup radius. So **safe ceiling ≈15px** if you care about the worst frame, not 20. By 40px the whole tail blows out (worst-1%≈1660mm). Even at sigma=0 the worst-1% is ~20mm = constant-velocity overshoot at the sharp drink reach/retract (harmless, RTS flattens it). **Judge a tracker on its worst-10%/worst-1% frame, not its median.**

**Nuance:** the ±20px budget is for clean *long* tracks — **P23** (374 frames, shortest/sparsest) already breaks at 20px (median 306mm, worst-1% ~1019mm) because less temporal redundancy = less noise to average away. Tighten the budget for short/low-coverage clips.

**(b) geometric redundancy** (no extra noise, ref-seeded to isolate *tracking* from *seeding*, feed only the n best-covered cams; median over participants):

| n_cams | median_mm (agg) | p25–p75 |
|---|---|---|
| 2 | 12.9 | 10.8–496 |
| 3 | 7.9  | 6.4–9.2 |
| 4 | 6.9  | 6.1–7.3 |
| 10 | 3.0 | 2.8–3.9 |

**Does the noise cliff move with #cameras?** Combined (N, sigma) grid → sigma* = noise where median crosses cup radius: 2 cams 10.6px, 3 cams 20.1px, 5/7/10 cams ~20.5px. **Redundancy buys noise tolerance only up to ~3 cams, then SATURATES** — the cliff is set by the EKF Mahalanobis gate (rejects out-of-gate detections), NOT by averaging, so extra cams can't average a too-noisy detection back in; they buy precision (σ=0 median 8→3mm from 3→10 cams) + dropout-robustness, not a higher noise ceiling. **Non-monotonic dip at N=4 (sigma*=11.7px)**: not noise — the top-4-by-coverage set leans on a marginal/poorly-spread view; viewpoint IDENTITY dominates over count (the [[project_e6_camera_transfer]] lesson). So sigma* isn't monotone in N.

**>=3 well-spread cams** is uniformly safe (matches the >=3-consensus gate; worst-1%=19mm). **2 cams the median LIES** — agg median 12.9mm looks OK but **worst-10%=50.4mm, worst-1%=69.5mm (~2× cup radius)**: degenerate depth ray drifts on hard frames. *How badly* it fails is participant-dependent — catastrophic on P01 (worst-1%=9494mm), mild on P19/P23 (~18mm) — so 2-cam median is misleading; the per-frame p90/p99 is what exposes the degeneracy.

**Online EKF vs offline +RTS (same degraded input, both scored):** RTS cuts the worst-1% ~30% across the usable range — at 20px it pulls worst-1% **44.8 → 30.5mm, back under cup radius** (restores the 20px ceiling the causal filter loses). Median barely moves (3.0→2.6) — RTS's value is entirely in the **bad-frame tail** (velocity overshoot). But RTS is NOT magic: can't fix 2-cam degeneracy (worst-1% 69.5→52.6, still >cup radius — geometry can't be smoothed) and can't rescue the 40px cliff (1660→1342 — gate already rejecting good dets). So: **live tracker → ±15px budget; offline label cleaner → full ±20px (RTS recovers tail); both still need >=3 cams.**

**How to apply:** two independent accuracy requirements — (1) detector localization within ~±15px (online) / ~±20px (offline w/ RTS) 1σ, judged on the **worst-1% frame not the median**, (2) >=3 simultaneous cameras must agree. `kf_accuracy.py` scores both modes (rts flag); `run_kf(rts=True)` does forward EKF + `_rts_backward`. Relates to [[project_label_kf_is_2d_only]], [[project_3dclean_vs_dropcam]], [[megapose_quat_order]].
