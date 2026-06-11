---
name: position-conditioned-yolo-idea
description: Future experiment — extend YOLO so it also takes past object positions as input
metadata: 
  node_type: memory
  type: project
  originSessionId: cc721c98-1dfd-4c4a-ac34-07f4d2873c00
---

User's parked idea (raised 2026-05-28 while we were investigating SOTFormer): take the existing cup YOLO and extend it so detection is conditioned on prior position estimates. The KF already predicts where the cup should be next frame, so the conditioning signal is essentially free.

**Why:** YOLO currently treats every frame independently. When occlusion / motion blur / unusual angle cause a miss, the network has no notion that "the cup was here 30 ms ago." Adding that as an input could boost recall without changing the backbone much. The 3D KF already provides per-cam projected position predictions, so the data plumbing exists.

**How to apply:** when the user returns to this, frame three implementation tiers:
1. Cheapest — add 4th input channel to YOLO: Gaussian heatmap of KF-predicted bbox center. Finetune from current `cup_5cam_demo_gen2` weights. Smallest change. Must include negative training samples (cup teleported / hidden / prior is wrong) or the model learns to just trust the prior.
2. Middle — second detection head conditioned on prior-position embedding, keep the original head intact for ablation.
3. Ambitious — query-based detector seeded with past tracks (TrackFormer / MOTRv2 style).

Related decisions: user looked at [[sotformer-deferred]] (no public weights) before parking this. The mask-based path forward at the time was SAM2.
