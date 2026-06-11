---
name: sotformer-deferred
description: "SOTFormer (CVPR 2026, zhongpingDong12) was considered for SOT but skipped — no public weights as of 2026-05-28"
metadata: 
  node_type: memory
  type: project
  originSessionId: cc721c98-1dfd-4c4a-ac34-07f4d2873c00
---

Considered SOTFormer (CVPR 2026, paper arxiv 2511.11824, repo github.com/zhongpingDong12/SOTFormer) for per-cam SOT in this project on 2026-05-28. Skipped.

**Why:** the `Ckpt/checkpoint_*.pth` files in the repo are 134-byte git-LFS pointer stubs — no real weights are published. README says "Available after acceptance: coming soon." Using SOTFormer would require training from scratch on LaSOT (multi-hour GPU job, uncertain convergence on our box). The user pivoted to [[position-conditioned-yolo-idea]] instead.

**How to apply:** if the user circles back to SOTFormer, first check whether weights have been released (`gh api repos/zhongpingDong12/SOTFormer/contents/Ckpt` — real weights would be tens of MB, not 134 B). If still missing, propose SAM2 as the practical alternative for a transformer-based tracker we can actually run today.
