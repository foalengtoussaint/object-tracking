---
name: feedback-progress-feedback
description: "Always announce what you're about to run, expected duration, and give a heartbeat for anything >15s"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: cc721c98-1dfd-4c4a-ac34-07f4d2873c00
---

When running a command that takes time:
1. Say *before* launching: "running X now, ~Y seconds expected"
2. **For anything >15s: always background it AND give the tail path** so the user can monitor in their own terminal. Don't foreground long commands.
3. For shorter commands (<15s), foreground is fine — just say what's running first.

**Why:** Silent execution leaves the user wondering if anything is running or if it's a bug. They can't see my tool calls — only my text. They've called this out repeatedly: "each time you run something and it takes time, you have to give me feedback" and "always give me the tail when running something that takes more than 15 seconds." Especially relevant for MegaPose runs (3-5min cold, 4s warm), conda env creates, model downloads, training runs, COLMAP MVS, etc.

**How to apply:**
- Before launch: one short sentence ("Running MegaPose viz now, ~30s.")
- Always background long commands + emit `tail -f /tmp/.../task.output` immediately
- Never silently dispatch a long command — speak first
- For short commands (<15s), foreground is OK
