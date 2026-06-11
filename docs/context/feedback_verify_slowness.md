---
name: feedback-verify-slowness
description: "When user says something is too slow, verify with concrete timestamps — don't handwave"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: cc721c98-1dfd-4c4a-ac34-07f4d2873c00
---

When the user reports "too slow" / "stuck" / "still nothing":
1. **Immediately** check concrete signals — don't reassure based on expectations alone
2. Tools:
   - `ps -p PID -o etime` — how long has the process been alive
   - `stat -c %Y FILE` or `ls -lh FILE` — file mtime / size growth
   - `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv` — is the GPU actually busy
   - `cat /proc/PID/wchan` — what kernel call is the process blocked on
   - `date +%s` — current epoch for diff math
3. Compare to expected duration. State the math: "etime 4:30, expected ~10s → hung."
4. If genuinely stuck, kill and debug. Don't suggest "wait it through" without evidence the process is actually making progress.

**Why:** I have NO inherent time sense between tool calls. Saying "JIT compile takes 30-60s, wait it through" without checking GPU activity is hand-waving and wastes the user's time. They've called this out: "you can't perceive time" / "why cant you verify that". The user's perception is the ground truth signal — my job is to corroborate or refute with diagnostics.

**How to apply:** Treat every "this is slow" report as a hypothesis to test in the next tool call. Lead with diagnostics, not assumptions.
