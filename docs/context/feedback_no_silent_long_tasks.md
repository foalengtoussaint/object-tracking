---
name: feedback_no_silent_long_tasks
description: Never run a long task in the foreground with no progress visibility; give a tailable log or live updates
metadata: 
  node_type: memory
  type: feedback
  originSessionId: bdb7162a-6c96-4391-8f59-078e4c60445f
---

Never dispatch a task that may run more than ~20-30s as a silent foreground Bash call. The user got zero feedback for ~2 minutes during an agreement-metric comparison and couldn't tell if it was running, stuck, or near done — that is not acceptable.

**Why:** the user is remote and watches their own terminal; an opaque long-running call is anxiety-inducing and unverifiable.

**How to apply:** for anything potentially slow (multi-clip inference, training, calibration, multi-model comparisons, big loops):
- Run it in the **background** writing to a log file (`... > /tmp/<name>.log 2>&1 &`), and immediately give the user the exact `tail -f /tmp/<name>.log` command to watch it.
- Make the script print **incremental progress** (e.g. per-model / per-clip / per-frame-block lines that flush), not just a final result — so the tail is informative.
- If foreground is unavoidable, keep it short, or pair it with [[feedback_use_monitor]] so errors/progress stream back.
Relates to [[feedback_progress_feedback]] (announce ETA before dispatch).
