---
name: feedback_verbose_long_scripts
description: "Long-running eval/inference scripts must print incremental progress, not only a result block at the end — empty logs look like a hang."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 7857798a-7e7a-40b9-bc77-a4a5a69893a9
---

When writing a script/heredoc that takes more than a few seconds (per-checkpoint
eval, per-clip inference, label generation, multi-participant agreement passes),
make it VERBOSE with incremental progress — print per-clip / per-camera / per-N
lines as it goes, not just a single result block at the very end.

**Why:** a script that only prints at the end leaves the log file empty the whole
time it runs, which is indistinguishable from a hang. The user repeatedly had to
ask "still running?" because `tail -f` showed nothing. It also wastes a tailable
log (one of the standing prefs — see [[feedback_no_silent_long_tasks]]).

**How to apply:**
- Print a start line (what's running, how many items) and a `[i/N] item` line per
  unit of work, with `flush=True`.
- For ultralytics, leave per-clip prints on or add an explicit counter; don't
  suppress all output and only print the final table.
- Pair with Monitor on a grep that matches the progress lines (and error
  signatures), so progress streams live.

Relates to [[feedback_no_silent_long_tasks]], [[feedback_progress_feedback]],
[[feedback_use_monitor]].
