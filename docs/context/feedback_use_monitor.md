---
name: feedback-use-monitor
description: "Use the Monitor tool to stream stdout from long bg tasks so I see errors live, not after task completes"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: cc721c98-1dfd-4c4a-ac34-07f4d2873c00
---

When running a long background task that may print errors during the run, **also spawn a Monitor** that tails the task's output file and emits each interesting line as a notification.

**Why:** Otherwise I'm blind during the run — task notifications only fire on completion. The user shouldn't have to paste errors into chat for me to see them.

**How to apply:**
```
Bash(run_in_background: true) → task.output  # the actual work
Monitor("tail -f task.output | grep -E --line-buffered 'Error|Traceback|FAIL|OOM|done|wrote'") 
```
- Use `grep --line-buffered` (pipe buffering otherwise delays events by minutes)
- Cover ALL terminal states in the grep (success AND failure signatures) — silence is not success
- Don't pipe raw logs to Monitor (too many events → auto-stopped). Filter to lines you'd act on.
- For "tell me when X is ready," prefer `Bash(run_in_background)` with an `until` loop that exits on the condition — one notification, no timeout drift.

**Recurring example for this project (MegaPose / YOLO long runs):**
```
Monitor("tail -f /tmp/.../task.output | grep -E --line-buffered \
  'Traceback|Error|OutOfMem|OOM|done|inf [0-9]+ms|wrote|loaded in'",
  description="megapose offline run")
```
