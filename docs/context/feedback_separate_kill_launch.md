---
name: separate-kill-launch
description: "Don't chain `kill ... ; ... ; python launch ...` in one Bash call — split into separate commands"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: cc721c98-1dfd-4c4a-ac34-07f4d2873c00
---

When restarting a long-running process (the live tracker, cam_server, etc.) don't put the kill, any wait, and the new launch in the same Bash command. Split them into separate Bash tool calls: first kill, then verify it's gone, then launch.

**Why:** the user has explicitly asked for this. Multiple practical failures came from chaining:
- `pkill -f X ; sleep 1 ; conda activate Y && python launch.py` returns the non-zero exit code from the last branch when something in the pre-amble exits 144 (signal-context) or 1; the harness reports the whole chain as failed even when the launch succeeded.
- When chained, the user can't see status between steps — e.g. they don't see "killed N processes" before the new launch starts.
- The cumulative output looks like the command hung when really it's just one tool call streaming logs slowly through a pipe.

**How to apply:** any time you want to restart a process,
1. one Bash call: `kill <pid>` (or `pkill -f <name>`) — let any non-zero exit return naturally, don't suppress with `|| true` in a chain
2. one Bash call: `pgrep -af <name>` to confirm it's dead (optional but cheap)
3. separate Bash call (background): the actual launch

Same applies for any "tear-down + set-up" sequence. The few extra tool calls are worth it for the cleaner logs and clearer failure mode.
