"""Are the per-trial mocap->W0 fits CONSISTENT within a recording session?

The mocap lab and the camera rig are both FIXED, so mocap->W0 is a property of the ROOM, not
the trial -- within one session it should be ONE constant (R, t). Fitting per-trial is what lets
the cup's rotational-symmetry degeneracy pick a different (wrong) rotation branch each trial. If
the per-trial fits ARE tightly clustered within a session, that validates using a single session-
wide fit (and the outliers ARE the degenerate reps to override).

Uses the ORIGINAL fit (F.mocap_to_w0 on position centroids, current method). Groups reps by
session = participant + recording DATE (P03 has 2 sessions; cameras can differ across sessions).
For each session reports the SPREAD of rotation (pairwise angle between trial R's, deg) and
translation (mm) around the session median. Cache-only.
"""
from __future__ import annotations
import glob
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F
from mocap import load_trial


def _session_key(video):
    """participant + recording date -> session id.  e.g. P16_..._20240306_105728 -> P16@20240306"""
    pid = video.split("_")[0]
    m = re.search(r"(20\d{6})", video)
    date = m.group(1) if m else "?"
    return f"{pid}@{date}"


def _rot_angle(R0, R1):
    c = (np.trace(R0.T @ R1) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def main():
    files = sorted(glob.glob(str(F.FUSED_DIR / "*.npz")))
    idx = F.align_index()
    print(f"per-trial ORIGINAL position-Kabsch fits over {len(files)} reps\n", flush=True)
    sessions = defaultdict(list)   # key -> list of (video, R, t, rms)
    for i, f in enumerate(files):
        try:
            npz = np.load(f, allow_pickle=True)
        except Exception:
            continue
        video = str(npz["video"])
        if video not in idx:
            continue
        r = idx[video]
        tr = load_trial(r["c3d"])
        # ORIGINAL fit: on RAW consensus (what build_rep uses), position centroids
        raw = np.asarray(npz["cons"], float) if "cons" in npz else np.asarray(npz["fused"], float)
        fit = F.mocap_to_w0(raw, tr.centroid(), tr.rate, r["lag"])
        if fit is None:
            continue
        R, t, rms = fit
        sessions[_session_key(video)].append((video, R, t, rms))
        if (i + 1) % 100 == 0 or i == len(files) - 1:
            print(f"  [{i+1}/{len(files)}] fits done, {len(sessions)} sessions", flush=True)

    print(f"\n{'session':<16}{'n':>4}{'rot_med':>9}{'rot_p90':>9}{'rot_max':>9}"
          f"{'t_spread_mm':>13}{'outliers':>10}")
    all_rot_spread, all_t_spread, tight = [], [], 0
    for key in sorted(sessions):
        reps = sessions[key]
        if len(reps) < 3:
            continue
        Rs = [x[1] for x in reps]; ts = np.array([x[2] for x in reps])
        # session-median rotation via chordal mean (SVD of mean R), then angle of each to it
        Rbar = np.mean(Rs, axis=0)
        U, _, Vt = np.linalg.svd(Rbar)
        D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
        Rmed = U @ D @ Vt
        rot_dev = np.array([_rot_angle(Rmed, R) for R in Rs])
        tmed = np.median(ts, axis=0)
        t_dev = np.linalg.norm(ts - tmed, axis=1)
        # outliers = trials >20deg from the session rotation (candidate degenerate fits)
        out = [reps[j][0].split("_", 1)[0] + " " + re.search(r"(\d{6})(?=__|$|\.)", reps[j][0]).group(1)
               if re.search(r"(\d{6})(?=__|$|\.)", reps[j][0]) else reps[j][0][:12]
               for j in np.where(rot_dev > 20)[0]]
        all_rot_spread.append(np.median(rot_dev)); all_t_spread.append(np.median(t_dev))
        if np.percentile(rot_dev, 90) < 10:
            tight += 1
        print(f"{key:<16}{len(reps):>4}{np.median(rot_dev):9.1f}{np.percentile(rot_dev,90):9.1f}"
              f"{rot_dev.max():9.1f}{np.median(t_dev):13.1f}{len(out):>10}", flush=True)

    ars = np.array(all_rot_spread); ats = np.array(all_t_spread)
    print(f"\n  sessions (>=3 reps): {len(ars)}")
    print(f"  per-session MEDIAN rotation deviation from session R: "
          f"median {np.median(ars):.1f}deg  (tight sessions p90<10deg: {tight}/{len(ars)})")
    print(f"  per-session MEDIAN translation deviation: median {np.median(ats):.1f}mm")
    print(f"\n  => if rotation spread is small (~few deg), a SINGLE session fit is valid and the"
          f"\n     >20deg trials are the degenerate reps to override with the session R.", flush=True)


if __name__ == "__main__":
    main()
