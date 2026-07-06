"""Batch-calibrate every drink_study participant that has local Charuco footage.

For each participant under $OT_CLIPS_ROOT (or /home/imove/Documents/clips) that has
`P<NN>_calibration_<ts>.<cam>.mp4` files and no existing data/calib/<P>/calibration.toml,
this links the first calibration session's footage as cam-1.mp4 .. cam-N.mp4 into
data/calib/<P>/ and runs recording/run_calibration.py on it.

Idempotent: skips participants that already have a calibration.toml (so re-running
only fills gaps — never re-does the ~minutes-each aniposelib bundle adjust).
Duplicate clip dirs like "P03 (1)" are ignored; only the canonical "P<NN>" dir is used.

Prints per-participant progress (flush=True) so the log is never silently quiet.

Usage:
    python experiments/drink_study/calibrate_all.py                 # all uncalibrated
    python experiments/drink_study/calibrate_all.py --only P04,P05  # subset
    python experiments/drink_study/calibrate_all.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from _paths import ROOT
CLIPS = Path(os.environ.get("OT_CLIPS_ROOT", "/home/imove/Documents/clips"))
CALIB_DIR = ROOT / "data" / "calib"
RUN_CALIB = ROOT / "recording" / "run_calibration.py"

CALIB_RE = re.compile(r"_calibration_(\d{8}_\d{6})\.(\d+)\.mp4$")


def sessions(pdir: Path) -> dict[str, dict[int, Path]]:
    """ts -> {cam_number: file} for every calibration capture in a participant dir."""
    out: dict[str, dict[int, Path]] = {}
    for f in pdir.iterdir():
        m = CALIB_RE.search(f.name)
        if m:
            out.setdefault(m.group(1), {})[int(m.group(2))] = f
    return out


def participant_dirs() -> list[Path]:
    # canonical "P<NN>" only — skip "P03 (1)" style duplicates
    return sorted(d for d in CLIPS.iterdir()
                  if d.is_dir() and re.fullmatch(r"P\d+", d.name))


def link_session(p: str, cams: dict[int, Path]) -> Path:
    dst = CALIB_DIR / p
    dst.mkdir(parents=True, exist_ok=True)
    for n, src in sorted(cams.items()):
        link = dst / f"cam-{n}.mp4"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(src)
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma list of participants (P04,P05)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None

    todo = []
    for pdir in participant_dirs():
        p = pdir.name
        if only and p not in only:
            continue
        if (CALIB_DIR / p / "calibration.toml").exists():
            print(f"[skip] {p}: already calibrated", flush=True)
            continue
        sess = sessions(pdir)
        if not sess:
            print(f"[skip] {p}: NO calibration footage", flush=True)
            continue
        ts = sorted(sess)[0]                       # earliest session
        cams = sess[ts]
        todo.append((p, ts, cams))
        print(f"[plan] {p}: session {ts}, {len(cams)} cams {sorted(cams)}", flush=True)

    print(f"\n{len(todo)} participant(s) to calibrate\n", flush=True)
    if args.dry_run:
        return

    results = []
    for i, (p, ts, cams) in enumerate(todo, 1):
        dst = link_session(p, cams)
        print(f"\n===== [{i}/{len(todo)}] calibrating {p} "
              f"({len(cams)} cams, session {ts}) =====", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, str(RUN_CALIB), str(dst)],
                           capture_output=True, text=True)
        dt = time.time() - t0
        err = None
        for line in r.stdout.splitlines():
            if "reprojection error" in line:
                err = line.split("=")[-1].strip()
        ok = r.returncode == 0 and (dst / "calibration.toml").exists()
        status = f"OK err={err}px" if ok else f"FAILED rc={r.returncode}"
        print(f"----- {p}: {status} in {dt:.0f}s -----", flush=True)
        if not ok:
            print(r.stdout[-2000:], flush=True)
            print(r.stderr[-2000:], flush=True)
        results.append((p, ok, err, dt))

    print("\n===== SUMMARY =====", flush=True)
    for p, ok, err, dt in results:
        print(f"  {p}: {'OK' if ok else 'FAIL'}  err={err}  {dt:.0f}s", flush=True)
    nfail = sum(1 for _, ok, _, _ in results if not ok)
    print(f"{len(results)-nfail}/{len(results)} succeeded", flush=True)


if __name__ == "__main__":
    main()
