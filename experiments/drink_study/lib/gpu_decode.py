"""GPU (NVDEC) video frame decoding via ffmpeg, with a CPU fallback.

This cv2 build has no CUDA (cudacodec unavailable) and decord/PyNvVideoCodec
aren't installed, but ffmpeg here ships NVDEC (`-hwaccel cuda -c:v h264_cuvid`).
We decode on the GPU's dedicated NVDEC engine and pipe raw BGR24 frames to numpy
— offloading the CPU-bound H.264 decode that bottlenecks the batch jobs.

    from gpu_decode import frames, dims
    w, h, n, fps = dims(path)
    for img in frames(path):        # BGR uint8 (h,w,3), same as cv2
        ...

Set OT_FORCE_CPU_DECODE=1 to force the cv2 path. `gpu_available()` reports
whether NVDEC h264 is usable.
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import os, shutil, subprocess
from functools import lru_cache
from pathlib import Path
import numpy as np
import cv2


def dims(path) -> tuple[int, int, int, float]:
    c = cv2.VideoCapture(str(path))
    w, h = int(c.get(3)), int(c.get(4))
    n, fps = int(c.get(7)), c.get(5)
    c.release()
    return w, h, n, fps


@lru_cache(maxsize=1)
def gpu_available() -> bool:
    if os.environ.get("OT_FORCE_CPU_DECODE"):
        return False
    if not shutil.which("ffmpeg"):
        return False
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-decoders"],
                             capture_output=True, text=True, timeout=10).stdout
        return "h264_cuvid" in out
    except Exception:
        return False


def _gpu_frames(path, w, h):
    fsize = w * h * 3
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-hwaccel", "cuda", "-c:v", "h264_cuvid", "-i", str(path),
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            bufsize=fsize)
    try:
        while True:
            buf = proc.stdout.read(fsize)
            if len(buf) < fsize:
                break
            # frombuffer is read-only; copy so callers (cv2.circle etc.) can draw
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
    finally:
        proc.stdout.close()
        proc.wait()


def _cpu_frames(path):
    cap = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            yield img
    finally:
        cap.release()


def frames(path):
    """Yield BGR uint8 frames; GPU (NVDEC) if available, else cv2 CPU."""
    if gpu_available():
        w, h, _, _ = dims(path)
        if w > 0 and h > 0:
            yield from _gpu_frames(path, w, h)
            return
    yield from _cpu_frames(path)
