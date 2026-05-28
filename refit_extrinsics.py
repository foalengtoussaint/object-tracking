"""Refit per-camera extrinsics in-place by solving PnP against one ArUco
marker that every camera can see. Keeps the May-21 iMOVE intrinsics (K +
distortion) since those have been verified to give sub-pixel reprojection,
but recomputes each cam's (R, t) in a new world frame anchored on the
chosen marker.

Why: the iMOVE TOML pose was correct as of May 21, but ≥1 camera has been
physically bumped since then, so the relative cam poses are now wrong (~80
px multi-cam reprojection error vs <1 px single-cam). Re-shooting Charuco
would take ~10 min; this script takes ~2 s.

Output TOML uses ZMQ stream names directly (`[cam_1]`, `[cam_2]`, …), so
`kalman_3d.load_calibration` reads it without needing a serial-mapping
layer. Cam-to-stream identification is by USB serial:

    /dev/videoN  --(sysfs)-->  serial  --(iMOVE meta)-->  iMOVE cam name
                 --(iMOVE TOML)-->  K, dist
    (then solvePnP marker → new R, t)

Usage:
    python refit_extrinsics.py                       # 92 mm marker 18, default I/O
    python refit_extrinsics.py --anchor 10 --marker-size 50
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import toml
import yaml
import zmq

from recording.cam_streams import load_streams
from live_track import detect_aruco_any, decode

DEFAULT_CAL = ("/home/imove/Documents/iMOVE/DEV/isr-supplementary/DATA/"
               "sub-105/20260521-155143/calibration/20260521-155143/"
               "trials/calibration_20260521_155143_calibration.toml")
DEFAULT_META = ("/home/imove/Documents/iMOVE/DEV/isr-supplementary/DATA/"
                "sub-105/20260521-155143/calibration/20260521-155143/"
                "_recording_meta.json")
DEFAULT_OUT = Path("data/calibration_local.toml")
DEFAULT_CONFIG = Path("recording/teleimager/cam_config_server.yaml")


def serial_for_video(v_name: str) -> str | None:
    """Walk sysfs from /dev/<v_name> up to the USB device with a `serial`."""
    cur = os.path.realpath(f"/sys/class/video4linux/{v_name}/device")
    while cur and not os.path.exists(os.path.join(cur, "serial")):
        p = os.path.dirname(cur)
        if p == cur:
            return None
        cur = p
    try:
        return open(os.path.join(cur, "serial")).read().strip()
    except OSError:
        return None


def build_intrinsics(cal_path: str, meta_path: str, yaml_path: Path
                     ) -> dict[str, tuple[np.ndarray, np.ndarray, tuple[int, int], str]]:
    """Return {zmq_name: (K_at_cal_res, dist, cal_size, imove_name)}.

    K is left at the calibration's native resolution; rescaling happens at
    write time once we know the live frame size.
    """
    meta = json.load(open(meta_path))
    serial_to_iname = {c["serial"]: str(c["idx"]) for c in meta["cameras"]}

    data = toml.load(cal_path)
    intr_by_iname = {}
    for k, b in data.items():
        if not k.startswith("cam_") or not isinstance(b, dict):
            continue
        intr_by_iname[b["name"]] = (
            np.array(b["matrix"], float),
            np.array(b["distortions"], float),
            tuple(b["size"]),
        )

    ycfg = yaml.safe_load(open(yaml_path))
    out = {}
    for zname, c in ycfg.items():
        if not isinstance(c, dict) or not c.get("enable_zmq"):
            continue
        serial = serial_for_video(f"video{c['video_id']}")
        if serial is None:
            print(f"  {zname}: no serial readable from /dev/video{c['video_id']}")
            continue
        iname = serial_to_iname.get(serial)
        if iname is None or iname not in intr_by_iname:
            print(f"  {zname}: serial {serial} not in iMOVE meta")
            continue
        K, dist, sz = intr_by_iname[iname]
        out[zname] = (K, dist, sz, iname)
        print(f"  {zname}: serial={serial} -> iMOVE name {iname!r} -> intrinsics OK")
    return out


def grab_one_frame_each(streams: dict[str, int], host: str, timeout_s: float
                        ) -> dict[str, np.ndarray]:
    ctx = zmq.Context.instance()
    socks = {}
    for n, port in streams.items():
        s = ctx.socket(zmq.SUB)
        s.setsockopt(zmq.CONFLATE, 1)
        s.setsockopt_string(zmq.SUBSCRIBE, "")
        s.connect(f"tcp://{host}:{port}")
        socks[n] = s
    poller = zmq.Poller()
    for s in socks.values():
        poller.register(s, zmq.POLLIN)
    frames = {}
    t0 = time.time()
    while time.time() - t0 < timeout_s and len(frames) < len(socks):
        ev = dict(poller.poll(200))
        for n, s in socks.items():
            if s in ev and n not in frames:
                f = decode(s.recv(flags=zmq.NOBLOCK))
                if f is not None:
                    frames[n] = f
    for s in socks.values():
        s.close()
    return frames


def write_toml(out_path: Path, refit: dict[str, dict], meta: dict) -> None:
    """Write a minimal aniposelib-compatible TOML keyed by stream name."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": meta}
    for zname, r in refit.items():
        payload[zname] = {
            "name": zname,
            "size": list(r["size"]),
            "matrix": [list(map(float, row)) for row in r["K"]],
            "distortions": list(map(float, r["dist"])),
            "rotation": list(map(float, r["rvec"])),
            "translation": list(map(float, r["tvec"])),
        }
    out_path.write_text(toml.dumps(payload))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cal", default=DEFAULT_CAL)
    ap.add_argument("--meta", default=DEFAULT_META)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--anchor", type=int, default=18,
                    help="ArUco marker id used as world origin")
    ap.add_argument("--marker-size", type=float, default=92.0,
                    help="Anchor marker side length in mm "
                         "(iMOVE Charuco default 92)")
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    print(f"intrinsics from: {args.cal}")
    intr = build_intrinsics(args.cal, args.meta, args.config)
    if not intr:
        raise SystemExit("no cameras mapped — check serials + iMOVE meta")

    print(f"\ngrabbing one frame per cam...")
    streams = load_streams(args.config)
    frames = grab_one_frame_each(streams, args.host, timeout_s=5.0)
    print(f"  got frames from {sorted(frames)}")

    # Rescale K to live frame resolution
    refit = {}
    s = args.marker_size / 2.0
    obj_pts = np.array([[-s,  s, 0], [ s,  s, 0],
                        [ s, -s, 0], [-s, -s, 0]], dtype=np.float32)
    residuals = []
    for zname, frame in frames.items():
        if zname not in intr:
            print(f"  {zname}: skip (no intrinsics)")
            continue
        K_cal, dist, cal_size, iname = intr[zname]
        h, w = frame.shape[:2]
        sx, sy = w / cal_size[0], h / cal_size[1]
        K = K_cal.copy()
        K[0] *= sx
        K[1] *= sy

        _, dets = detect_aruco_any(frame)
        if args.anchor not in dets:
            print(f"  {zname}: anchor marker {args.anchor} NOT visible "
                  f"(saw {sorted(dets)})")
            continue
        corners = dets[args.anchor].astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(obj_pts, corners, K, dist)
        if not ok:
            print(f"  {zname}: solvePnP failed")
            continue
        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
        resid = float(np.linalg.norm(proj.reshape(-1, 2) - corners, axis=1).mean())
        residuals.append(resid)
        refit[zname] = {
            "K": K, "dist": dist, "size": (w, h),
            "rvec": rvec.flatten(), "tvec": tvec.flatten(),
        }
        print(f"  {zname}: solvePnP OK  resid={resid:5.3f} px  "
              f"t={tvec.flatten().round(1)} mm")

    if not refit:
        raise SystemExit(f"anchor marker {args.anchor} not visible in any cam")

    meta = {
        "source": "refit_extrinsics",
        "anchor_marker": args.anchor,
        "marker_size_mm": args.marker_size,
        "mean_resid_px": float(np.mean(residuals)) if residuals else 0.0,
    }
    write_toml(args.out, refit, meta)
    print(f"\nwrote {args.out}  ({len(residuals)} cams,  "
          f"mean single-cam resid = {np.mean(residuals):.3f} px)")
    print(f"next:  python live_track.py --check --cal {args.out}")


if __name__ == "__main__":
    main()
