"""Inter-camera agreement: a no-ground-truth precision metric.

Run a model on a rep's 10 synchronized cameras, triangulate the cup centroid
(box center) from the cameras that detect it, reproject the 3D point back into
each camera, and measure the pixel reprojection error. Low error = the views
geometrically agree on one 3D cup => precise, consistent detections. High error
or failed triangulation = sloppy / inconsistent => the model is off.

This captures what the presence-based recall/P_loose metric is blind to
(localization precision), and doubles as the runtime self-check.

CLI:
    python experiments/drink_study/agreement.py --weights best.pt \\
        --calib data/calib/P06/calibration.toml \\
        --rep-dir data/clips/.../P06_one_rep   # 10 cam-N drinking clips
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kalman_3d import load_calibration, triangulate_dlt, project

RES = (1920, 1080)


def _cam_key(clip_name: str) -> str | None:
    """drinking clip '..._<ts>.7.mp4' -> calib key 'cam_7'."""
    m = re.search(r"\.(\d+)\.mp4$", clip_name)
    return f"cam_{int(m.group(1))}" if m else None


def cup_centroids(model, clip: Path, conf: float,
                  classes: list[int] | None = None) -> list[tuple[float, float] | None]:
    """Per-frame top-confidence cup box center (or None).

    Pass `classes` (e.g. COCO cup-like) when the model is multi-class, so we only
    keep cup detections; leave None for a single-class fine-tuned student.
    """
    cap = cv2.VideoCapture(str(clip))
    out = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        r = model(f, conf=conf, classes=classes, verbose=False)[0]
        if r.boxes is not None and len(r.boxes) > 0:
            i = int(r.boxes.conf.argmax())
            b = r.boxes.xyxy[i].cpu().numpy()
            out.append((float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2)))
        else:
            out.append(None)
    cap.release()
    return out


def detect_rep(model, rep_clips: dict[str, Path], conf: float,
               classes: list[int] | None = None, verbose: bool = False
               ) -> dict[str, list]:
    """Per-camera per-frame cup centroids for one rep."""
    dets = {}
    for i, (cam, clip) in enumerate(sorted(rep_clips.items()), 1):
        dets[cam] = cup_centroids(model, clip, conf, classes)
        if verbose:
            hit = sum(1 for d in dets[cam] if d is not None)
            print(f"    [{i}/{len(rep_clips)}] {cam}: {hit}/{len(dets[cam])} frames", flush=True)
    return dets


def triangulate_per_frame(dets: dict[str, list], calib: dict,
                          inlier_px: float = 30.0) -> list[dict | None]:
    """Per-frame triangulation; entry is None when <2 cams saw the cup."""
    n = min(len(v) for v in dets.values())
    out: list[dict | None] = []
    for t in range(n):
        obs = {c: dets[c][t] for c in dets if dets[c][t] is not None}
        if len(obs) < 2:
            out.append(None)
            continue
        cams = [calib[c] for c in obs]
        pts = [np.array(obs[c]) for c in obs]
        X = triangulate_dlt(cams, pts)
        errs = []
        for c in obs:
            uv, infront = project(calib[c], X)
            errs.append(float(np.hypot(*(uv - np.array(obs[c])))) if infront else 1e6)
        errs = np.array(errs)
        out.append({"median_err": float(np.median(errs)), "n_cams": len(obs),
                    "inlier_frac": float(np.mean(errs < inlier_px))})
    return out


def summarize(per_frame: list[dict | None]) -> dict:
    valid = [p for p in per_frame if p]
    if not valid:
        return {"frames_triangulated": 0}
    return {
        "frames_triangulated": len(valid),
        "frames_total": len(per_frame),
        "tri_rate": round(len(valid) / len(per_frame), 3),
        "median_reproj_px": round(float(np.median([p["median_err"] for p in valid])), 2),
        "mean_reproj_px": round(float(np.mean([p["median_err"] for p in valid])), 2),
        "mean_cams_agreeing": round(float(np.mean([p["n_cams"] for p in valid])), 2),
        "mean_inlier_frac": round(float(np.mean([p["inlier_frac"] for p in valid])), 3),
    }


def agreement_for_rep(model, rep_clips: dict[str, Path], calib: dict,
                      conf: float = 0.25, inlier_px: float = 30.0,
                      classes: list[int] | None = None, verbose: bool = False) -> dict:
    """rep_clips: {cam_key: clip_path} for one rep's cameras that have calib."""
    dets = detect_rep(model, rep_clips, conf, classes, verbose)
    return summarize(triangulate_per_frame(dets, calib, inlier_px))


def agreement_eval(weights: str, participants: list[str], reps: int,
                   conf: float = 0.25, classes: list[int] | None = None,
                   hand: str = "right", calib_root: str = "data/calib",
                   clips_root: str = "/home/imove/Documents/clips",
                   verbose: bool = False) -> dict:
    """Aggregate inter-camera agreement for one model across held-out
    participants (each with its own calibration). Returns coverage + fair-ish
    precision summary; `per_rep` holds the breakdown."""
    from ultralytics import YOLO
    model = YOLO(weights)
    per_rep = []
    for p in participants:
        calib_path = Path(calib_root) / p / "calibration.toml"
        if not calib_path.exists():
            print(f"  [agreement] no calib for {p}, skipping", flush=True)
            continue
        calib = load_calibration(calib_path, target_size=RES)
        pdir = Path(clips_root) / p
        stems = sorted({f.name.rsplit(".", 2)[0]
                        for f in pdir.glob(f"{p}_drinking_{hand}_*.mp4")})
        for stem in stems[:reps]:
            rep_clips = {f"cam_{c}": pdir / f"{stem}.{c}.mp4"
                         for c in range(1, 11)
                         if f"cam_{c}" in calib and (pdir / f"{stem}.{c}.mp4").exists()}
            if len(rep_clips) < 2:
                continue
            if verbose:
                print(f"  [agreement] {p} {stem}", flush=True)
            dets = detect_rep(model, rep_clips, conf, classes, verbose)
            s = summarize(triangulate_per_frame(dets, calib))
            s.update({"participant": p, "rep": stem})
            per_rep.append(s)
    valid = [r for r in per_rep if r.get("frames_triangulated", 0) > 0]
    if not valid:
        return {"reps": 0, "per_rep": per_rep}
    return {
        "reps": len(valid),
        "tri_rate": round(float(np.mean([r["tri_rate"] for r in valid])), 3),
        "mean_cams_agreeing": round(float(np.mean([r["mean_cams_agreeing"] for r in valid])), 2),
        "median_reproj_px": round(float(np.median([r["median_reproj_px"] for r in valid])), 2),
        "mean_inlier_frac": round(float(np.mean([r["mean_inlier_frac"] for r in valid])), 3),
        "per_rep": per_rep,
    }


def agreement_eval_full(weights: str, participants: list[str], reps: int,
                        conf: float = 0.25, classes: list[int] | None = None,
                        hand: str = "right", calib_root: str = "data/calib",
                        clips_root: str = "/home/imove/Documents/clips",
                        verbose: bool = False) -> dict:
    """Like `agreement_eval`, but each per_rep entry also keeps the full
    `per_frame` triangulation list (median_err / n_cams / inlier_frac per frame).
    Lets us recover the per-frame distribution later (e.g. inspect why an epoch's
    median spiked) without re-running the slow detection on the saved .pt."""
    from ultralytics import YOLO
    model = YOLO(weights)
    per_rep = []
    for p in participants:
        calib_path = Path(calib_root) / p / "calibration.toml"
        if not calib_path.exists():
            print(f"  [agreement] no calib for {p}, skipping", flush=True)
            continue
        calib = load_calibration(calib_path, target_size=RES)
        pdir = Path(clips_root) / p
        stems = sorted({f.name.rsplit(".", 2)[0]
                        for f in pdir.glob(f"{p}_drinking_{hand}_*.mp4")})
        for stem in stems[:reps]:
            rep_clips = {f"cam_{c}": pdir / f"{stem}.{c}.mp4"
                         for c in range(1, 11)
                         if f"cam_{c}" in calib and (pdir / f"{stem}.{c}.mp4").exists()}
            if len(rep_clips) < 2:
                continue
            if verbose:
                print(f"  [agreement] {p} {stem}", flush=True)
            dets = detect_rep(model, rep_clips, conf, classes, verbose)
            pf = triangulate_per_frame(dets, calib)
            s = summarize(pf)
            s.update({"participant": p, "rep": stem,
                      "per_frame": [None if x is None else
                                    {"median_err": round(x["median_err"], 3),
                                     "n_cams": x["n_cams"],
                                     "inlier_frac": round(x["inlier_frac"], 3)}
                                    for x in pf]})
            per_rep.append(s)
    valid = [r for r in per_rep if r.get("frames_triangulated", 0) > 0]
    out = {"reps": len(valid), "per_rep": per_rep}
    if valid:
        out.update({
            "tri_rate": round(float(np.mean([r["tri_rate"] for r in valid])), 3),
            "mean_cams_agreeing": round(float(np.mean([r["mean_cams_agreeing"] for r in valid])), 2),
            "median_reproj_px": round(float(np.median([r["median_reproj_px"] for r in valid])), 2),
            "mean_reproj_px": round(float(np.mean([r["mean_reproj_px"] for r in valid])), 2),
            "mean_inlier_frac": round(float(np.mean([r["mean_inlier_frac"] for r in valid])), 3),
        })
    return out


def load_rep(rep_dir: Path, calib: dict) -> dict[str, Path]:
    rep = {}
    for clip in sorted(rep_dir.glob("*.mp4")):
        ck = _cam_key(clip.name)
        if ck and ck in calib:
            rep[ck] = clip
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("--rep-dir", type=Path, required=True,
                    help="dir with one rep's cam-N drinking clips")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()
    from ultralytics import YOLO
    calib = load_calibration(args.calib, target_size=RES)
    rep = load_rep(args.rep_dir, calib)
    print(f"calib cams: {sorted(calib)}\nrep cams: {sorted(rep)}")
    model = YOLO(args.weights)
    summary = agreement_for_rep(model, rep, calib, conf=args.conf)
    print(summary)


if __name__ == "__main__":
    main()
