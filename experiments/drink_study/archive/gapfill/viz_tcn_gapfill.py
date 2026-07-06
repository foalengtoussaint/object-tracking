"""Rerun viewer for the VELOCITY-FILL TCN vs the plain KF at the drinking apex.

Shows, for one drinking rep, three 3D cup tracks on a scrubbable timeline:
  - mocap TRUTH        (green)  -- sub-mm QTM cup
  - KF baseline        (red)    -- Kalman track; COASTS through the apex occlusion and
                                   under-reaches / drifts (this is what we improve on)
  - VELOCITY-FILL TCN  (cyan)   -- predicts cup MOVEMENT across the gap and integrates,
                                   anchored to the gap-exit consensus -> reaches the apex

The OCCLUSION GAP (where the cup is lost and both methods must fill) is highlighted: the
cup marker turns bright during the gap, and a text log reports the live mm error of KF vs
TCN vs the mocap truth so you can watch the KF diverge while the TCN tracks.

Same LOPO protocol as learn_velocity_fill.py: the held-out participant's fold is trained
fresh (GPU), so the shown track is a genuine leave-one-out prediction, not a fit. A synced
best-camera panel shows the real occlusion.

    python experiments/drink_study/viz_tcn_gapfill.py                        # default rep, spawn viewer
    python experiments/drink_study/viz_tcn_gapfill.py P15_..._105000         # a specific rep (substring)
    python experiments/drink_study/viz_tcn_gapfill.py --no-frames            # 3D only, fast
    python experiments/drink_study/viz_tcn_gapfill.py --save gap.rrd         # write a recording
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import rerun as rr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import learn_velocity_fill as VF          # build_seq/fill_track/TCN/SEQ/DEV + prep
from learn_seq_kf import build_seq, TCN, SEQ, DEV
import tune_interp as T
import qtm_align as Q                      # sync + robust Kabsch, to put mocap in the video frame
import render_phase_compare as RP
import gpu_decode

MM_PER_M = 1000.0
HZ = 60.0
DEFAULT = "P15_P15_drinking_left_20240229_105022__clean3d_refill"


def prep_all():
    """Replicate learn_velocity_fill.main()'s rep preparation (adds local frames / gap masks)."""
    reps = [r for rep in T._reps() if (r := build_seq(rep)) is not None]
    for r in reps:
        valid = np.isfinite(r["cons"]).all(1)
        r["valid"] = valid
        rest = r["rest"]; basis = r["basis"]
        cl = np.zeros((len(r["cons"]), 3)); cl[valid] = (r["cons"][valid] - rest) @ basis.T
        r["cons_local_t"] = cl
        r["base_local_t"] = (r["base"] - rest) @ basis.T
        tgt = r["tgt"]
        vel = np.vstack([np.zeros(3), np.diff(tgt, axis=0)]).astype(np.float32)
        r["vel_tgt"] = vel
        x0 = np.linspace(0, 1, len(valid)); x1 = np.linspace(0, 1, SEQ)
        r["gap_seq"] = (np.interp(x1, x0, valid.astype(float)) <= 0.5) & r["tmask"]
    return reps


def train_fold_and_fill(reps, held, target_video):
    """Train the held-out fold (excluding `held`); return the filled world track for target."""
    trn = [r for r in reps if r["pid"] != held]
    X = torch.tensor(np.stack([r["feats"] for r in trn])).transpose(1, 2).to(DEV)
    V = torch.tensor(np.stack([r["vel_tgt"] for r in trn])).transpose(1, 2).to(DEV)
    Gm = torch.tensor(np.stack([r["gap_seq"] for r in trn]).astype(np.float32)).to(DEV)
    fin = reps[0]["feats"].shape[1]
    net = TCN(fin, 3).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    net.train()
    for _ in range(300):
        opt.zero_grad()
        pv = net(X); m = Gm.unsqueeze(1)
        loss = (((pv - V) ** 2).sum(1, keepdim=True) * m).sum() / m.sum().clamp(min=1)
        loss.backward(); opt.step()
    net.eval()
    r = next(r for r in reps if r["video"] == target_video)
    with torch.no_grad():
        pv = net(torch.tensor(r["feats"][None]).transpose(1, 2).to(DEV))[0].cpu().numpy().T
    x0 = np.linspace(0, 1, SEQ); x1 = np.linspace(0, 1, r["Tn"])
    vel_t = np.stack([np.interp(x1, x0, pv[:, k]) for k in range(3)], 1)
    filled_local = VF.fill_track(r["base_local_t"], r["cons_local_t"], r["valid"], vel_t)
    track_world = filled_local @ r["basis"] + r["rest"]
    return r, track_world


def mocap_in_video_frame(ref_track, mr, Tn):
    """Put the mocap truth `mr` into the video coordinate frame, aligned to `ref_track`
    (the KF baseline), using the SAME sync + robust Kabsch as tune_interp._score. Returns
    a (Tn,3) array on the ref_track's frame index (NaN where mocap doesn't overlap)."""
    vr = Q._resample(ref_track, Q.VIDEO_FPS)
    lag, corr = Q._xcorr_lag(Q._speed(vr, Q.COMMON_HZ), Q._speed(mr, Q.COMMON_HZ))
    if corr < Q.MIN_SYNC_CORR:
        return np.full((Tn, 3), np.nan), corr
    # overlap window (mirrors _score), then Kabsch mocap->video
    if lag >= 0:
        v = vr[lag:]; mo = mr[:len(v)]; voff = lag
    else:
        mo = mr[-lag:]; v = vr[:len(mo)]; voff = 0
    L = min(len(v), len(mo)); v, mo = v[:L], mo[:L]
    ok = ~(np.isnan(v).any(1) | np.isnan(mo).any(1))
    R, t, _ = Q.kabsch(mo[ok], v[ok], robust=True)
    mo_v = mo @ R.T + t                                  # mocap in video frame, on the vr grid
    # scatter back onto the Tn-length track index (vr is at VIDEO_FPS ~= track grid)
    out = np.full((Tn, 3), np.nan)
    for i in range(L):
        j = voff + i
        if 0 <= j < Tn:
            out[j] = mo_v[i]
    return out, corr


def log_track(name, xyz, color, radius=0.004):
    m = np.isfinite(xyz).all(1)
    if m.any():
        rr.log(name, rr.LineStrips3D([xyz[m] / MM_PER_M], colors=[color], radii=[radius]), static=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rep", nargs="?", default=DEFAULT)
    ap.add_argument("--no-frames", action="store_true")
    ap.add_argument("--save", default=None)
    ap.add_argument("--spawn", action="store_true")
    ap.add_argument("--web-port", type=int, default=9091)
    ap.add_argument("--grpc-port", type=int, default=9877)
    args = ap.parse_args()

    print("building sequences (this loads the QTM + track caches)...", flush=True)
    reps = prep_all()
    byv = {r["video"]: r for r in reps}
    hit = [v for v in byv if args.rep in v or v in args.rep]
    if not hit:
        raise SystemExit(f"no velocity-fill rep matching '{args.rep}'")
    video = hit[0]; held = byv[video]["pid"]
    print(f"rep {video}  (held-out participant {held}); training LOPO fold ...", flush=True)
    r, tcn_world = train_fold_and_fill(reps, held, video)

    kf = np.asarray(r["base"], float)              # KF baseline (red)
    valid = np.asarray(r["valid"])
    Tn = min(len(kf), len(valid), len(tcn_world))
    kf, valid, tcn_world = kf[:Tn], valid[:Tn], tcn_world[:Tn]
    # mocap truth lives in the MOCAP frame — align it into the video frame (sync + Kabsch)
    # against the KF track so all three overlay meaningfully.
    truth, corr = mocap_in_video_frame(kf, np.asarray(r["mr"], float), Tn)
    print(f"  mocap->video sync corr={corr:.2f}", flush=True)
    # error vs truth on the apex/gap frames
    def med_gap_err(trk):
        m = (~valid) & np.isfinite(trk).all(1) & np.isfinite(truth).all(1)
        return float(np.median(np.linalg.norm(trk[m] - truth[m], axis=1))) if m.any() else float("nan")
    print(f"  gap frames: {int((~valid).sum())}/{Tn}   "
          f"apex-gap median err  KF={med_gap_err(kf):.0f}mm  TCN={med_gap_err(tcn_world):.0f}mm", flush=True)

    # --- sink ---
    rr.init("tcn_gapfill", recording_id=video)
    if args.save:
        rr.save(args.save)
    elif args.spawn:
        rr.spawn()
    else:
        import urllib.parse
        uri = rr.serve_grpc(grpc_port=args.grpc_port)
        rr.serve_web_viewer(web_port=args.web_port, open_browser=False, connect_to=uri)
        enc = urllib.parse.quote(f"rerun+http://127.0.0.1:{args.grpc_port}/proxy", safe="")
        print(f"  open http://127.0.0.1:{args.web_port}/?url={enc}", flush=True)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    log_track("world/truth_mocap", truth, [80, 230, 120], 0.003)
    log_track("world/kf", kf, [235, 80, 80])
    log_track("world/tcn", tcn_world, [70, 200, 235])

    # camera panel
    caps = None
    if not args.no_frames:
        try:
            tj = json.loads((RP.TRACK / f"{video}.json").read_text())
            cs = tj["stem"]; p = cs.split("_")[0]
            cam = RP.best_cam(p, cs); cn = cam.split("_")[1] if cam else "3"
            clip = RP.CLIPS / p / f"{cs}.{cn}.mp4"
            if clip.exists():
                caps = gpu_decode.frames(clip)
                print(f"  camera panel: {clip.name} (cam{cn})", flush=True)
        except Exception as e:
            print(f"  (camera panel skipped: {e})", flush=True)

    frame_iter = enumerate(caps) if caps is not None else ((fr, None) for fr in range(Tn))
    for fr, img in frame_iter:
        if fr >= Tn:
            break
        rr.set_time("frame", sequence=fr)
        rr.set_time("time", duration=fr / HZ)
        in_gap = not valid[fr]
        for nm, trk, col in [("world/pt_truth", truth, [80, 230, 120]),
                             ("world/pt_kf", kf, [235, 80, 80]),
                             ("world/pt_tcn", tcn_world, [70, 200, 235])]:
            if np.isfinite(trk[fr]).all():
                rad = 0.02 if (in_gap and nm != "world/pt_truth") else 0.013
                rr.log(nm, rr.Points3D((trk[fr] / MM_PER_M).reshape(1, 3), colors=[col], radii=[rad]))
        # live per-method error vs truth + gap flag
        def e(trk):
            return (np.linalg.norm(trk[fr] - truth[fr]) if np.isfinite(trk[fr]).all()
                    and np.isfinite(truth[fr]).all() else float("nan"))
        ek, et = e(kf), e(tcn_world)
        rr.log("err/KF", rr.Scalars(ek if ek == ek else 0.0))
        rr.log("err/TCN", rr.Scalars(et if et == et else 0.0))
        tag = f"{'GAP (occluded — filling)' if in_gap else 'visible'}   KF err={ek:5.1f}mm   TCN err={et:5.1f}mm"
        rr.log("status", rr.TextLog(tag, level="WARN" if in_gap else "INFO"))
        if img is not None:
            im = img[..., ::-1]
            h, w = im.shape[:2]
            if w > 960:
                import cv2
                im = cv2.resize(im, (960, int(h * 960 / w)))
            rr.log("camera/best", rr.Image(im).compress(jpeg_quality=75))

    tail = ("wrote " + args.save) if args.save else "viewer serving — scrub the 'frame' timeline"
    print(f"  done. {tail}", flush=True)
    if not args.save and not args.spawn:
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
