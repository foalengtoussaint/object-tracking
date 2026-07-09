"""Multi-angle overlay: project the SAME OMC head/cup + biomech head + tracked cup into
several calibrated cameras and tile them into one grid video. Lets you see WHY a rep fails —
occlusion is view-dependent, so a landmark bad in one cam may be clean in another.

Reuses overlay.py's data + fit (features.mocap_to_w0 = the SAME alignment the model uses) and
kf_accuracy.project. One shared fit; each cam just re-projects the shared W0 points.

    python experiments/drink_dwell/overlay_multicam.py <rep-substring> [--cams 2,4,6,8] [--out ...]

Markers (per cam):  green=OMC cup markers  blueX=OMC head  yellow=OMC cup centroid
                    magentaX=tracked cup(raw)  orange=tracked cup(fused)  white=biomech head67
"""
from __future__ import annotations
import sys, pathlib, os, argparse, json, glob, re
import numpy as np, cv2

HERE = pathlib.Path(__file__).resolve().parent
_DS = HERE.parents[0] / "drink_study"
os.chdir(HERE.parents[1])                       # repo root, so data/calib/... resolves
# same path shim as overlay.py: drink_study/lib (kf_accuracy) + repo-parent, then HERE last
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(_DS / "lib"))
sys.path.insert(0, str(_DS.parents[1]))
import kf_accuracy as ka
from mocap import load_trial, VIDEO_FPS, CUP_MARKERS, HEAD_MARKERS, resample as _rs
import features as F

RES = (1920, 1080)
CLIPS = pathlib.Path(os.environ.get("OT_CLIPS_ROOT", str(HERE.parents[1] / "clips")))
TRACK = HERE.parents[0] / "drink_study" / "cache" / "track3d_clean3d_refill"
ALIGN = HERE.parents[0] / "drink_study" / "cache" / "qtm_align.json"
CACHE = HERE.parents[0] / "drink_study" / "cache"
OUTDIR = HERE / "renders"
TILE_W = 640                                    # per-cam tile width


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rep")
    ap.add_argument("--cams", default=None, help="comma cam list, e.g. 2,4,6,8. default=all calibrated")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    V = json.load(open(ALIGN))
    V = {r["video"]: r for ent in V.values() for r in ent.get("reps", [])}
    vk = [k for k in V if a.rep in k]
    if not vk:
        raise SystemExit(f"no aligned rep matching {a.rep!r}")
    video = vk[0]; rec = V[video]; c3d = rec["c3d"]; lag = rec["lag"]
    tj = json.loads((TRACK / f"{video}.json").read_text())
    stem = tj["stem"]; pid = stem.split("_")[0]

    calib = ka.load_calibration(f"data/calib/{pid}/calibration.toml", target_size=RES)
    # cams = requested ∩ calibrated ∩ has-clip
    want = [f"cam_{c.strip()}" for c in a.cams.split(",")] if a.cams else sorted(
        calib, key=lambda c: int(c.split("_")[1]))
    cams = [c for c in want if c in calib and (CLIPS / pid / f"{stem}.{c.split('_')[1]}.mp4").exists()]
    if not cams:
        raise SystemExit("no usable cams")
    print(f"{video}  c3d={c3d}  cams={cams}", flush=True)

    # --- shared tracks + fit (same as overlay.py) ---
    def _xyz(fr):
        for k in ("rts", "consensus", "kf"):
            if fr.get(k) is not None:
                return fr[k]
        return [np.nan] * 3
    cup_w0 = np.array([_xyz(fr) for fr in tj["frames"]], float)
    cup_raw = np.array([fr.get("consensus") if fr.get("consensus") is not None else [np.nan] * 3
                        for fr in tj["frames"]], float)
    cup_fused = None
    _npz = glob.glob(str(F.FUSED_DIR / f"*{video}*.npz"))
    if _npz:
        cup_fused = np.asarray(np.load(_npz[0], allow_pickle=True)["fused"], float)
    bio_head = None
    _base = video.replace("__clean3d_refill", ""); _single = re.sub(r'^(P\d+)_\1_', r'\1_', _base)
    _bm = next((str(CACHE / f"biomech_{c}.npz") for c in (_base, _single)
                if (CACHE / f"biomech_{c}.npz").exists()), None)
    if _bm:
        _kp = np.load(_bm, allow_pickle=True)["keypoints3d"]
        bio_head = _kp[:, 67, :3].astype(float); bio_head[_kp[:, 67, 3] < 0.1] = np.nan

    # --- per-participant SCALE correction (2026-07-09) -----------------------------------
    # P24/P16/P23/P02 markerless W0 is ~3-6% COMPRESSED (calib depth-scale defect). s_p =
    # median MMC/OMC chord ratio; we un-compress the W0/video side by dividing by s_p about a
    # common centre so OMC (which is metric) lines up. Only participants with |s_p-1|>3% move.
    s_p = 1.0
    _spf = CACHE / "per_participant_scale.json"
    if _spf.exists():
        s_p = json.load(open(_spf))["scale_per_participant"].get(pid, 1.0)
    if abs(s_p - 1.0) > 0.03:
        _all = [p for p in (cup_w0, cup_raw, cup_fused, bio_head) if p is not None]
        _c = np.nanmean(np.concatenate([p[np.isfinite(p).all(1)] for p in _all]), axis=0)
        for _p in _all:
            _p[:] = _c + (_p - _c) / s_p          # divide W0 by s_p about shared centre
        print(f"  SCALE-CORRECTED pid={pid} s_p={s_p:.3f} ({(s_p-1)*100:+.1f}%)", flush=True)

    tr = load_trial(c3d)
    R, t, rms = F.mocap_to_w0(cup_raw, tr.centroid(), tr.rate, lag)
    print(f"  Kabsch rms={rms:.1f}mm lag={lag} s_p={s_p:.3f}", flush=True)
    idx = {l: i for i, l in enumerate(tr.labels)}
    cupm = tr.markers[:, [idx[m] for m in CUP_MARKERS if m in idx]]
    headm = tr.markers[:, [idx[m] for m in HEAD_MARKERS if m in idx]]
    cup_c = tr.centroid(); head_c = tr.head_centroid()
    ratio = tr.rate / VIDEO_FPS

    def to_w0(X):
        return (X @ R.T) + t

    # --- open all cam clips ---
    caps = {c: cv2.VideoCapture(str(CLIPS / pid / f"{stem}.{c.split('_')[1]}.mp4")) for c in cams}
    Tv = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps.values())
    W0 = int(caps[cams[0]].get(cv2.CAP_PROP_FRAME_WIDTH))
    H0 = int(caps[cams[0]].get(cv2.CAP_PROP_FRAME_HEIGHT))
    th = int(TILE_W * H0 / W0)
    ncol = min(len(cams), 3); nrow = (len(cams) + ncol - 1) // ncol

    OUTDIR.mkdir(exist_ok=True)
    out = a.out or str(OUTDIR / f"MULTICAM_{stem}.mp4")
    wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 30, (TILE_W * ncol, th * nrow))

    def draw(img, X, color, sx, sy, r=5, x=False):
        if not np.isfinite(X).all():
            return
        (u, v), ok = ka.project(cam, X)
        if ok and np.isfinite(u) and np.isfinite(v):
            u, v = int(u * sx), int(v * sy)
            (cv2.drawMarker(img, (u, v), color, cv2.MARKER_TILTED_CROSS, 14, 2) if x
             else cv2.circle(img, (u, v), r, color, -1))

    for fr in range(Tv):
        mi = int(round((fr - lag) * ratio)) if lag >= 0 else int(round(fr * ratio)) - (-lag)
        tiles = []
        for c in cams:
            cam = calib[c]
            ret, img = caps[c].read()
            if not ret:
                img = np.zeros((H0, W0, 3), np.uint8)
            sx, sy = W0 / RES[0], H0 / RES[1]
            if 0 <= mi < tr.n_frames:
                for k in range(cupm.shape[1]):
                    draw(img, to_w0(cupm[mi, k]), (80, 220, 80), sx, sy)
                for k in range(headm.shape[1]):
                    draw(img, to_w0(headm[mi, k]), (230, 140, 60), sx, sy)
                draw(img, to_w0(head_c[mi]), (60, 60, 235), sx, sy, x=True)
                draw(img, to_w0(cup_c[mi]), (60, 220, 235), sx, sy, r=6)
            if 0 <= fr < len(cup_raw):
                draw(img, cup_raw[fr], (235, 60, 235), sx, sy, r=7, x=True)
            if cup_fused is not None and 0 <= fr < len(cup_fused):
                draw(img, cup_fused[fr], (60, 170, 255), sx, sy, r=6)
            if bio_head is not None and 0 <= fr < len(bio_head):
                draw(img, bio_head[fr], (255, 255, 255), sx, sy, r=6)
            tile = cv2.resize(img, (TILE_W, th))
            cv2.putText(tile, c, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            tiles.append(tile)
        # pad to full grid
        while len(tiles) < ncol * nrow:
            tiles.append(np.zeros((th, TILE_W, 3), np.uint8))
        rows = [np.hstack(tiles[r * ncol:(r + 1) * ncol]) for r in range(nrow)]
        grid = np.vstack(rows)
        cv2.putText(grid, f"{stem}  c3d={c3d}  rms={rms:.0f}mm  fr={fr}", (10, grid.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        wr.write(grid)
        if fr % 60 == 0:
            print(f"  frame {fr}/{Tv}", flush=True)
    wr.release()
    for cap in caps.values():
        cap.release()
    print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
