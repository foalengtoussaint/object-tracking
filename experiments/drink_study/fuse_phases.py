"""Confidence-weighted cup+pose fusion for drink-task phase segmentation.

Unifies the two trackers onto ONE physical signal -- distance to the mouth
(head joint 67) -- measured two ways, each with its own per-frame confidence:

  cup -> mouth : rescued refill cup track   vs head67   (conf = cup tracker conf)
  wrist-> mouth: dominant wrist (85/77)      vs head67   (conf = pose joint conf)

The drink dwell is "near the mouth + slow". We compute that evidence from each
source, weight by that source's per-frame confidence, and fuse:

      E = (w_cup * e_cup + w_pose * e_pose) / (w_cup + w_pose)

Because the cup confidence collapses while it's occluded at the mouth and the
pose confidence holds there (and vice-versa during fast transport), the handoff
is automatic -- no hard winner rule. Hysteresis on E -> the drinking interval.
Transport/rest boundaries come from the cup motion window (clean end-effector).

This REPLACES the rest-relative "near peak displacement" gate with cup-mouth
distance (anatomically meaningful, self-normalised per trial), which is what the
container segmenter calls the drink-trigger signal.

    python experiments/drink_study/fuse_phases.py            # all 4 cached biomech trials
    python experiments/drink_study/fuse_phases.py P23_P23_drinking_right_20240716_151359
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import segment_cup_only as sc

CACHE = Path("experiments/drink_study/cache")
TRACK = CACHE / "track3d_clean3d_refill"
FPS = 60.0
J_HEAD, J_LWRIST, J_RWRIST = 67, 77, 85

# fusion params
NEAR_PAD       = 80.0    # "near mouth" = within (min_dist + pad) mm of closest approach
DRINK_SPEED    = 120.0   # mm/s, source is slow (same scale as cup segmenter)
SOFTNESS_MM    = 60.0    # logistic softness for the near-mouth membership
SPD_SOFT       = 80.0    # logistic softness for the slow membership
E_ON, E_OFF    = 0.55, 0.40   # hysteresis on fused evidence
MIN_DRINK_FR   = max(int(0.20 * FPS), 5)


def track_stem(trial):
    """Normalise a biomech-npz trial name to the double-P track/rescue stem.
    biomech_P01_drinking_right_... -> P01_P01_drinking_right_...; pass through if already doubled."""
    p = trial.split("_")[0]
    return trial if trial.startswith(f"{p}_{p}_") else f"{p}_{trial}"


def biomech_npz(trial):
    """Find the cached biomech npz whether the name is single- or double-P.
    Accepts P01_drinking_right_... or P01_P01_drinking_right_... interchangeably."""
    p = trial.split("_")[0]
    single = trial[len(p) + 1:] if trial.startswith(f"{p}_{p}_") else trial
    for name in (trial, single, f"{p}_{single}"):
        f = CACHE / f"biomech_{name}.npz"
        if f.exists():
            return f
    raise FileNotFoundError(f"no biomech npz for {trial} (tried single/double-P)")


def _smooth(x, w=7):
    return sc._median_smooth(np.asarray(x, float), w)


def _logi(x, soft):
    return 1.0 / (1.0 + np.exp(np.clip(x / soft, -30, 30)))   # ->1 when x<<0, ->0 when x>>0


def _logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(np.clip(-x, -30, 30)))


def _interp_nan(a):
    a = a.copy(); v = np.isfinite(a); idx = np.flatnonzero(v)
    if len(idx) < 2:
        return a
    a[:] = np.interp(np.arange(len(a)), idx, a[idx])
    return a


def cup_track_conf(trial):
    """Rescued cup xyz (T,3) + per-frame cup confidence in [0,1].
    conf = min(kept_cams/3,1) * tightness(px); 0 on interpolated frames (no kept cams)."""
    ts = track_stem(trial)
    d = json.loads((TRACK / f"{ts}__clean3d_refill.json").read_text())["frames"]
    rf = CACHE / f"rescue2cam_{ts}_full.json"
    if rf.exists():
        rts = json.loads(rf.read_text())["rts_new"]
        rescued = {int(fr) for fr, *_ in json.loads(rf.read_text())["rescued"]}
    else:
        rts = [f["rts"] for f in d]; rescued = set()
    T = len(d)
    xyz = np.array([p if p else [np.nan] * 3 for p in rts], float)
    conf = np.zeros(T)
    for i, f in enumerate(d):
        nk = len(f.get("kept", [])); px = f.get("median_px")
        if nk >= 2 and px is not None:
            tight = float(np.clip(1.0 - (px - 3.0) / 25.0, 0.15, 1.0))   # 3px->1, 28px->0.15
            conf[i] = min(nk / 3.0, 1.0) * tight
        elif i in rescued:
            conf[i] = 0.35                                                # 2-cam rescue = medium-low
    return xyz, conf, T


def pose_wrist_head(trial):
    """Dominant wrist xyz, head xyz, per-frame pose confidence (wrist*head joint conf).

    Dropped frames (MeTRAbs zero-fill: conf~0 / position at the origin) are set to
    NaN so the downstream distance/speed signals INTERPOLATE over them instead of
    reading a spurious (0,0,0) -> a single dropped frame otherwise poisons
    distance-to-mouth and produces a huge phantom speed spike (the P24 'head
    moving 137 m/s' bug — it was 4 zero-filled frames, not real motion)."""
    b = np.load(biomech_npz(trial), allow_pickle=True)
    kp = b["keypoints3d"]                          # (T,87,4)
    side = str(b["side"])
    jw = J_RWRIST if side.startswith("r") else J_LWRIST
    wrist = kp[:, jw, :3].astype(float); head = kp[:, J_HEAD, :3].astype(float)
    wconf = kp[:, jw, 3].astype(float); hconf = kp[:, J_HEAD, 3].astype(float)
    # drop frames: near-zero conf OR position at the origin (zero-fill sentinel)
    wbad = (wconf < 0.05) | (np.abs(wrist).sum(1) < 1e-6)
    hbad = (hconf < 0.05) | (np.abs(head).sum(1) < 1e-6)
    wrist[wbad] = np.nan; head[hbad] = np.nan
    pose_conf = np.sqrt(np.clip(wconf, 0, 1) * np.clip(hconf, 0, 1))   # geo-mean, both must be good
    return wrist, head, pose_conf, side


def evidence(dist_to_mouth, speed, conf):
    """Per-frame drink evidence in [0,1].

    Distance-to-mouth IS the drink signal, so NEAR is the evidence (self-normalised
    to this source's own closest approach -> anatomy/skeleton-free). SLOW is only a
    GATE: a fast-moving frame cannot be a dwell, so it caps the evidence, but a slow
    frame does not dilute a clear near-mouth reading (option A).

        e = near * gate(speed),  gate = 1 if clearly slow, ramps to 0 when fast
    """
    d = _interp_nan(dist_to_mouth)
    dmin = np.nanpercentile(d, 5)                  # robust 'closest approach'
    near = _logi(d - (dmin + NEAR_PAD), SOFTNESS_MM)
    # slow as a gate: full credit while slow, only bites above DRINK_SPEED
    gate = _logi(_smooth(speed) - DRINK_SPEED, SPD_SOFT)
    gate = np.clip(gate * 1.6, 0.0, 1.0)           # widen the 'slow' plateau (gate, not dilutor)
    e = near * gate
    return e, d, dmin


def hysteresis(E, on=E_ON, off=E_OFF, min_run=MIN_DRINK_FR):
    T = len(E); m = np.zeros(T, bool); state = False
    for i in range(T):
        state = (E[i] >= on) if not state else (E[i] >= off)
        m[i] = state
    runs = sc._runs(m)
    return [(s, e) for s, e in runs if e - s >= min_run]


def fuse(trial):
    cup_xyz, w_cup, Tc = cup_track_conf(trial)
    wrist, head, w_pose, side = pose_wrist_head(trial)
    T = min(Tc, len(head))
    cup_xyz, w_cup = cup_xyz[:T], w_cup[:T]
    wrist, head, w_pose = wrist[:T], head[:T], w_pose[:T]

    # interpolate over dropped (NaN'd) cup/pose frames so distance & speed are
    # computed on continuous positions -- a single zero-fill frame otherwise
    # poisons distance-to-mouth and spikes the speed (the P24 head-drop bug)
    cup_i = _interp_nan_xyz(cup_xyz)
    wrist_i = _interp_nan_xyz(wrist)
    head_i = _interp_nan_xyz(head)

    # one physical signal, two sources: distance to mouth
    cup_mouth = np.linalg.norm(cup_i - head_i, axis=1)
    wrist_mouth = np.linalg.norm(wrist_i - head_i, axis=1)
    cup_spd = np.r_[0, np.linalg.norm(np.diff(cup_i, axis=0), axis=1)] * FPS
    wrist_spd = np.r_[0, np.linalg.norm(np.diff(wrist_i, axis=0), axis=1)] * FPS

    e_cup, cup_mouth_i, _ = evidence(cup_mouth, cup_spd, w_cup)
    e_pose, wrist_mouth_i, _ = evidence(wrist_mouth, wrist_spd, w_pose)

    # confidence-weighted fusion in LOG-ODDS (the "kalman using how sure each is"):
    # each source votes in log-odds, scaled by its confidence; votes ADD, so two
    # confident sources both saying "drinking" reinforce -> E exceeds either alone
    # (true agreement), while a low-confidence source barely moves the needle.
    # A neutral prior (logit 0) keeps E~0.5 only when nobody is confident.
    L = w_cup * _logit(e_cup) + w_pose * _logit(e_pose)
    E = _sigmoid(L)
    E = _smooth(E, 5)
    drink_runs = hysteresis(E)
    drink = (drink_runs[0][0], drink_runs[-1][1]) if drink_runs else None

    # transport/rest window from the cup motion (clean end-effector) -- reuse cup segmenter
    seg = sc.segment_cup_only(_interp_nan_xyz(cup_xyz))
    grasp = seg["grasp"]
    onset, offset = grasp
    rec = build_phases(T, onset, offset, drink)

    return dict(trial=trial, T=T, side=side,
                w_cup=w_cup, w_pose=w_pose, e_cup=e_cup, e_pose=e_pose, E=E,
                cup_mouth=cup_mouth_i, wrist_mouth=wrist_mouth_i,
                cup_xyz=cup_xyz, wrist_xyz=wrist, head_xyz=head,
                drink=drink, drink_runs=drink_runs, onset=onset, offset=offset,
                reconciled=rec)


def _interp_nan_xyz(xyz):
    out = xyz.copy()
    for a in range(3):
        out[:, a] = _interp_nan(xyz[:, a])
    return out


def build_phases(T, onset, offset, drink):
    rec = []
    onset = onset or 0; offset = offset or T
    if onset > 0:
        rec.append(("rest_pre", 0, onset))
    if drink and drink[0] > onset:
        rec.append(("forward_transport", onset, drink[0]))
        d1 = min(drink[1], offset) if offset > drink[0] else drink[1]
        rec.append(("drinking", drink[0], d1))
        if offset > d1:
            rec.append(("back_transport", d1, offset))
    else:
        rec.append(("forward_transport", onset, offset))
    if offset < T:
        rec.append(("rest_post", offset, T))
    return rec


def cup_only_drink(trial):
    xyz, _, _ = cup_track_conf(trial)
    r = sc.segment_cup_only(_interp_nan_xyz(xyz))
    dr = [(s, e) for n, s, e in r["intervals"] if n == "drinking"]
    return (dr[0][0], dr[-1][1]) if dr else None


def pose_only_drink(trial):
    b = np.load(biomech_npz(trial), allow_pickle=True)
    if "phase_intervals" not in b.files:
        return None                          # standalone pose npz (no container segmenter run)
    dr = [(int(s), int(e)) for n, s, e in b["phase_intervals"] if n == "drinking"]
    return (dr[0][0], dr[-1][1]) if dr else None


def plot(res, out):
    T = res["T"]; t = np.arange(T) / FPS
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(t, res["cup_mouth"], color="#1f77b4", label="cup→mouth dist (mm)")
    ax[0].plot(t, res["wrist_mouth"], color="#d62728", label="wrist→mouth dist (mm)")
    ax[0].set_ylabel("dist to mouth (mm)"); ax[0].legend(fontsize=8, loc="upper right")
    ax[0].set_ylim(0, min(900, np.nanmax([res["cup_mouth"].max(), 900])))
    ax[1].plot(t, res["w_cup"], color="#1f77b4", label="cup confidence")
    ax[1].plot(t, res["w_pose"], color="#d62728", label="pose confidence")
    ax[1].set_ylabel("confidence"); ax[1].legend(fontsize=8, loc="upper right"); ax[1].set_ylim(0, 1)
    ax[2].plot(t, res["e_cup"], color="#1f77b4", ls=":", label="e_cup")
    ax[2].plot(t, res["e_pose"], color="#d62728", ls=":", label="e_pose")
    ax[2].plot(t, res["E"], color="k", lw=2, label="fused E")
    ax[2].axhline(E_ON, color="gray", ls="--", lw=0.7); ax[2].axhline(E_OFF, color="gray", ls="--", lw=0.7)
    ax[2].set_ylabel("drink evidence"); ax[2].legend(fontsize=8, loc="upper right"); ax[2].set_ylim(0, 1)
    ax[2].set_xlabel("time (s)")
    for a in ax:
        if res["drink"]:
            a.axvspan(res["drink"][0] / FPS, res["drink"][1] / FPS, color="#e8554c", alpha=0.18, lw=0)
    co = cup_only_drink(res["trial"]); po = pose_only_drink(res["trial"])
    ax[0].set_title(f"{res['trial']}\nFUSED drink: {_f(res['drink'])}   "
                    f"cup-only: {_f(co)}   pose-only: {_f(po)}", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print(f"  wrote {out}", flush=True)


def _f(iv):
    return f"{iv[0]/FPS:.2f}-{iv[1]/FPS:.2f}s" if iv else "NONE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trials", nargs="*")
    args = ap.parse_args()
    trials = args.trials or [Path(f).name[len("biomech_"):-4]
                             for f in sorted(glob.glob(str(CACHE / "biomech_*.npz")))]
    rows = []
    for t in trials:
        res = fuse(t)
        co = cup_only_drink(t); po = pose_only_drink(t)
        rows.append((t, res["drink"], co, po))
        print(f"{t}")
        print(f"   FUSED drink : {_f(res['drink'])}")
        print(f"   cup-only    : {_f(co)}")
        print(f"   pose-only   : {_f(po)}")
        plot(res, CACHE / f"fuse_{t}.png")
        print()
    print(f"=== {len(rows)} trials ===")
    print(f"  drink found:  fused {sum(r[1] is not None for r in rows)}/{len(rows)}  "
          f"cup-only {sum(r[2] is not None for r in rows)}/{len(rows)}  "
          f"pose-only {sum(r[3] is not None for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
