"""Metric capture for drinking-study training runs.

`train_with_metrics` wraps a YOLO-seg fine-tune and preserves, per run, the
evidence we need to later answer "how long should we train before the model
starts becoming useless":

    runs/<name>/metrics.json        full per-epoch curves (losses, mAP) from results.csv
    runs/<name>/eval_by_epoch.json  held-out recall/P_loose at each saved checkpoint
    runs/<name>/curves.png          val-loss vs held-out-recall over epochs
    runs/<name>/config.json         the exact config (for reproducibility)

`long_run=True` disables early-stop and checkpoints every `save_period` epochs
so we can chart the useful->useless turn; the default uses the val-loss plateau
early-stop (same pattern as pipeline_lib.loss_plateau_train).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # experiments/drink_study/ -> repo root
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline_lib import eval_gate


def _parse_results_csv(csv_path: Path) -> list[dict]:
    rows = list(csv.DictReader(csv_path.open()))
    out = []
    for r in rows:
        rr = {k.strip(): v.strip() for k, v in r.items()}

        def g(key):
            v = rr.get(key, "")
            return float(v) if v not in ("", "nan") else None

        vloss = sum(float(v) for k, v in rr.items()
                    if k.startswith("val/") and k.endswith("loss") and v not in ("", "nan"))
        tloss = sum(float(v) for k, v in rr.items()
                    if k.startswith("train/") and k.endswith("loss") and v not in ("", "nan"))
        out.append({
            "epoch": int(float(rr["epoch"])),
            "train_box": g("train/box_loss"), "train_seg": g("train/seg_loss"),
            "train_cls": g("train/cls_loss"), "train_loss_sum": tloss,
            "val_box": g("val/box_loss"), "val_seg": g("val/seg_loss"),
            "val_cls": g("val/cls_loss"), "val_loss_sum": vloss,
            "mAP50": g("metrics/mAP50(M)") if g("metrics/mAP50(M)") is not None
                     else g("metrics/mAP50(B)"),
            "mAP50_95": g("metrics/mAP50-95(M)") if g("metrics/mAP50-95(M)") is not None
                        else g("metrics/mAP50-95(B)"),
        })
    return out


def _f1(recall: float, precision: float) -> float:
    return 2 * recall * precision / (recall + precision) if (recall + precision) else 0.0


def _plot(per_epoch, eval_by_epoch, out_png: Path, title: str) -> None:
    """Two stacked panels sharing the epoch axis: training-side metrics on top,
    held-out test performance below, so the train->test relationship is visible."""
    eps = [r["epoch"] for r in per_epoch]
    fig, (axT, axB) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # --- top: training-side (losses + val mAP) ---
    axT.plot(eps, [r["train_loss_sum"] for r in per_epoch], color="tab:orange",
             lw=1.8, label="train loss")
    axT.plot(eps, [r["val_loss_sum"] for r in per_epoch], color="tab:red",
             lw=1.8, label="val loss")
    axT.set_ylabel("loss sum")
    axT.legend(loc="upper left")
    axT.grid(alpha=0.3)
    mt = axT.twinx()
    m3 = [r["mAP50_95"] for r in per_epoch]
    if any(m is not None for m in m3):
        mt.plot(eps, m3, color="tab:green", lw=1.5, ls="--", label="val mAP50-95")
        mt.set_ylabel("val mAP50-95", color="tab:green")
        mt.tick_params(axis="y", labelcolor="tab:green")
        mt.set_ylim(0, 1)
    axT.set_title(f"{title} — training metrics")

    # --- bottom: held-out test (recall + P_loose) ---
    if eval_by_epoch:
        ee = sorted((int(k), v) for k, v in eval_by_epoch.items())
        xs = [e for e, _ in ee]
        axB.plot(xs, [v["recall"] for _, v in ee], "o-", color="tab:blue",
                 lw=2, label="held-out recall")
        axB.plot(xs, [v["p_loose"] for _, v in ee], "s--", color="tab:purple",
                 lw=1.3, alpha=0.8, label="held-out P_loose")
        axB.plot(xs, [v.get("f1", _f1(v["recall"], v["p_loose"])) for _, v in ee],
                 "^-", color="black", lw=2, label="held-out F1")
        best = max(ee, key=lambda kv: kv[1].get("f1", _f1(kv[1]["recall"], kv[1]["p_loose"])))
        axB.axvline(best[0], color="gray", ls=":", lw=1)
        axB.annotate(f"best F1 @ep{best[0]}", (best[0], best[1]["recall"]),
                     textcoords="offset points", xytext=(5, -12), fontsize=8)
    axB.set_ylabel("held-out metric")
    axB.set_xlabel("epoch")
    axB.set_ylim(0, 1)
    axB.legend(loc="lower right")
    axB.grid(alpha=0.3)
    axB.set_title("held-out test performance")

    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def _plot_loss_vs_f1(per_epoch, eval_by_epoch, out_png: Path, title: str,
                     loss_key: str = "val_cls") -> None:
    """Held-out F1 vs a val loss component, one point per epoch (colored by epoch).

    Defaults to `val_cls` (the classification / objectness loss) because our F1
    is detection-presence based, not IoU-based: a box counts as a detection when
    its confidence clears the threshold, so the confidence/cls term is what
    drives recall (and, via clean scores, precision). box/dfl loss measure
    localization tightness, which this F1 does not reward. If F1 keeps rising as
    the loss falls, the loss is a good proxy; if points stack vertically (F1 flat
    while the loss still drops) the model is fitting the training subject, not
    gaining generalization.
    """
    loss_by_ep = {r["epoch"]: r.get(loss_key, r["val_loss_sum"]) for r in per_epoch}
    pts = []
    for k, v in eval_by_epoch.items():
        ep = int(k)
        if ep in loss_by_ep:
            f1 = v.get("f1", _f1(v["recall"], v["p_loose"]))
            pts.append((ep, loss_by_ep[ep], f1))
    if not pts:
        return
    pts.sort()
    eps = [p[0] for p in pts]
    loss = [p[1] for p in pts]
    f1 = [p[2] for p in pts]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(loss, f1, "-", color="lightgray", lw=1, zorder=1)   # epoch trajectory
    sc = ax.scatter(loss, f1, c=eps, cmap="viridis", s=60, zorder=2)
    for ep, lo, f in pts:
        if ep in (eps[0], eps[-1]) or f == max(f1):
            ax.annotate(f"ep{ep}", (lo, f), textcoords="offset points",
                        xytext=(6, 4), fontsize=8)
    ax.set_xlabel(loss_key)
    ax.set_ylabel("held-out F1")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.invert_xaxis()              # training progresses left as loss falls
    fig.colorbar(sc, ax=ax, label="epoch")
    ax.set_title(f"{title} — held-out F1 vs {loss_key}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def _plot_loss_vs_metrics(per_epoch, eval_by_epoch, out_png: Path, title: str,
                          loss_key: str = "val_cls") -> None:
    """recall / precision / F1 each vs a val loss component, side by side, so we
    can see whether they decouple from the loss differently (recall is driven by
    detection presence; precision by phantom suppression)."""
    loss_by_ep = {r["epoch"]: r.get(loss_key, r["val_loss_sum"]) for r in per_epoch}
    rows = []
    for k, v in eval_by_epoch.items():
        ep = int(k)
        if ep in loss_by_ep:
            rows.append((ep, loss_by_ep[ep], v["recall"], v["p_loose"],
                         v.get("f1", _f1(v["recall"], v["p_loose"]))))
    if not rows:
        return
    rows.sort()
    eps = [r[0] for r in rows]
    loss = [r[1] for r in rows]
    series = [("recall", 2), ("precision (P_loose)", 3), ("F1", 4)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, (label, idx) in zip(axes, series):
        y = [r[idx] for r in rows]
        ax.plot(loss, y, "-", color="lightgray", lw=1, zorder=1)
        sc = ax.scatter(loss, y, c=eps, cmap="viridis", s=45, zorder=2)
        ax.set_xlabel(loss_key)
        ax.set_title(label)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.invert_xaxis()
    axes[0].set_ylabel("held-out metric")
    fig.colorbar(sc, ax=axes, label="epoch", shrink=0.85)
    fig.suptitle(f"{title} — held-out metrics vs {loss_key}")
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def _plot_loss_vs_agreement(per_epoch, eval_by_epoch, out_png: Path, title: str,
                            loss_key: str = "val_cls") -> None:
    """Inter-camera agreement vs a val loss component, per epoch. tri_rate +
    mean_cams are 'higher=better' (left axes); median reproj px is
    'lower=better' (so we can see when geometric precision stops improving)."""
    loss_by_ep = {r["epoch"]: r.get(loss_key, r["val_loss_sum"]) for r in per_epoch}
    rows = [(int(k), loss_by_ep[int(k)], v.get("tri_rate"), v.get("cams"),
             v.get("median_px")) for k, v in eval_by_epoch.items()
            if int(k) in loss_by_ep and v.get("tri_rate") is not None]
    if not rows:
        return
    rows.sort()
    eps = [r[0] for r in rows]
    loss = [r[1] for r in rows]
    panels = [("tri_rate (coverage)", 2, (0, 1)),
              ("mean cams agreeing", 3, None),
              ("median reproj px (lower=better)", 4, None)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (label, idx, ylim) in zip(axes, panels):
        y = [r[idx] for r in rows]
        ax.plot(loss, y, "-", color="lightgray", lw=1, zorder=1)
        sc = ax.scatter(loss, y, c=eps, cmap="viridis", s=45, zorder=2)
        ax.set_xlabel(loss_key)
        ax.set_title(label)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)
        ax.invert_xaxis()
    fig.colorbar(sc, ax=axes, label="epoch", shrink=0.85)
    fig.suptitle(f"{title} — inter-camera agreement vs {loss_key}")
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def replot(run_dir: Path) -> None:
    """Regenerate all loss-vs-metric figures from saved metrics.json /
    eval_by_epoch.json (computes F1 if absent). No retraining."""
    run_dir = Path(run_dir)
    per_epoch = json.loads((run_dir / "metrics.json").read_text())
    eval_by_epoch = json.loads((run_dir / "eval_by_epoch.json").read_text())
    for v in eval_by_epoch.values():
        v.setdefault("f1", _f1(v["recall"], v["p_loose"]))
    (run_dir / "eval_by_epoch.json").write_text(json.dumps(eval_by_epoch, indent=2))
    name = run_dir.name
    _plot(per_epoch, eval_by_epoch, run_dir / "curves.png", name)
    _plot_loss_vs_f1(per_epoch, eval_by_epoch, run_dir / "loss_vs_f1.png", name)
    _plot_loss_vs_metrics(per_epoch, eval_by_epoch, run_dir / "loss_vs_metrics.png", name)
    if any(v.get("tri_rate") is not None for v in eval_by_epoch.values()):
        _plot_loss_vs_agreement(per_epoch, eval_by_epoch,
                                run_dir / "loss_vs_agreement.png", name)
    print(f"replotted figures in {run_dir}")


def train_with_metrics(data: Path, weights: str, project: str, name: str,
                       holdout_clips: Path, *, imgsz: int = 640, batch: int = 16,
                       eval_conf: float = 0.25, long_run: bool = False,
                       max_epochs: int = 120, save_period: int = 10,
                       plateau_patience: int = 6, plateau_delta: float = 0.01,
                       plateau_min_epoch: int = 10, keep_checkpoints: bool = True,
                       agr_participants: list[str] | None = None, agr_reps: int = 1,
                       agr_hand: str = "right", config: dict | None = None) -> tuple[Path, dict]:
    from ultralytics import YOLO
    model = YOLO(weights)

    if not long_run:                       # val-loss plateau early-stop
        st = {"best": 1e9, "best_ep": 0}

        def cb(tr):
            ep = int(getattr(tr, "epoch", 0)) + 1
            met = getattr(tr, "metrics", {}) or {}
            vl = sum(float(v) for k, v in met.items()
                     if k.startswith("val/") and k.endswith("_loss"))
            if vl and vl < st["best"] - plateau_delta:
                st["best"], st["best_ep"] = vl, ep
            if ep >= plateau_min_epoch and (ep - st["best_ep"]) >= plateau_patience:
                print(f"[plateau] stopping at epoch {ep}")
                tr.stop = True

        model.add_callback("on_fit_epoch_end", cb)

    model.train(data=str(data), epochs=max_epochs, imgsz=imgsz, batch=batch,
                project=project, name=name, exist_ok=True,
                save_period=save_period if long_run else -1)

    # Use the trainer's actual output dir (robust to any ultralytics renaming).
    run_dir = Path(getattr(model.trainer, "save_dir", Path(project) / name))

    # 1) full per-epoch curves
    per_epoch = _parse_results_csv(run_dir / "results.csv")
    (run_dir / "metrics.json").write_text(json.dumps(per_epoch, indent=2))

    # 2) held-out eval at each saved checkpoint (the "becomes useless" curve)
    wdir = run_dir / "weights"
    ckpts: dict[int, Path] = {}
    for p in sorted(wdir.glob("epoch*.pt")):
        digits = "".join(c for c in p.stem if c.isdigit())
        if digits:
            ckpts[int(digits)] = p
    if (wdir / "last.pt").exists() and per_epoch:
        ckpts[per_epoch[-1]["epoch"]] = wdir / "last.pt"
    if agr_participants:
        from agreement import agreement_eval
    eval_by_epoch = {}
    for ep, p in sorted(ckpts.items()):
        r = eval_gate(str(p), Path(holdout_clips), eval_conf)
        rec, pl = r.metrics["overall_recall"], r.metrics["overall_p_loose"]
        entry = {"recall": rec, "p_loose": pl, "f1": _f1(rec, pl)}
        if agr_participants:                      # inter-camera agreement this epoch
            a = agreement_eval(str(p), agr_participants, agr_reps, classes=None,
                               hand=agr_hand)
            entry.update({"tri_rate": a.get("tri_rate"),
                          "median_px": a.get("median_reproj_px"),
                          "cams": a.get("mean_cams_agreeing")})
        eval_by_epoch[ep] = entry
        print(f"  ckpt ep{ep}: recall={rec:.3f} f1={_f1(rec, pl):.3f}"
              + (f" tri={entry.get('tri_rate')} px={entry.get('median_px')}"
                 if agr_participants else ""), flush=True)
    (run_dir / "eval_by_epoch.json").write_text(json.dumps(eval_by_epoch, indent=2))

    # Keep the per-epoch checkpoints by default: we may want to score them with
    # NEW per-epoch metrics later (e.g. inter-camera agreement). Only purge once
    # the metric suite is final and everything has been computed.
    if not keep_checkpoints:
        for p in wdir.glob("epoch*.pt"):
            p.unlink()

    if config is not None:
        (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    _plot(per_epoch, eval_by_epoch, run_dir / "curves.png", name)
    _plot_loss_vs_f1(per_epoch, eval_by_epoch, run_dir / "loss_vs_f1.png", name)
    _plot_loss_vs_metrics(per_epoch, eval_by_epoch, run_dir / "loss_vs_metrics.png", name)
    if agr_participants:
        _plot_loss_vs_agreement(per_epoch, eval_by_epoch,
                                run_dir / "loss_vs_agreement.png", name)

    return wdir / "best.pt", {"per_epoch": per_epoch, "eval_by_epoch": eval_by_epoch}


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) != 2:
        raise SystemExit("usage: python metrics.py <run_dir>   # regenerate plots")
    replot(Path(_sys.argv[1]))
