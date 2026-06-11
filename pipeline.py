"""Gated auto-pipeline: take a directory of clips + an object name and drive the
whole new-object tracking pipeline end to end, surfacing every decision.

Spine (shared by both entry paths):
    round-1 labels  ->  finetune #1  ->  dense KF self-label  ->  finetune #2  ->  eval
The only difference between paths is where round-1 labels come from:
    AUTO  = COCO / YOLO-World teacher          (object is in or near COCO)
    SAM   = manual click seeds via sam_label_server  (novel object)

Gates between stages auto-proceed when healthy, take the recommended default on
minor flags (logged to the manifest), and HALT only on genuine forks:
    * label source (AUTO vs SAM)            [stage 0 probe]
    * "labels look good — train?"           [before each finetune]
    * patch blind cameras vs proceed        [stage 4 coverage]

Examples
--------
    # Full auto-driven run on a fresh recording
    python pipeline.py --clips data/clips/stone_p01 --object "wooden stone"

    # Reuse an existing SAM seed dataset (skip manual clicking)
    python pipeline.py --clips data/clips/cube_p01_all --object cube \\
        --label-source sam --seed-dataset data/datasets/cube_p01_seed \\
        --eval-clips data/clips/cube_p02_10cm --run cube_p01
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import pipeline_lib as L
from pseudo_label import build_dataset, resolve_classes


def _gate(run_dir, m, result, *, fork_action=None, artifacts=None):
    """Print a gate, halt on FORK (return chosen index), record to manifest."""
    if result.status == L.FLAG_FORK:
        choice = L.ask_fork(result)
        decision = result.options[choice - 1]
        L.record_stage(run_dir, m, result.stage, result, decision, artifacts)
        return choice
    L.print_gate(result)
    decision = "auto-proceed" if result.status == L.PASS else "auto-default (logged)"
    L.record_stage(run_dir, m, result.stage, result, decision, artifacts)
    return None


def _review_video_gate(run_dir, m, dataset, stage, viz_dir):
    """Render the label-review video and halt go/no-go before a finetune."""
    out = viz_dir / f"{stage}.mp4"
    print(f"\nrendering label-review video for {dataset.name} ...")
    L.render_label_video(dataset, out)
    r = L.GateResult(
        stage=stage, status=L.FLAG_FORK,
        summary="Review the pseudo-labels before training. The masks should "
                "track the object across the clip.",
        artifacts=[str(out)],
        recommendation="watch the video, then confirm",
        options=["labels look good — train", "labels bad — abort"])
    choice = L.ask_fork(r)
    L.record_stage(run_dir, m, stage, r, r.options[choice - 1], {stage: out})
    if choice == 2:
        sys.exit("aborted at label review — fix labeling and re-run.")


def _wait_for_sam(dataset: Path, port: int):
    """Launch the SAM labeler subprocess, wait for the user, then stop it."""
    proc = subprocess.Popen(
        [sys.executable, "sam_label_server.py", "--dataset", str(dataset),
         "--port", str(port)])
    try:
        print(f"\n  SAM labeler starting -> open http://localhost:{port}")
        print("  Label the images, then press Enter here when done.")
        input("  [Enter when labeling is complete] ")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _finetune(dataset, run_dir, m, project, name, thr, stage_label):
    """val split + loss-plateau train + convergence gate. Returns best.pt."""
    L.make_val_split(dataset, thr["val_frac"])
    best = L.loss_plateau_train(dataset / "data.yaml", thr["student_weights"],
                                project, name, thr)
    ok, cm = L.converged(project, name)
    r = L.GateResult(stage=stage_label, status=L.PASS if ok else L.FLAG_AUTO,
                     summary=("training converged." if ok else
                              "weak convergence — keeping best.pt anyway."),
                     metrics=cm, artifacts=[str(best)])
    _gate(run_dir, m, r, artifacts={f"{stage_label}_weights": best})
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=Path, required=True, help="dir of *.mp4 clips")
    ap.add_argument("--object", required=True, help="object name (YOLO-World prompt / label)")
    ap.add_argument("--run", default=None, help="run name (default: <object>)")
    ap.add_argument("--label-source", choices=["probe", "auto", "sam"], default="probe")
    ap.add_argument("--seed-dataset", type=Path, default=None,
                    help="existing SAM seed dataset to reuse (skips manual clicking)")
    ap.add_argument("--eval-clips", type=Path, default=None,
                    help="held-out clips dir for the final eval (default: --clips)")
    ap.add_argument("--classes", default="cup-like",
                    help="AUTO path COCO class filter (cup-like|all|ints)")
    ap.add_argument("--config", type=Path, default=None, help="json threshold overrides")
    ap.add_argument("--sam-port", type=int, default=5008)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    thr = dict(L.DEFAULTS)
    if args.config:
        thr.update(json.loads(args.config.read_text()))

    run = args.run or args.object.replace(" ", "_")
    run_dir = Path("data/runs/pipeline") / run
    ds_root = Path("data/datasets")
    viz_dir = run_dir / "viz"; viz_dir.mkdir(parents=True, exist_ok=True)
    m = L.load_manifest(run_dir) if args.resume else {"stages": {}, "config": {}, "artifacts": {}}
    m["config"] = {"clips": str(args.clips), "object": args.object, "run": run, **thr}
    L.save_manifest(run_dir, m)

    # ---- Stage 0: label-source decision (FORK) ----------------------------- #
    source = args.label_source
    if source == "probe":
        r, extra = L.probe_auto_detect(args.clips, args.object, run_dir / "probe", thr)
        choice = _gate(run_dir, m, r)
        source = {1: "sam", 2: "auto", 3: "abort"}[choice]
        if source == "abort":
            sys.exit("aborted at probe.")

    # ---- Stage 1: round-1 labels ------------------------------------------ #
    if source == "auto":
        ds1 = ds_root / f"{run}_gen1"
        print(f"\n[round-1 AUTO] COCO teacher -> {ds1}")
        stats = build_dataset(args.clips, ds1, thr["coco_weights"], thr["label_conf"],
                              resolve_classes(args.classes), class_name=args.object)
    else:  # sam
        if args.seed_dataset:
            ds1 = args.seed_dataset
            print(f"\n[round-1 SAM] reusing seed dataset {ds1}")
        else:
            ds1 = ds_root / f"{run}_seed"
            n = L.extract_seed_frames(args.clips, ds1, thr["seed_frames_per_clip"])
            (ds1 / "data.yaml").write_text(
                f"path: {ds1.resolve()}\ntrain: images/train\nval: images/train\n"
                f"nc: 1\nnames: [{args.object}]\n")
            print(f"\n[round-1 SAM] extracted {n} frames for labeling")
            _wait_for_sam(ds1, args.sam_port)

    # ---- Stage 2: label QA (auto) ----------------------------------------- #
    _gate(run_dir, m, L.qa_labels(ds1))

    # ---- Stage 1.9 + 3: review video FORK, then finetune #1 --------------- #
    _review_video_gate(run_dir, m, ds1, "3a_review_round1", viz_dir)
    student1 = _finetune(ds1, run_dir, m, str(run_dir.resolve() / "segment"),
                         "student1", thr, "3_finetune1")

    # ---- Stage 4: dense self-label + coverage gate (FORK if blind) -------- #
    while True:
        ds_dense = ds_root / f"{run}_dense"
        print(f"\n[dense] teacher=student1 -> {ds_dense}")
        stats = build_dataset(args.clips, ds_dense, str(student1), thr["label_conf"],
                              [0], class_name=args.object)
        if stats["total"] == 0:
            sys.exit("dense labeling produced 0 labels — the teacher (student1) "
                     "detected nothing. It is too weak or the object/clips are "
                     "wrong; aborting before finetune #2.")
        cov = L.coverage_report(stats["per_clip"], 0.0, thr)
        choice = _gate(run_dir, m, cov)
        if cov.status == L.FLAG_FORK and choice == 1:   # patch blind cams via SAM
            blind = cov.metrics["blind_cams"]
            patch_clips = run_dir / "patch_clips"; patch_clips.mkdir(exist_ok=True)
            for clip in sorted(args.clips.glob("*.mp4")):
                if L.camera_id(clip.stem) in blind:
                    (patch_clips / clip.name).unlink(missing_ok=True)
                    (patch_clips / clip.name).symlink_to(clip.resolve())
            print(f"  patching blind cameras {blind}: label a few seeds from them")
            L.extract_seed_frames(patch_clips, ds1, thr["seed_frames_per_clip"])
            _wait_for_sam(ds1, args.sam_port)
            student1 = _finetune(ds1, run_dir, m, str(run_dir.resolve() / "segment"),
                                 "student1_patched", thr, "3_finetune1_patched")
            continue   # re-dense with the patched teacher
        break

    # ---- Stage 4.9 + 5: review video FORK, then finetune #2 --------------- #
    _review_video_gate(run_dir, m, ds_dense, "5a_review_dense", viz_dir)
    student2 = _finetune(ds_dense, run_dir, m, str(run_dir.resolve() / "segment"),
                         "student2", thr, "5_finetune2")

    # ---- Stage 6: final eval (auto) + experiment doc ---------------------- #
    eval_clips = args.eval_clips or args.clips
    print(f"\n[eval] {student2} on {eval_clips}")
    ev = L.eval_gate(str(student2), eval_clips, thr["eval_conf"])
    _gate(run_dir, m, ev, artifacts={"final_weights": student2})

    _write_experiment_doc(run, args.object, student2, m, ev)
    print(f"\nDONE. final student: {student2}")
    print(f"manifest (all decisions): {L.manifest_path(run_dir)}")


def _write_experiment_doc(run, obj, weights, manifest, ev):
    out = Path("experiments") / f"{date.today().isoformat()}_{run}.md"
    lines = [f"# {run} — {obj} (auto-pipeline)", "",
             f"- date: {date.today().isoformat()}",
             f"- final weights: `{weights}`",
             f"- final eval: recall {ev.metrics.get('overall_recall')}, "
             f"P_loose {ev.metrics.get('overall_p_loose')}", "",
             "## Per-camera eval", "",
             "| cam | recall | P_loose |", "|---|---|---|"]
    for cam, d in ev.metrics.get("per_camera", {}).items():
        lines.append(f"| {cam} | {d['recall']} | {d['p_loose']} |")
    lines += ["", "## Decision log", ""]
    for stage, rec in manifest["stages"].items():
        lines.append(f"- **{stage}** [{rec['status']}]: {rec.get('decision')} "
                     f"— {rec.get('summary', '')}")
    out.write_text("\n".join(lines))
    print(f"experiment doc: {out}")


if __name__ == "__main__":
    main()
