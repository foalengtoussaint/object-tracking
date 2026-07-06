"""Build a disputed-frame STRIP for slide 9: sampled frames across the dwell, each
labelled by which segmentation includes it (TRUE mocap-gate vs HYBRID). The hybrid-only
frames (green) show the cup ALREADY AT THE MOUTH -- i.e. the speed-gate truth is late.

    python experiments/drink_study/build_dispute_strip.py
Writes slides/fig13_dispute_strip.png
"""
from __future__ import annotations
import sys as _s, pathlib as _p  # drink_study lib path shim
for _q in _p.Path(__file__).resolve().parents:
    if (_q / 'lib' / 'segment_cup_only.py').exists():
        _s.path.insert(0, str(_q / 'lib')); _s.path.insert(0, str(_q)); _s.path.insert(0, str(_q.parents[1])); break
import sys
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_decode

CLIPS = Path("clips")
OUT = Path("experiments/drink_study/slides"); OUT.mkdir(exist_ok=True)

# rep: (stem, cam, TRUE span, HYBRID span). P23 cam4 = clear side view of cup->mouth.
STEM = "P23_drinking_right_20240716_151359"
CAM = "4"
TRUE = (164, 185)
HYB = (148, 199)

# sample 6 frames spanning: hybrid-onset (disputed) -> true core -> true-offset -> hybrid-offset (disputed)
FRAMES = [150, 160, 172, 185, 193, 198]


def classify(fi):
    inh = HYB[0] <= fi < HYB[1]
    int_ = TRUE[0] <= fi < TRUE[1]
    if int_:
        return "TRUE + HYBRID", (90, 200, 90)      # both agree = green-ish (drinking)
    if inh:
        return "HYBRID only", (60, 220, 60)         # disputed = bright green (cup at mouth, truth misses)
    return "neither", (150, 150, 150)


def main():
    p = STEM.split("_")[0]
    video = CLIPS / p / f"{STEM}.{CAM}.mp4"
    W, H, nv, _ = gpu_decode.dims(video)
    want = set(FRAMES)
    got = {}
    for fi, img in enumerate(gpu_decode.frames(video)):
        if fi in want:
            got[fi] = img.copy()
        if fi > max(FRAMES):
            break
    # crop to the head/hand action region (left-center of this cam4 view), then tile
    # crop box as fractions of full frame: the participant's head+cup sit left-of-centre
    cx0, cx1 = int(0.30 * W), int(0.62 * W)
    cy0, cy1 = int(0.05 * H), int(0.62 * H)
    tiles = []
    for fi in FRAMES:
        img = got[fi][cy0:cy1, cx0:cx1]
        h, w = img.shape[:2]
        lab, col = classify(fi)
        bar = 46
        canvas = np.full((h + bar, w, 3), 255, np.uint8)
        canvas[bar:] = img
        cv2.rectangle(canvas, (0, 0), (w, bar), col, -1)
        t = f"f{fi}  t={fi/60:.2f}s"
        cv2.putText(canvas, t, (8, 20), 0, 0.6, (20, 20, 20), 2)
        cv2.putText(canvas, lab, (8, 40), 0, 0.5, (20, 20, 20), 1)
        # colored border
        canvas = cv2.copyMakeBorder(canvas, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=col)
        tiles.append(canvas)
    strip = np.hstack(tiles)
    outp = OUT / "fig13_dispute_strip.png"
    cv2.imwrite(str(outp), strip)
    print("wrote", outp, strip.shape,
          f"\n  TRUE{TRUE} HYBRID{HYB} -- disputed frames {[f for f in FRAMES if not (TRUE[0]<=f<TRUE[1])]}")


if __name__ == "__main__":
    main()
