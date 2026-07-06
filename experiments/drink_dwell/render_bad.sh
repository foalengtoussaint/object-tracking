#!/bin/bash
# Render dwell-bar marker overlays for the worst dwell reps that have LOCAL footage, into
# experiments/drink_dwell/renders/. Uses overlay.py (features.mocap_to_w0 = the SAME alignment
# the model uses) + truth/proxy21/base17 dwell bars. Run plot.py first for the model spans.
cd /home/imove-laptop-01/object-tracking-master
REPS=(
  "P07_P07_drinking_left_20240124_142839__clean3d_refill"
  "P10_P10_drinking_right_20240202_153316__clean3d_refill"
  "P10_P10_drinking_right_20240202_152807__clean3d_refill"
  "P16_P16_drinking_right_20240306_105401__clean3d_refill"
  "P07_P07_drinking_left_20240124_142730__clean3d_refill"
  "P19_P19_drinking_right_20240312_115839__clean3d_refill"
  "P19_P19_drinking_right_20240312_115809__clean3d_refill"
  "P19_P19_drinking_right_20240312_115919__clean3d_refill"
  "P16_P16_drinking_right_20240306_105546__clean3d_refill"
)
for i in "${!REPS[@]}"; do
  echo "[$((i+1))/${#REPS[@]}] ${REPS[$i]}"
  python experiments/drink_dwell/overlay.py "${REPS[$i]}" 2>&1 | grep -E "Kabsch rms|wrote"
done
echo "ALL DONE"
