#!/bin/bash
# Render marker overlays for the worst dwell reps that have LOCAL footage, into
# experiments/drink_dwell/renders/. cup=green, head=blue, centroid=yellow.
cd /home/imove-laptop-01/object-tracking-master
OUT=experiments/drink_dwell/renders
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
  r="${REPS[$i]}"
  short=$(echo "$r" | sed 's/__clean3d_refill//' | sed 's/_P[0-9]*_drinking//')
  echo "[$((i+1))/${#REPS[@]}] rendering $short"
  python experiments/drink_study/analysis/overlay_markers.py "$r" \
    --out "$OUT/BAD_${short}.mp4" 2>&1 | tail -1
done
echo "ALL DONE"
