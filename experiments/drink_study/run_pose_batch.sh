#!/usr/bin/env bash
# Batch standalone pose (MeTRAbs->triangulate->npz) for the no-drink trials, in
# isr-containers-dev-1, bypassing DataJoint. Stages each trial's 10 cam clips +
# calib into the DATA scratch, runs run_pose.py, copies the npz into our cache.
# Idempotent: skips trials whose npz already exists.
set -u
CLIPS_ROOT=/home/imove/Documents/clips
CALIB_ROOT=/home/imove/Documents/object_tracking/data/calib
SC=/home/imove/Documents/iMOVE/DEV/isr-supplementary/DATA/ot_pose_scratch
CACHE=/home/imove/Documents/object_tracking/experiments/drink_study/cache
LIST="${1:-/tmp/worst20.txt}"

mkdir -p "$SC/clips" "$SC/calib" "$SC/out"
n=0; tot=$(wc -l < "$LIST")
while read -r trial; do
  [ -z "$trial" ] && continue
  n=$((n+1))
  if [ -f "$CACHE/biomech_${trial}.npz" ]; then
    echo "[$n/$tot] $trial — npz exists, skip"; continue
  fi
  p="${trial%%_*}"
  # Clips on disk are named "<P>_drinking_<side>_<ts>.<cam>.mp4". Older lists
  # passed the trial WITHOUT the leading "<P>_"; newer lists pass the full
  # name. Resolve the on-disk stem robustly: prefer the full trial name, fall
  # back to the P-stripped form.
  fname="$trial"
  [[ "$trial" == "${p}_"* ]] || fname="${p}_${trial}"
  side=right; [[ "$trial" == *drinking_left* ]] && side=left
  echo "[$n/$tot] $trial (side=$side) — staging"
  rm -f "$SC/clips/"*.mp4 "$SC/calib/calibration.toml"
  cp "$CLIPS_ROOT/$p/$fname".*.mp4 "$SC/clips/" 2>/dev/null
  cp "$CALIB_ROOT/$p/calibration.toml" "$SC/calib/calibration.toml" 2>/dev/null
  ncam=$(ls "$SC/clips/" | wc -l)
  echo "    $ncam cam clips staged; running MeTRAbs..."
  docker exec isr-containers-dev-1 bash -lc "
    cd /home/vscode/workspace/DATA/ot_pose_scratch
    python run_pose.py --trial $trial --clips ./clips --calib ./calib --out ./out --side $side
  " 2>&1 | sed 's/^/    /'
  if [ -f "$SC/out/biomech_${trial}.npz" ]; then
    cp "$SC/out/biomech_${trial}.npz" "$CACHE/"
    echo "    -> cached biomech_${trial}.npz"
  else
    echo "    !! no npz produced for $trial"
  fi
done < "$LIST"
echo "POSE_BATCH_DONE"
