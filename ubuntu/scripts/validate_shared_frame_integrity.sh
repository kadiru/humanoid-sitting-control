#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1
version=${1:?unique output suffix required}
trap 'echo $? > logs/shared_frame_integrity_'"$version"'.exit' EXIT
jobs=(
 'shared_frame_junction_fresh_capture_01 clean fresh_clean'
 'shared_frame_junction_fresh_capture_01 noisy fresh_noisy'
 'shared_frame_junction_fresh_capture_01 occlusion fresh_occlusion'
 'shared_frame_rear_down_six_capture_01 clean rear_clean'
)
pids=()
for job in "${jobs[@]}"; do
 read -r capture mode name <<< "$job"
 run=shared_frame_integrity_${name}_${version}
 (
  bash scripts/run.sh python ../scripts/run_shared_frame.py --capture ../runs/"$capture" --output ../runs/"$run" --mode "$mode"
  bash scripts/run.sh python ../scripts/evaluate_shared_frame.py --capture ../runs/"$capture" --tracking ../runs/"$run"
 ) > logs/"$run".log 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" == 0 ]]
names=(); for job in "${jobs[@]}"; do read -r capture mode name <<< "$job"; names+=(shared_frame_integrity_${name}_${version}); done
bash scripts/run.sh python ../scripts/audit_shared_frame_observability.py --output ../runs/shared_frame_integrity_audit_"$version" --inputs "${names[@]}"
