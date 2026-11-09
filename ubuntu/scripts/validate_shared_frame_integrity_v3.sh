#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1
trap 'echo $? > logs/shared_frame_integrity_03.exit' EXIT
storage=/data/humanoid-sitting-perception
jobs=(
 'shared_frame_junction_fresh_capture_01 clean fresh_clean'
 'shared_frame_junction_fresh_capture_01 noisy fresh_noisy'
 'shared_frame_junction_fresh_capture_01 occlusion fresh_occlusion'
 'shared_frame_rear_down_six_capture_01 clean rear_clean'
 'shared_frame_rear_down_six_capture_01 noisy rear_noisy'
 'shared_frame_rear_down_six_capture_01 occlusion rear_occlusion'
 'shared_frame_capture_01 clean original_clean'
 'shared_frame_capture_01 noisy original_noisy'
 'shared_frame_capture_01 occlusion original_occlusion'
 'shared_frame_stress_capture_01 clean stress'
 'shared_frame_holdout_capture_01 clean motion'
 'shared_frame_integrity_unseen_capture_02 clean unseen_clean'
 'shared_frame_integrity_unseen_capture_02 noisy unseen_noisy'
 'shared_frame_integrity_unseen_capture_02 occlusion unseen_occlusion'
)
pids=();names=()
for job in "${jobs[@]}"; do
 read -r capture mode name <<< "$job"
 run=shared_frame_integrity_${name}_03;names+=("$run")
 (
  test ! -e runs/"$run"
  bash scripts/run.sh python ../scripts/run_shared_frame.py --capture ../runs/"$capture" --output "$storage/$run" --mode "$mode"
  ln -s "$storage/$run" runs/"$run"
  bash scripts/run.sh python ../scripts/evaluate_shared_frame.py --capture ../runs/"$capture" --tracking ../runs/"$run"
 ) > logs/"$run".log 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" == 0 ]]
bash scripts/run.sh python ../scripts/audit_shared_frame_observability.py --output "$storage/shared_frame_integrity_audit_03" --inputs "${names[@]}"
ln -s "$storage/shared_frame_integrity_audit_03" runs/shared_frame_integrity_audit_03
