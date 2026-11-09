#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1
bash scripts/run.sh python ../scripts/test_shared_frame.py > logs/shared_frame_junction_tests_01.log 2>&1
jobs=(
  'shared_frame_rear_down_six_capture_01 clean shared_frame_junction_rear_clean_01'
  'shared_frame_rear_down_six_capture_01 noisy shared_frame_junction_rear_noisy_01'
  'shared_frame_rear_down_six_capture_01 occlusion shared_frame_junction_rear_occlusion_01'
  'shared_frame_capture_01 clean shared_frame_junction_original_clean_01'
  'shared_frame_capture_01 noisy shared_frame_junction_original_noisy_01'
  'shared_frame_capture_01 occlusion shared_frame_junction_original_occlusion_01'
  'shared_frame_stress_capture_01 clean shared_frame_junction_stress_01'
  'shared_frame_holdout_capture_01 clean shared_frame_junction_motion_01'
)
pids=()
for job in "${jobs[@]}"; do
  read -r capture mode run <<< "$job"
  (
    bash scripts/run.sh python ../scripts/run_shared_frame.py --capture ../runs/"$capture" --output ../runs/"$run" --mode "$mode"
    bash scripts/run.sh python ../scripts/evaluate_shared_frame.py --capture ../runs/"$capture" --tracking ../runs/"$run"
  ) > logs/"$run".log 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" == 0 ]]
bash scripts/run.sh python ../scripts/analyze_shared_frame_six.py --capture shared_frame_rear_down_six_capture_01 --prefix shared_frame_junction_rear --original-run shared_frame_junction_original_clean_01 --output ../runs/shared_frame_junction_six_validation_01
bash scripts/run.sh python ../scripts/audit_shared_frame_observability.py --output ../runs/shared_frame_junction_regression_01 --inputs shared_frame_junction_original_clean_01 shared_frame_junction_original_noisy_01 shared_frame_junction_original_occlusion_01 shared_frame_junction_stress_01 shared_frame_junction_motion_01
