#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1
bash scripts/run.sh python ../scripts/test_shared_frame.py > logs/shared_frame_component_fix_tests.log 2>&1
jobs=(
  'shared_frame_rear_down_six_capture_01 clean shared_frame_rear_down_six_clean_02'
  'shared_frame_rear_down_six_capture_01 noisy shared_frame_rear_down_six_noisy_02'
  'shared_frame_rear_down_six_capture_01 occlusion shared_frame_rear_down_six_occlusion_02'
  'shared_frame_capture_01 clean shared_frame_quality_clean_05'
  'shared_frame_capture_01 noisy shared_frame_quality_noisy_05'
  'shared_frame_capture_01 occlusion shared_frame_quality_occlusion_04'
  'shared_frame_stress_capture_01 clean shared_frame_stress_05'
  'shared_frame_holdout_capture_01 clean shared_frame_holdout_03'
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
bash scripts/run.sh python ../scripts/analyze_shared_frame_six.py --capture shared_frame_rear_down_six_capture_01 --prefix shared_frame_rear_down_six --version 02 --original-run shared_frame_quality_clean_05 --output ../runs/shared_frame_six_validation_02 --video
bash scripts/run.sh python ../scripts/audit_shared_frame_observability.py --output ../runs/shared_frame_component_regression_01 --inputs shared_frame_quality_clean_05 shared_frame_quality_noisy_05 shared_frame_quality_occlusion_04 shared_frame_stress_05 shared_frame_holdout_03
bash scripts/run.sh python ../scripts/audit_shared_frame_integrity.py --output runs/shared_frame_six_integrity_02.json
