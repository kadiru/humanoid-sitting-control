#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1
trap 'echo $? > logs/shared_frame_integrity_holdout_03.exit' EXIT
storage=/data/humanoid-sitting-perception
capture=shared_frame_integrity_holdout_capture_03
mkdir -p "$storage"
test ! -e runs/"$capture"
bash scripts/run.sh python ../scripts/capture_shared_frame.py --output "$storage/$capture" --count 3 --seconds 16 --track-online --camera-profile rear_down --geometry-bank ../experiments/shared_frame_integrity_holdout_geometry_03.json --device cuda:0 --kit_args='--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false'
ln -s "$storage/$capture" runs/"$capture"
pids=()
for mode in clean noisy occlusion; do
 run=shared_frame_integrity_holdout_${mode}_03
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
bash scripts/run.sh python ../scripts/audit_shared_frame_observability.py --output "$storage/shared_frame_integrity_holdout_audit_03" --inputs shared_frame_integrity_holdout_clean_03 shared_frame_integrity_holdout_noisy_03 shared_frame_integrity_holdout_occlusion_03
ln -s "$storage/shared_frame_integrity_holdout_audit_03" runs/shared_frame_integrity_holdout_audit_03
