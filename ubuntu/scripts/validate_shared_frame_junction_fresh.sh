#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1
capture=shared_frame_junction_fresh_capture_01
bash scripts/run.sh python ../scripts/capture_shared_frame.py --output ../runs/$capture --count 6 --seconds 16 --track-online --camera-profile rear_down --geometry-bank ../experiments/shared_frame_junction_fresh_geometry_01.json --device cuda:0 --kit_args='--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false'
pids=()
for mode in clean noisy occlusion; do
  run=shared_frame_junction_fresh_${mode}_01
  (
    bash scripts/run.sh python ../scripts/run_shared_frame.py --capture ../runs/$capture --output ../runs/$run --mode "$mode"
    bash scripts/run.sh python ../scripts/evaluate_shared_frame.py --capture ../runs/$capture --tracking ../runs/$run
  ) > logs/$run.log 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" == 0 ]]
bash scripts/run.sh python ../scripts/audit_shared_frame_observability.py --output ../runs/shared_frame_junction_fresh_audit_01 --inputs shared_frame_junction_fresh_clean_01 shared_frame_junction_fresh_noisy_01 shared_frame_junction_fresh_occlusion_01
