#!/usr/bin/env bash
# Persistently finish the frozen six-surface camera experiment after capture.
set -euo pipefail
cd /home/kadir/humanoid-sitting
capture=shared_frame_rear_down_six_capture_01
while [[ ! -f runs/${capture}.exit ]]; do sleep 10; done
[[ $(cat runs/${capture}.exit) == 0 ]]
export OPENBLAS_NUM_THREADS=1
for mode in clean noisy occlusion; do
  run=shared_frame_rear_down_six_${mode}_01
  bash scripts/run.sh python ../scripts/run_shared_frame.py --capture ../runs/$capture --output ../runs/$run --mode "$mode"
  bash scripts/run.sh python ../scripts/evaluate_shared_frame.py --capture ../runs/$capture --tracking ../runs/$run
done
bash scripts/run.sh python ../scripts/analyze_shared_frame_six.py --capture "$capture" --prefix shared_frame_rear_down_six --output ../runs/shared_frame_six_validation_01 --video
bash scripts/run.sh python ../scripts/audit_shared_frame_integrity.py --output runs/shared_frame_six_integrity_01.json
