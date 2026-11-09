#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1
trap 'echo $? > logs/shared_frame_reference_finalize_01.exit' EXIT
for name in reference_01 reference_validation_02 reference_sensor_only_01; do
  while [[ ! -f logs/shared_frame_${name}.exit ]]; do sleep 10; done
  [[ "$(cat logs/shared_frame_${name}.exit)" == 0 ]]
done
storage=/data/humanoid-sitting-perception
bash scripts/run.sh python ../scripts/report_shared_frame_references.py --validation
ln -s "$storage/shared_frame_reference_results_01" runs/shared_frame_reference_results_01
bash scripts/run.sh python ../scripts/plot_shared_frame_reference_coverage.py
bash scripts/run.sh python ../scripts/visualize_shared_frame_references_01.py
ln -s "$storage/shared_frame_reference_visuals_01" runs/shared_frame_reference_visuals_01
bash scripts/run.sh python ../scripts/audit_shared_frame_integrity.py --output runs/shared_frame_reference_preservation_01.json
