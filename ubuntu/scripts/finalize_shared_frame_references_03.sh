#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1
trap 'echo $? > logs/shared_frame_reference_finalize_03.exit' EXIT
for name in reference_03 reference_validation_04 reference_sensor_only_03; do
  while [[ ! -f logs/shared_frame_${name}.exit ]]; do sleep 10; done
  [[ "$(cat logs/shared_frame_${name}.exit)" == 0 ]]
done
storage=/data/humanoid-sitting-perception
bash scripts/run.sh python ../scripts/report_shared_frame_references_v3.py --validation
ln -s "$storage/shared_frame_reference_results_03" runs/shared_frame_reference_results_03
bash scripts/run.sh python ../scripts/plot_shared_frame_reference_coverage_v3.py
bash scripts/run.sh python ../scripts/visualize_shared_frame_references_03.py
ln -s "$storage/shared_frame_reference_visuals_03" runs/shared_frame_reference_visuals_03
bash scripts/run.sh python ../scripts/audit_shared_frame_integrity.py --output runs/shared_frame_reference_preservation_03.json
