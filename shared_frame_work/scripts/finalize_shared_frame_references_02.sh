#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1
trap 'echo $? > logs/shared_frame_reference_finalize_02.exit' EXIT
for name in reference_02 reference_validation_03 reference_sensor_only_02; do
  while [[ ! -f logs/shared_frame_${name}.exit ]]; do sleep 10; done
  [[ "$(cat logs/shared_frame_${name}.exit)" == 0 ]]
done
storage=/data/humanoid-sitting-perception
bash scripts/run.sh python ../scripts/report_shared_frame_references_v2.py --validation
ln -s "$storage/shared_frame_reference_results_02" runs/shared_frame_reference_results_02
bash scripts/run.sh python ../scripts/plot_shared_frame_reference_coverage_v2.py
bash scripts/run.sh python ../scripts/visualize_shared_frame_references_02.py
ln -s "$storage/shared_frame_reference_visuals_02" runs/shared_frame_reference_visuals_02
bash scripts/run.sh python ../scripts/audit_shared_frame_integrity.py --output runs/shared_frame_reference_preservation_02.json
