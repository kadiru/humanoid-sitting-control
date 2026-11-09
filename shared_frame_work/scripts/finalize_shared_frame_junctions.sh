#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
for name in shared_frame_junction_validation_01 shared_frame_junction_fresh_01; do
  while [[ ! -f runs/${name}.exit ]]; do sleep 10; done
  [[ $(cat runs/${name}.exit) == 0 ]]
done
export OPENBLAS_NUM_THREADS=1
bash scripts/run.sh python ../scripts/report_shared_frame_junctions.py
bash scripts/run.sh python ../scripts/visualize_shared_frame_junctions.py
mkdir runs/shared_frame_junction_results_02/source
cp scripts/report_shared_frame_junctions.py scripts/visualize_shared_frame_junctions.py sitting_task/src/humanoid_sitting/shared_frame.py runs/shared_frame_junction_results_02/source/
bash scripts/run.sh python ../scripts/audit_shared_frame_integrity.py --output runs/shared_frame_junction_integrity_01.json
