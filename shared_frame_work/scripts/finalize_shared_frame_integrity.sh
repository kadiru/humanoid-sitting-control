#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1
trap 'echo $? > logs/shared_frame_integrity_final_02.exit' EXIT
storage=/data/humanoid-sitting-perception
for name in shared_frame_integrity_02 shared_frame_integrity_remaining_02 shared_frame_integrity_unseen_02; do
 while [[ ! -f logs/${name}.exit ]]; do sleep 10; done
 [[ $(cat logs/${name}.exit) == 0 ]]
done
bash scripts/run.sh python ../scripts/report_shared_frame_integrity.py
ln -s "$storage/shared_frame_integrity_results_02" runs/shared_frame_integrity_results_02
bash scripts/run.sh python ../scripts/visualize_shared_frame_integrity.py
ln -s "$storage/shared_frame_integrity_visuals_02" runs/shared_frame_integrity_visuals_02
bash scripts/run.sh python ../scripts/audit_shared_frame_integrity.py --output runs/shared_frame_integrity_preservation_02.json
