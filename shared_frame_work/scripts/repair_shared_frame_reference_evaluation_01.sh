#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1
trap 'echo $? > logs/shared_frame_reference_evaluation_repair_01.exit' EXIT
while [[ ! -f logs/shared_frame_reference_validation_02.exit ]]; do sleep 10; done
for mode in clean noisy occlusion; do
  run=shared_frame_reference_validation_${mode}_01
  if [[ ! -f runs/$run/errors.json ]]; then
    bash scripts/run.sh python ../scripts/evaluate_shared_frame.py --capture ../runs/shared_frame_reference_validation_capture_02 --tracking ../runs/"$run" > logs/"${run}_evaluation_repair".log 2>&1
  fi
done
storage=/data/humanoid-sitting-perception
bash scripts/run.sh python ../scripts/audit_shared_frame_observability.py --output "$storage/shared_frame_reference_validation_audit_01" --inputs shared_frame_reference_validation_clean_01 shared_frame_reference_validation_noisy_01 shared_frame_reference_validation_occlusion_01
ln -s "$storage/shared_frame_reference_validation_audit_01" runs/shared_frame_reference_validation_audit_01
check=shared_frame_reference_evaluator_regression_01
mkdir "$storage/$check"
for file in tracks.npz estimates.json; do
  ln -s /home/kadir/humanoid-sitting/runs/shared_frame_integrity_holdout_clean_03/"$file" "$storage/$check/$file"
done
bash scripts/run.sh python ../scripts/evaluate_shared_frame.py --capture ../runs/shared_frame_integrity_holdout_capture_03 --tracking "$storage/$check" > logs/"$check".log 2>&1
ln -s "$storage/$check" runs/"$check"
bash scripts/run.sh python - <<'PY'
import json
from pathlib import Path
old=json.loads(Path('../runs/shared_frame_integrity_holdout_clean_03/errors.json').read_text())
new=json.loads(Path('../runs/shared_frame_reference_evaluator_regression_01/errors.json').read_text())
assert old==new
print('EVALUATOR_UNCHANGED_EXISTING_METRICS',len(old),flush=True)
PY
