#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
for name in shared_frame_quality_clean_04 shared_frame_quality_noisy_04 shared_frame_quality_occlusion_03 shared_frame_stress_04 shared_frame_holdout_02 shared_frame_quality_live_02 shared_frame_quality_sensor_only_02; do
  while [ ! -f "runs/$name.exit" ]; do sleep 3; done
  if [ "$(cat "runs/$name.exit")" != 0 ]; then
    echo "FAILED dependency: $name"
    exit 1
  fi
done
export OPENBLAS_NUM_THREADS=1
bash scripts/run.sh python ../scripts/verify_shared_frame_quality.py
bash scripts/run.sh python ../scripts/audit_shared_frame_observability.py \
  --output ../runs/shared_frame_observability_audit_final \
  --inputs shared_frame_quality_clean_04 shared_frame_quality_noisy_04 shared_frame_quality_occlusion_03 shared_frame_stress_04 shared_frame_holdout_02
bash scripts/run.sh python ../scripts/visualize_shared_frame_quality.py --output ../runs/shared_frame_quality_visuals
bash scripts/run.sh python ../scripts/audit_shared_frame_integrity.py --output runs/shared_frame_quality_integrity.json
bash scripts/run.sh python ../scripts/report_shared_frame_quality.py
