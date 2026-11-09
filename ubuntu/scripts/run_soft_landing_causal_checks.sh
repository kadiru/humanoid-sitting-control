#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
for intervention in height delay; do
  extra=()
  if [[ "$intervention" == height ]]; then extra=(--seat-height-error .01); else extra=(--sensor-delay-steps 3); fi
  bash scripts/run.sh python ../scripts/evaluate_geometry_reference.py --deployment-checkpoint ../runs/deployment_status_mask_05/student.pt --contact-estimator ../runs/contact_estimator_02/contact.pt --soft-landing-mode actions --deployment-perception ../datasets/deploy_rgbd_confirmation_05 --deployment-action-rate .05 --summary-only --compact-recording --geometry-seed 25001 --geometry-count 12 --seeds 3401 "${extra[@]}" --output "../runs/soft_landing_causal_${intervention}_02" --device cuda:0 > "logs/soft_landing_causal_${intervention}_02.log" 2>&1
done
