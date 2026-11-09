#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
for condition in clean noisy; do
  if [[ "$condition" == clean ]]; then
    perception=../datasets/soft_landing_rgbd_fresh_01
  else
    perception=../runs/soft_landing_perception_fresh_01/depth_noise_5mm_dropout_10pct
  fi
  bash scripts/run.sh python ../scripts/evaluate_geometry_reference.py --deployment-checkpoint ../runs/deployment_status_mask_05/student.pt --deployment-perception "$perception" --deployment-action-rate .05 --summary-only --compact-recording --geometry-seed 26001 --geometry-count 8 --seeds 3501 3502 --output "../runs/soft_landing_baseline_fresh_${condition}_01" --device cuda:0 > "logs/soft_landing_baseline_fresh_${condition}_01.log" 2>&1
done
bash scripts/run.sh python ../scripts/evaluate_geometry_reference.py --deployment-checkpoint ../runs/deployment_status_mask_05/student.pt --contact-estimator ../runs/contact_estimator_window_03/contact.pt --soft-landing-mode actions --deployment-perception ../datasets/deploy_rgbd_confirmation_05 --deployment-action-rate .05 --compact-recording --geometry-seed 25001 --geometry-count 12 --seeds 3401 --output ../runs/soft_landing_window_clean_03 --device cuda:0 > logs/soft_landing_window_clean_03.log 2>&1
