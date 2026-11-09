#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
for condition in clean noisy; do
  if [[ "$condition" == clean ]]; then perception=../datasets/soft_landing_rgbd_fresh_01; else perception=../runs/soft_landing_perception_fresh_01/depth_noise_5mm_dropout_10pct; fi
  bash scripts/run.sh python ../scripts/evaluate_geometry_reference.py --deployment-checkpoint ../runs/deployment_status_mask_05/student.pt --contact-estimator ../runs/contact_estimator_conservative_05/contact.pt --soft-landing-mode actions --deployment-perception "$perception" --deployment-action-rate .05 --compact-recording --geometry-seed 26001 --geometry-count 8 --seeds 3501 3502 --output "../runs/soft_landing_conservative_fresh_${condition}_02" --device cuda:0 > "logs/soft_landing_conservative_fresh_${condition}_02.log" 2>&1
done
