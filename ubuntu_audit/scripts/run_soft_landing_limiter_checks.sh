#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
for condition in confirmation_noise fresh_clean fresh_noise novel; do
 case "$condition" in
 confirmation_noise) perception=../runs/deployment_perception_stress_confirmation_05/depth_noise_5mm_dropout_10pct; bank=25001; count=12; seeds=(3401); extra=();;
 fresh_clean) perception=../datasets/soft_landing_rgbd_fresh_01; bank=26001; count=8; seeds=(3501 3502); extra=();;
 fresh_noise) perception=../runs/soft_landing_perception_fresh_01/depth_noise_5mm_dropout_10pct; bank=26001; count=8; seeds=(3501 3502); extra=();;
 novel) perception=../datasets/deploy_rgbd_novel_01; bank=24001; count=4; seeds=(3201); extra=(--novel-shapes);;
 esac
 bash scripts/run.sh python ../scripts/evaluate_geometry_reference.py --deployment-checkpoint ../runs/deployment_status_mask_05/student.pt --contact-estimator ../runs/contact_estimator_02/contact.pt --soft-landing-mode actions --disable-contact-feedback --deployment-perception "$perception" --deployment-action-rate .05 --summary-only --compact-recording --geometry-seed "$bank" --geometry-count "$count" --seeds "${seeds[@]}" "${extra[@]}" --output "../runs/soft_landing_limiter_${condition}_01" --device cuda:0 > "logs/soft_landing_limiter_${condition}_01.log" 2>&1
done
