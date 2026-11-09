#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
for mode in full_fresh depth_noise position_height_error delay60 delay100 reset_memory; do
 flags=(--deployment-checkpoint ../runs/deployment_geometry_recurrent_04/student.pt --deployment-perception ../runs/deployment_perception_stress_fresh_01/clean --deployment-action-rate .05 --geometry-stage 2 --geometry-seed 24001 --geometry-count 4 --seeds 3201 --compact-recording --summary-only --device cuda:0)
 case "$mode" in
 full_fresh) flags+=(--geometry-count 16 --seeds 3201 3202 3203);;
 depth_noise) flags+=(--deployment-perception ../runs/deployment_perception_stress_fresh_01/depth_noise_5mm_dropout_10pct);;
 position_height_error) flags+=(--seat-position-error .02 --seat-height-error .01);;
 delay60) flags+=(--sensor-delay-steps 3);;
 delay100) flags+=(--sensor-delay-steps 5);;
 reset_memory) flags+=(--reset-deployment-memory);;
 esac
 bash scripts/run.sh python ../scripts/evaluate_geometry_reference.py "${flags[@]}" --output "../runs/deployment_robust_${mode}_04" > "runs/deployment_interface_01/robust_${mode}_04.log" 2>&1
 done
