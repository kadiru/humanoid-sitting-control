#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
for mode in feedback geometry; do
 flag=--freeze-deployment-feedback
 [[ "$mode" == geometry ]] && flag=--canonical-deployment-geometry
 bash scripts/run.sh python ../scripts/evaluate_geometry_reference.py --deployment-checkpoint ../runs/deployment_geometry_recurrent_04/student.pt --deployment-perception ../runs/deployment_perception_stress_fresh_01/clean --deployment-action-rate .05 --geometry-stage 2 --geometry-seed 24001 --geometry-count 4 --seeds 3201 --compact-recording --summary-only "$flag" --output "../runs/deployment_ablated_${mode}_04" --device cuda:0 > "runs/deployment_interface_01/ablated_${mode}_04.log" 2>&1
 done
