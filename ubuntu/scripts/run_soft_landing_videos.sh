#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
for controller in baseline contact; do
  extra=()
  if [[ "$controller" == contact ]]; then
    extra=(--contact-estimator ../runs/contact_estimator_02/contact.pt --soft-landing-mode actions)
  fi
  bash scripts/run.sh python ../scripts/evaluate_geometry_reference.py --deployment-checkpoint ../runs/deployment_status_mask_05/student.pt "${extra[@]}" --deployment-perception ../datasets/deploy_rgbd_confirmation_05 --deployment-action-rate .05 --compact-recording --geometry-seed 25001 --geometry-count 12 --seeds 3401 --video --video-geometry-index 4 --video-benchmark-config 0 --output "../runs/soft_landing_${controller}_video_01" --device cuda:0 --kit_args='--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false' > "logs/soft_landing_${controller}_video_01.log" 2>&1
  bash scripts/run.sh python ../scripts/plot_scripted.py "../runs/soft_landing_${controller}_video_01" > "logs/soft_landing_${controller}_video_plot_01.log" 2>&1
done
