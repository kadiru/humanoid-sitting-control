#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1
trap 'status=$?; echo "$status" > logs/support_rl_ablation_01.exit' EXIT
while ! test -f logs/support_rl_pilot_01.exit; do sleep 15; done
test "$(cat logs/support_rl_pilot_01.exit)" = 0
bash scripts/run.sh python ../scripts/evaluate_support_rl_without_live_depth.py \
  --mode evaluate --checkpoint /data/humanoid-sitting-perception/support_rl_train_01/final.pt \
  --output /data/humanoid-sitting-perception/support_rl_no_live_depth_01 \
  --num-envs 8 --geometry-count 4 --geometry-seed 51103 --seed 51107 --device cuda:0 \
  --kit_args='--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false' \
  > logs/support_rl_no_live_depth_01.log 2>&1
test -e runs/support_rl_no_live_depth_01 || ln -s /data/humanoid-sitting-perception/support_rl_no_live_depth_01 runs/support_rl_no_live_depth_01
