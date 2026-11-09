#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
trap 'status=$?; echo "$status" > logs/support_control_timing_01.exit' EXIT
store=/data/humanoid-sitting-perception
for scope in none startup live both; do
 name="support_control_timing_${scope}_02"
 bash scripts/run.sh python ../scripts/capture_support_control_timing.py \
  --output "$store/$name" --mode evaluate --checkpoint "$store/support_rl_independent_train_04/final.pt" \
  --num-envs 16 --device cuda:0 --sync-scope "$scope" --capture-refresh due \
  --bank-profile legacy --geometry-count 8 --geometry-seed 51403 --seed 51407 \
  --kit_args='--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false' > "logs/$name.log" 2>&1
 ln -s "$store/$name" "runs/$name"
done
