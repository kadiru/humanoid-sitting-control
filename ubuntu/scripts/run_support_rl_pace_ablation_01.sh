#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
trap 'status=$?; echo "$status" > logs/support_rl_pace_ablation_01.exit' EXIT
trap 'exit 130' INT TERM
while [[ ! -f logs/support_rl_pace_02.exit ]]; do sleep 15; done
test "$(cat logs/support_rl_pace_02.exit)" = 0
store=/data/humanoid-sitting-perception
name=support_rl_pace_easy_neutral_01
bash scripts/run.sh python ../scripts/evaluate_support_rl_pace_neutral.py --output "$store/$name" \
 --mode evaluate --checkpoint "$store/support_rl_pace_mixed_train_02/final.pt" --bank-profile legacy \
 --num-envs 16 --geometry-count 8 --geometry-seed 51403 --seed 51407 --device cuda:0 \
 --kit_args='--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false' > "logs/$name.log" 2>&1
ln -s "$store/$name" "runs/$name"
