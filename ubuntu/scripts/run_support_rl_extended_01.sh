#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1
trap 'status=$?; echo "$status" > logs/support_rl_extended_01.exit' EXIT
while ! test -f logs/support_rl_ablation_01.exit; do sleep 15; done
test "$(cat logs/support_rl_ablation_01.exit)" = 0
store=/data/humanoid-sitting-perception
renderer='--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false'
bash scripts/run.sh python ../scripts/train_support_rl.py --mode zero \
  --output "$store/support_rl_zero_extended_01" --num-envs 24 --geometry-count 12 \
  --geometry-seed 51203 --seed 51207 --device cuda:0 --kit_args="$renderer" \
  > logs/support_rl_zero_extended_01.log 2>&1
bash scripts/run.sh python ../scripts/train_support_rl.py --mode evaluate \
  --checkpoint "$store/support_rl_train_01/final.pt" --output "$store/support_rl_eval_extended_01" \
  --num-envs 24 --geometry-count 12 --geometry-seed 51203 --seed 51207 --device cuda:0 --kit_args="$renderer" \
  > logs/support_rl_eval_extended_01.log 2>&1
for name in support_rl_zero_extended_01 support_rl_eval_extended_01; do
  test -e "runs/$name" || ln -s "$store/$name" "runs/$name"
done
