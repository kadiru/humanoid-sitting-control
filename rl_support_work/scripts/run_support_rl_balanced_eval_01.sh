#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1
trap 'status=$?; echo "$status" > logs/support_rl_balanced_eval_01.exit' EXIT
while ! test -f logs/support_rl_diverse_01.exit; do sleep 15; done
test "$(cat logs/support_rl_diverse_01.exit)" = 0
store=/data/humanoid-sitting-perception
renderer='--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false'
bash scripts/run.sh python ../scripts/evaluate_support_rl_bank.py --mode zero \
  --bank-from "$store/support_rl_diverse_train_01/geometry_bank.json" \
  --output "$store/support_rl_zero_balanced_01" --num-envs 16 --geometry-count 16 \
  --geometry-seed 51317 --seed 51319 --device cuda:0 --kit_args="$renderer" \
  > logs/support_rl_zero_balanced_01.log 2>&1
bash scripts/run.sh python ../scripts/evaluate_support_rl_bank.py --mode evaluate \
  --bank-from "$store/support_rl_diverse_train_01/geometry_bank.json" \
  --checkpoint "$store/support_rl_diverse_train_01/final.pt" \
  --output "$store/support_rl_eval_balanced_01" --num-envs 16 --geometry-count 16 \
  --geometry-seed 51317 --seed 51319 --device cuda:0 --kit_args="$renderer" \
  > logs/support_rl_eval_balanced_01.log 2>&1
for name in support_rl_zero_balanced_01 support_rl_eval_balanced_01; do
  test -e "runs/$name" || ln -s "$store/$name" "runs/$name"
done
