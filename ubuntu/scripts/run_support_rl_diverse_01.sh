#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1
trap 'status=$?; echo "$status" > logs/support_rl_diverse_01.exit' EXIT
while ! test -f logs/support_rl_extended_01.exit; do sleep 15; done
test "$(cat logs/support_rl_extended_01.exit)" = 0
store=/data/humanoid-sitting-perception
renderer='--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false'
bash scripts/run.sh python ../scripts/train_support_rl_diverse.py --mode train --iterations 96 \
  --output "$store/support_rl_diverse_train_01" --num-envs 16 --geometry-count 16 \
  --geometry-seed 51317 --seed 51319 --device cuda:0 --kit_args="$renderer" \
  > logs/support_rl_diverse_train_01.log 2>&1
# Previously failed bank is now a regression, not an untouched holdout.
bash scripts/run.sh python ../scripts/train_support_rl.py --mode evaluate \
  --checkpoint "$store/support_rl_diverse_train_01/final.pt" --output "$store/support_rl_diverse_regression_01" \
  --num-envs 8 --geometry-count 4 --geometry-seed 51103 --seed 51107 --device cuda:0 --kit_args="$renderer" \
  > logs/support_rl_diverse_regression_01.log 2>&1
bash scripts/run.sh python ../scripts/train_support_rl.py --mode zero \
  --output "$store/support_rl_zero_fresh_01" --num-envs 16 --geometry-count 8 \
  --geometry-seed 51403 --seed 51407 --device cuda:0 --kit_args="$renderer" \
  > logs/support_rl_zero_fresh_01.log 2>&1
bash scripts/run.sh python ../scripts/train_support_rl.py --mode evaluate \
  --checkpoint "$store/support_rl_diverse_train_01/final.pt" --output "$store/support_rl_diverse_fresh_01" \
  --num-envs 16 --geometry-count 8 --geometry-seed 51403 --seed 51407 --device cuda:0 --kit_args="$renderer" \
  > logs/support_rl_diverse_fresh_01.log 2>&1
for name in support_rl_diverse_train_01 support_rl_diverse_regression_01 support_rl_zero_fresh_01 support_rl_diverse_fresh_01; do
  test -e "runs/$name" || ln -s "$store/$name" "runs/$name"
done
