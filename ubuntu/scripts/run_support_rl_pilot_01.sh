#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1
store=/data/humanoid-sitting-perception
renderer='--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false'
trap 'status=$?; echo "$status" > logs/support_rl_pilot_01.exit' EXIT
bash scripts/run.sh python ../scripts/train_support_rl.py --mode train --iterations 192 \
  --output "$store/support_rl_train_01" --num-envs 8 --geometry-count 4 \
  --geometry-seed 51031 --seed 51041 --device cuda:0 --kit_args="$renderer" \
  > logs/support_rl_train_01.log 2>&1
bash scripts/run.sh python ../scripts/train_support_rl.py --mode evaluate \
  --checkpoint "$store/support_rl_train_01/final.pt" --output "$store/support_rl_eval_train_01" \
  --num-envs 8 --geometry-count 4 --geometry-seed 51031 --seed 51041 --device cuda:0 --kit_args="$renderer" \
  > logs/support_rl_eval_train_01.log 2>&1
bash scripts/run.sh python ../scripts/train_support_rl.py --mode zero \
  --output "$store/support_rl_zero_holdout_01" --num-envs 8 --geometry-count 4 \
  --geometry-seed 51103 --seed 51107 --device cuda:0 --kit_args="$renderer" \
  > logs/support_rl_zero_holdout_01.log 2>&1
bash scripts/run.sh python ../scripts/train_support_rl.py --mode evaluate \
  --checkpoint "$store/support_rl_train_01/final.pt" --output "$store/support_rl_eval_holdout_01" \
  --num-envs 8 --geometry-count 4 --geometry-seed 51103 --seed 51107 --device cuda:0 --kit_args="$renderer" \
  > logs/support_rl_eval_holdout_01.log 2>&1
for name in support_rl_train_01 support_rl_eval_train_01 support_rl_zero_train_01 support_rl_zero_holdout_01 support_rl_eval_holdout_01; do
  test -e "runs/$name" || ln -s "$store/$name" "runs/$name"
done
bash scripts/run.sh python ../scripts/audit_shared_frame_integrity.py --output runs/support_rl_preservation_01.json \
  > logs/support_rl_preservation_01.log 2>&1
