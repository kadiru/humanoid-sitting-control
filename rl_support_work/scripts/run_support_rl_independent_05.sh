#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
trap 'status=$?; echo "$status" > logs/support_rl_independent_05.exit' EXIT
store=/data/humanoid-sitting-perception
renderer='--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false'
old="$store/support_rl_diverse_train_02/final.pt"
new="$store/support_rl_independent_train_04/final.pt"
run() {
  local name="$1"; shift
  bash scripts/run.sh python ../scripts/train_support_rl_independent.py --output "$store/$name" \
    --device cuda:0 --kit_args="$renderer" "$@" > "logs/$name.log" 2>&1
  test -e "runs/$name" || ln -s "$store/$name" "runs/$name"
}
# Integration probe and tiny optimizer smoke are explicitly not research scores.
run support_rl_reset_probe_04 --mode train --iterations 2 --seconds 2 --reset-probe \
  --bank-profile legacy --num-envs 4 --geometry-count 4 --geometry-seed 51501 --seed 51503
# Fixed, previously unused broad holdout. No tuning/checkpoint selection on it.
run support_rl_independent_fresh_zero_04 --mode zero --num-envs 16 --geometry-count 8 --geometry-seed 51541 --seed 51543
run support_rl_independent_fresh_prior_04 --mode evaluate --checkpoint "$old" \
  --num-envs 16 --geometry-count 8 --geometry-seed 51541 --seed 51543
# Continue pilot 2 for exactly 98,304 additional transitions; fixed final model.
run support_rl_independent_train_04 --mode train --iterations 96 --checkpoint "$old" \
  --num-envs 16 --geometry-count 16 --geometry-seed 51511 --seed 51513
run support_rl_independent_fresh_candidate_04 --mode evaluate --checkpoint "$new" \
  --num-envs 16 --geometry-count 8 --geometry-seed 51541 --seed 51543
# Known pilot-2 fresh bank is now a regression set, using the new reset protocol.
run support_rl_independent_regression_prior_04 --mode evaluate --checkpoint "$old" \
  --bank-profile legacy --num-envs 16 --geometry-count 8 --geometry-seed 51403 --seed 51407
run support_rl_independent_regression_candidate_04 --mode evaluate --checkpoint "$new" \
  --bank-profile legacy --num-envs 16 --geometry-count 8 --geometry-seed 51403 --seed 51407
echo 'All independent-reset pilot simulations completed.'
