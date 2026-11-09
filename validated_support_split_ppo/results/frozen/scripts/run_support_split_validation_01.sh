#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
trap 'status=$?; echo "$status" > logs/support_split_validation_01.exit' EXIT
store=/data/humanoid-sitting-perception
IsaacLab/.venv/bin/python scripts/test_support_split_observation.py > logs/support_split_interface_tests_02.log 2>&1
IsaacLab/.venv/bin/python scripts/replay_support_split_observation.py \
 --capture "$store/support_control_validation_fresh_candidate_01" \
 --output "$store/support_split_replay_01" > logs/support_split_replay_01.log 2>&1
run() {
 local name="$1"; shift
 bash scripts/run.sh python ../scripts/train_support_split_observation.py \
  --output "$store/$name" --parent "$store/support_rl_independent_train_04/final.pt" \
  --interface split --device cuda:0 \
  --kit_args='--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false' \
  "$@" > "logs/$name.log" 2>&1
 ln -s "$store/$name" "runs/$name"
}
run support_split_regression_parent_01 --mode evaluate --bank-profile legacy \
 --num-envs 16 --geometry-count 8 --geometry-seed 51403 --seed 51407
IsaacLab/.venv/bin/python scripts/check_support_split_observation.py "$store/support_split_regression_parent_01" \
 --reference "$store/support_continuous_control_dev_01" --gate > logs/support_split_regression_gate_01.log
run support_split_reset_probe_01 --mode evaluate --reset-probe --seconds 2 \
 --num-envs 4 --geometry-count 4 --geometry-seed 52601 --seed 52603
test -f "$store/support_split_reset_probe_01/reset_probe.json"
echo 'Observation interface validated; no PPO has run in this launcher.'
