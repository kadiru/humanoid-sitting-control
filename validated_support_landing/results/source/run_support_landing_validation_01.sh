#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
trap 'status=$?; echo "$status" > logs/support_landing_validation_01.exit' EXIT
trap 'exit 130' INT TERM
store=/data/humanoid-sitting-perception
run() {
 local name="$1"; shift
 bash scripts/run.sh python ../scripts/capture_support_landing_shadow.py \
  --output "$store/$name" --mode evaluate --checkpoint "$store/support_rl_independent_train_04/final.pt" \
  --num-envs 16 --device cuda:0 --capture-refresh due \
  --kit_args='--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false' \
  "$@" > "logs/$name.log" 2>&1
 ln -s "$store/$name" "runs/$name"
 IsaacLab/.venv/bin/python scripts/score_landing_shadow.py --capture "$store/$name"
 IsaacLab/.venv/bin/python scripts/score_support_landing.py --tracking "$store/$name/shadow_score" > "logs/${name}_score.log"
}
run support_landing_validation_capture_01 --bank-from "$store/support_rl_pace_mixed_train_02/geometry_bank.json" \
 --geometry-count 16 --geometry-seed 51621 --seed 51823
run support_landing_regression_capture_01 --bank-profile legacy --geometry-count 8 --geometry-seed 51403 --seed 51407
run support_landing_fresh_capture_01 --geometry-count 8 --geometry-seed 51841 --seed 51843

for fault in none noise blackout; do
 track="$store/support_landing_fresh_replay_${fault}_01"
 IsaacLab/.venv/bin/python scripts/replay_support_landing.py --capture "$store/support_landing_fresh_capture_01" --output "$track" --fault "$fault" > "logs/support_landing_fresh_replay_${fault}_01.log" 2>&1
 IsaacLab/.venv/bin/python scripts/score_support_landing.py --tracking "$track" > "logs/support_landing_fresh_score_${fault}_01.log" 2>&1
done
