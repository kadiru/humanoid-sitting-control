#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
trap 'status=$?; echo "$status" > logs/support_tracking_control_dev_01.exit' EXIT
# Serialize simulations on the 4090. This waits for a specific local job only.
while tmux has-session -t control_timing_02 2>/dev/null; do sleep 5; done
test "$(cat logs/support_control_timing_01.exit)" = 0
store=/data/humanoid-sitting-perception
name=support_tracking_control_dev_01
bash scripts/run.sh python ../scripts/capture_support_tracking_control.py \
 --output "$store/$name" --mode evaluate --checkpoint "$store/support_rl_independent_train_04/final.pt" \
 --num-envs 16 --device cuda:0 --sync-scope both --capture-refresh due \
 --bank-profile legacy --geometry-count 8 --geometry-seed 51403 --seed 51407 \
 --kit_args='--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false' > "logs/$name.log" 2>&1
ln -s "$store/$name" "runs/$name"
IsaacLab/.venv/bin/python scripts/score_landing_shadow.py --capture "$store/$name"
IsaacLab/.venv/bin/python scripts/score_support_landing.py --tracking "$store/$name/shadow_score" > "logs/${name}_score.log"
