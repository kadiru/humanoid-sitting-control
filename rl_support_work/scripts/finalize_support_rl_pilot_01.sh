#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1
trap 'status=$?; echo "$status" > logs/support_rl_finalize_01.exit' EXIT
while ! test -f logs/support_rl_pilot_01.exit; do sleep 15; done
test "$(cat logs/support_rl_pilot_01.exit)" = 0
bash scripts/run.sh python ../scripts/report_support_rl.py \
  --output /data/humanoid-sitting-perception/support_rl_results_01 \
  > logs/support_rl_results_01.log 2>&1
test -e runs/support_rl_results_01 || ln -s /data/humanoid-sitting-perception/support_rl_results_01 runs/support_rl_results_01
