#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
trap 'status=$?; echo "$status" > logs/support_rl_retention_finalize_01.exit' EXIT
trap 'exit 130' INT TERM
# Bounded wait for the companion persistent simulation process.
for ((poll=0; poll<240; poll++)); do
 if [[ -f logs/support_rl_retention_01.exit ]]; then break; fi
 sleep 15
done
test "$(cat logs/support_rl_retention_01.exit)" = 0
IsaacLab/.venv/bin/python scripts/report_support_rl_retention.py > logs/support_rl_retention_report_01.log 2>&1
IsaacLab/.venv/bin/python scripts/render_support_rl_retention.py > logs/support_rl_retention_video_01.log 2>&1
out=/data/humanoid-sitting-perception/support_rl_retention_results_01
test -e runs/support_rl_retention_results_01 || ln -s "$out" runs/support_rl_retention_results_01
