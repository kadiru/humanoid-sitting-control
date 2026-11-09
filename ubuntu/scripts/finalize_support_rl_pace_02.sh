#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
trap 'status=$?; echo "$status" > logs/support_rl_pace_finalize_02.exit' EXIT
trap 'exit 130' INT TERM
store=/data/humanoid-sitting-perception
while [[ ! -f logs/support_rl_pace_02.exit ]]; do sleep 15; done
test "$(cat logs/support_rl_pace_02.exit)" = 0
IsaacLab/.venv/bin/python scripts/audit_shared_frame_integrity.py --output "$store/support_rl_pace_integrity_01.json" > logs/support_rl_pace_integrity_01.log 2>&1
out="$store/support_rl_pace_results_01"
IsaacLab/.venv/bin/python scripts/report_support_rl_pace.py --output "$out"
cp "$store/support_rl_pace_integrity_01.json" "$out/"
cp docs/SUPPORT_RL_PACE_PROTOCOL.md "$out/"
cp scripts/run_support_rl_pace_02.sh "$out/"
ln -s "$out" runs/support_rl_pace_results_01
