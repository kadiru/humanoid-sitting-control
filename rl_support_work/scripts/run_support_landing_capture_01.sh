#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
trap 'status=$?; echo "$status" > logs/support_landing_capture_01.exit' EXIT
trap 'exit 130' INT TERM
store=/data/humanoid-sitting-perception
bash scripts/run.sh python ../scripts/capture_support_landing.py \
 --output "$store/support_landing_development_capture_01" --mode evaluate \
 --checkpoint "$store/support_rl_independent_train_04/final.pt" \
 --bank-from "$store/support_rl_pace_mixed_train_02/geometry_bank.json" \
 --num-envs 16 --geometry-count 16 --geometry-seed 51621 --seed 51813 \
 --device cuda:0 --kit_args='--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false' \
 > logs/support_landing_development_capture_01.log 2>&1
ln -s "$store/support_landing_development_capture_01" runs/support_landing_development_capture_01
