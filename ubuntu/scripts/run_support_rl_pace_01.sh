#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
trap 'status=$?; echo "$status" > logs/support_rl_pace_01.exit' EXIT
trap 'exit 130' INT TERM
store=/data/humanoid-sitting-perception
renderer='--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false'
parent="$store/support_rl_independent_train_04/final.pt"
run() {
  local name="$1"; shift
  bash scripts/run.sh python ../scripts/train_support_rl_pace.py --output "$store/$name" \
    --device cuda:0 --kit_args="$renderer" "$@" > "logs/$name.log" 2>&1
  test -e "runs/$name" || ln -s "$store/$name" "runs/$name"
}
IsaacLab/.venv/bin/python -m unittest discover -s scripts -p 'test_support_*.py' -v > logs/support_rl_pace_tests_01.log 2>&1
run support_rl_pace_probe_01 --mode train --iterations 2 --seconds 2 --reset-probe \
  --warm-start-23 "$parent" --bank-profile legacy --num-envs 4 --geometry-count 4 --geometry-seed 51601 --seed 51603
run support_rl_pace_easy_train_01 --mode train --iterations 64 --warm-start-23 "$parent" \
  --bank-profile legacy --num-envs 16 --geometry-count 16 --geometry-seed 51611 --seed 51613
run support_rl_pace_mixed_train_01 --mode train --iterations 128 --checkpoint "$store/support_rl_pace_easy_train_01/final.pt" \
  --pace-curriculum mixed --num-envs 16 --geometry-count 16 --geometry-seed 51621 --seed 51623
candidate="$store/support_rl_pace_mixed_train_01/final.pt"
for which in prior candidate; do
  if [[ "$which" == prior ]]; then checkpoint=(--warm-start-23 "$parent"); else checkpoint=(--checkpoint "$candidate"); fi
  run "support_rl_pace_easy_${which}_01" --mode evaluate "${checkpoint[@]}" --bank-profile legacy \
    --num-envs 16 --geometry-count 8 --geometry-seed 51403 --seed 51407
  run "support_rl_pace_wide_${which}_01" --mode evaluate "${checkpoint[@]}" \
    --num-envs 16 --geometry-count 8 --geometry-seed 51541 --seed 51543
  run "support_rl_pace_fresh_${which}_01" --mode evaluate "${checkpoint[@]}" \
    --num-envs 16 --geometry-count 8 --geometry-seed 51641 --seed 51643
done
echo 'Completed fixed pace-control pilot.'
