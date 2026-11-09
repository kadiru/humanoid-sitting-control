#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
trap 'status=$?; echo "$status" > logs/support_rl_retention_01.exit' EXIT
trap 'exit 130' INT TERM
store=/data/humanoid-sitting-perception
parent="$store/support_rl_independent_train_04/final.pt"
renderer='--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false'
run() {
 local name="$1"; shift
 bash scripts/run.sh python ../scripts/train_support_rl_retention.py --output "$store/$name" \
   --parent "$parent" --device cuda:0 --kit_args="$renderer" "$@" > "logs/$name.log" 2>&1
 test -e "runs/$name" || ln -s "$store/$name" "runs/$name"
}
gate() { IsaacLab/.venv/bin/python scripts/check_support_rl_retention.py "$@"; }
run support_rl_retention_easy_parent_01 --mode evaluate --checkpoint "$parent" --bank-profile legacy \
 --num-envs 16 --geometry-count 8 --geometry-seed 51403 --seed 51407
gate parity support_rl_retention_easy_parent_01
eligible=()
for arm in control anchored; do
 coefficient=0; if [[ "$arm" == anchored ]]; then coefficient=1; fi
 run "support_rl_retention_${arm}_train_01" --mode train --iterations 32 --retention-coefficient "$coefficient" \
  --training-bank "$store/support_rl_pace_mixed_train_02/geometry_bank.json" \
  --num-envs 16 --geometry-count 16 --geometry-seed 51621 --seed 51713
 checkpoint="$store/support_rl_retention_${arm}_train_01/final.pt"
 run "support_rl_retention_easy_${arm}_01" --mode evaluate --checkpoint "$checkpoint" --bank-profile legacy \
  --num-envs 16 --geometry-count 8 --geometry-seed 51403 --seed 51407
 if gate easy "support_rl_retention_easy_${arm}_01"; then
  run "support_rl_retention_wide_${arm}_01" --mode evaluate --checkpoint "$checkpoint" \
   --num-envs 16 --geometry-count 8 --geometry-seed 51541 --seed 51543
  if gate wide "support_rl_retention_wide_${arm}_01"; then eligible+=("$arm");
  else status=$?; test "$status" = 2; fi
 else status=$?; test "$status" = 2; fi
done
if (( ${#eligible[@]} )); then
 run support_rl_retention_fresh_parent_01 --mode evaluate --checkpoint "$parent" \
  --num-envs 16 --geometry-count 8 --geometry-seed 51741 --seed 51743
 for arm in "${eligible[@]}"; do
  run "support_rl_retention_fresh_${arm}_01" --mode evaluate \
   --checkpoint "$store/support_rl_retention_${arm}_train_01/final.pt" \
   --num-envs 16 --geometry-count 8 --geometry-seed 51741 --seed 51743
 done
fi
echo 'Completed the predeclared retention gates; no deployment change.'
