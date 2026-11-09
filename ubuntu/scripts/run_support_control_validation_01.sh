#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
trap 'status=$?; echo "$status" > logs/support_control_validation_01.exit' EXIT
store=/data/humanoid-sitting-perception
run() {
 local kind="$1"; local bank="$2"; shift 2
 local name="support_control_validation_${bank}_${kind}_01"
 local script=capture_support_control_timing.py
 if [[ "$kind" == candidate ]]; then script=capture_support_continuous_control.py; fi
 bash scripts/run.sh python "../scripts/$script" \
  --output "$store/$name" --mode evaluate --checkpoint "$store/support_rl_independent_train_04/final.pt" \
  --num-envs 16 --device cuda:0 --sync-scope both --capture-refresh due \
  --kit_args='--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false' \
  "$@" > "logs/$name.log" 2>&1
 ln -s "$store/$name" "runs/$name"
 if [[ "$kind" == candidate ]]; then
  IsaacLab/.venv/bin/python scripts/score_landing_shadow.py --capture "$store/$name"
  IsaacLab/.venv/bin/python scripts/score_support_landing.py --tracking "$store/$name/shadow_score" > "logs/${name}_score.log"
  IsaacLab/.venv/bin/python scripts/score_support_continuous_features.py --capture "$store/$name" > "logs/${name}_patch_score.log"
 fi
}
for kind in baseline candidate; do
 run "$kind" fresh --geometry-count 8 --geometry-seed 51941 --seed 51943
done
for kind in baseline candidate; do
 run "$kind" starts --bank-from "$store/support_rl_pace_mixed_train_02/geometry_bank.json" \
  --geometry-count 16 --geometry-seed 51621 --seed 51923
done
