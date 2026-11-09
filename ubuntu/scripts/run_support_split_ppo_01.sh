#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
trap 'status=$?; echo "$status" > logs/support_split_ppo_01.exit' EXIT
store=/data/humanoid-sitting-perception
test "$(cat logs/support_split_validation_01.exit)" = 0
IsaacLab/.venv/bin/python scripts/replay_support_split_observation.py \
 --capture "$store/support_split_regression_parent_01" \
 --output "$store/support_split_online_replay_01" > logs/support_split_online_replay_01.log 2>&1
IsaacLab/.venv/bin/python - <<'PY'
import hashlib,json,shutil
from pathlib import Path
root=Path.cwd(); store=Path('/data/humanoid-sitting-perception')
assert json.loads((store/'support_split_regression_parent_01/split_observation_score.json').read_text())['passed']
assert json.loads((store/'support_split_replay_01/results.json').read_text())['passed']
assert json.loads((store/'support_split_online_replay_01/results.json').read_text())['passed']
assert json.loads((store/'support_split_reset_probe_01/reset_probe.json').read_text())['passed']
frozen=store/'support_split_ppo_frozen_01';frozen.mkdir(exist_ok=False)
paths=[p for p in (root/'scripts').glob('*support*split*') if p.is_file()]
paths += [root/'docs/SUPPORT_SPLIT_PPO_PROTOCOL.md']
manifest=json.loads((store/'support_split_regression_parent_01/manifest.json').read_text())
paths += [root/p for p in manifest['hashes']]
hashes={}
for p in sorted(set(paths)):
 relative=str(p.relative_to(root));hashes[relative]=hashlib.sha256(p.read_bytes()).hexdigest()
 if p.suffix in ('.py','.sh','.md'):
  dest=frozen/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
(frozen/'manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
PY
run() {
 local name="$1"; shift
 bash scripts/run.sh python ../scripts/train_support_split_observation.py \
  --output "$store/$name" --parent "$store/support_rl_independent_train_04/final.pt" \
  --interface split --num-envs 16 --device cuda:0 \
  --kit_args='--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false' \
  "$@" > "logs/$name.log" 2>&1
 ln -s "$store/$name" "runs/$name"
}
run support_split_ppo_train_01 --mode train --iterations 32 \
 --bank-from "$store/support_rl_pace_mixed_train_02/geometry_bank.json" \
 --geometry-count 16 --geometry-seed 51621 --seed 52613
checkpoint="$store/support_split_ppo_train_01/final.pt"
run support_split_regression_trained_01 --mode evaluate --checkpoint "$checkpoint" --bank-profile legacy \
 --geometry-count 8 --geometry-seed 51403 --seed 51407
for policy in parent trained; do
 checkpoint_args=()
 if [[ "$policy" == trained ]]; then checkpoint_args=(--checkpoint "$checkpoint"); fi
 run "support_split_fresh_${policy}_01" --mode evaluate "${checkpoint_args[@]}" \
  --geometry-count 8 --geometry-seed 52641 --seed 52643
 run "support_split_starts_${policy}_01" --mode evaluate "${checkpoint_args[@]}" \
  --bank-from "$store/support_rl_pace_mixed_train_02/geometry_bank.json" \
  --geometry-count 16 --geometry-seed 51621 --seed 52623
done
for bank in regression fresh starts; do
 for policy in parent trained; do
  if [[ "$bank" == regression && "$policy" == parent ]]; then continue; fi
  IsaacLab/.venv/bin/python scripts/check_support_split_observation.py "$store/support_split_${bank}_${policy}_01" \
   > "logs/support_split_${bank}_${policy}_score_01.log"
 done
done
IsaacLab/.venv/bin/python scripts/report_support_split_ppo.py --store "$store" --output "$store/support_split_ppo_results_01" \
 > logs/support_split_ppo_results_01.log 2>&1
echo 'One PPO pilot and all paired evaluations completed; defaults unchanged.'
