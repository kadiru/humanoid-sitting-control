#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
trap 'status=$?; echo "$status" > logs/support_rl_pace_finalize_03.exit' EXIT
trap 'exit 130' INT TERM
test "$(cat logs/support_rl_pace_02.exit)" = 0
store=/data/humanoid-sitting-perception
# The old checker verified preservation, then rejected root's arbitrary 30 GiB
# reserve. Keep that failure visible. All new large outputs are on /data.
python3 - <<'PY'
import json,shutil
from pathlib import Path
s=Path('/data/humanoid-sitting-perception')
r=json.loads((s/'support_rl_pace_integrity_01.json').read_text())
assert not r['original_source_changed'] and not r['archived_baseline_changed']
assert not r['current_checkpoint_changed'] and r['isaaclab_status']==''
check=dict(preservation_checks_passed=True,legacy_root_reserve_passed=r['free_gib']>30,
    root_free_gib=shutil.disk_usage('/home/kadir/humanoid-sitting').free/2**30,
    output_volume=str(s),output_free_gib=shutil.disk_usage(s).free/2**30,
    reason='Legacy checker stopped only at its root reserve. Report/video are explicitly written to /data; no existing files were removed.')
assert check['output_free_gib']>30
(s/'support_rl_pace_storage_02.json').write_text(json.dumps(check,indent=2)+'\n')
print(json.dumps(check),flush=True)
PY
out="$store/support_rl_pace_results_01"
IsaacLab/.venv/bin/python scripts/report_support_rl_pace.py --output "$out"
cp "$store/support_rl_pace_integrity_01.json" "$store/support_rl_pace_storage_02.json" "$out/"
cp docs/SUPPORT_RL_PACE_PROTOCOL.md docs/SUPPORT_RL_PACE_PROTOCOL_ORIGINAL.md "$out/"
cp scripts/run_support_rl_pace_02.sh "$out/"
ln -s "$out" runs/support_rl_pace_results_01
while [[ ! -f logs/support_rl_pace_ablation_01.exit ]]; do sleep 15; done
test "$(cat logs/support_rl_pace_ablation_01.exit)" = 0
IsaacLab/.venv/bin/python scripts/report_support_rl_pace_ablation.py
