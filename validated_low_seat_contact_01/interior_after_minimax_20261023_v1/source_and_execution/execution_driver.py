"""Execution-owner timer for the explicitly approved single diagnostic."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import signal
import subprocess
import time

S = Path('/home/kadir/humanoid-sitting/scripts/interior_after_minimax_20261023_v1')
O = Path('/data/humanoid-sitting-perception/low_seat_contact_01/interior_after_minimax_20261023_v1')
from common import contract
contract()  # Reject before claim or worker creation unless this exact stage is approved.
launch = json.loads((S / 'execution_launch.json').read_text())
assert hashlib.sha256((S / 'contract.json').read_bytes()).hexdigest() == launch['approved_contract_sha256']
assert not O.exists()
with (S / 'execution_owner_claim.json').open('x') as f:
    json.dump({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'pid': os.getpid(), 'single_attempt': True,
               'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, f)
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
env.pop('PYTHONOPTIMIZE', None)
timed_out = False
with (S / 'supervisor.log').open('x') as log:
    started = time.monotonic()
    p = subprocess.Popen(launch['command'], stdout=log, stderr=subprocess.STDOUT,
                         env=env, start_new_session=True)
    print(json.dumps({'event': 'launched', 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      'supervisor_pid': p.pid, 'total_cap_s': 600}), flush=True)
    try:
        p.wait(timeout=max(0, 600 - (time.monotonic() - started)))
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(timeout=2)
        except subprocess.TimeoutExpired:
            # Terminate only descendants of this launch and its own process group.
            descendants = {p.pid}
            relations = {}
            for item in Path('/proc').glob('[0-9]*/status'):
                try:
                    lines = item.read_text().splitlines()
                    relations[int(item.parent.name)] = int(next(x.split()[1] for x in lines if x.startswith('PPid:')))
                except (OSError, StopIteration, ValueError):
                    pass
            while True:
                added = {pid for pid, parent in relations.items() if parent in descendants} - descendants
                if not added:
                    break
                descendants.update(added)
            for pid in sorted(descendants, reverse=True):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            p.wait()
    child_elapsed = time.monotonic() - started
result = json.loads((O / 'result.json').read_text()) if (O / 'result.json').exists() else None
elapsed = time.monotonic() - started
accepted = bool(p.returncode == 0 and not timed_out and elapsed <= 600 and result and result['candidate_accepted'])
record = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'actual_exit': p.returncode, 'supervisor_pid': p.pid,
          'child_process_wall_s': child_elapsed, 'through_result_read_wall_s': elapsed,
          'total_cap_s': 600, 'outer_timeout': timed_out,
          'overshoot_s': max(0, elapsed - 600), 'candidate_accepted': accepted,
          'result': result, 'retries': 0, 'simulation_calls': 0, 'PPO_updates': 0,
          'whole_process_finalization_not_included_in_child_wall': True}
(S / 'execution_exit.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
print(json.dumps(record), flush=True)
