"""Read saved checkpoints to verify fresh GPU0 updates and GPU1's first-update audit."""
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

import torch

D = Path('/data/humanoid-sitting-perception/low_seat_contact_01')
O = D / 'contact_parallel_gpu_v1'
M = D / 'contact_reward_comparison_128_v4'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    baseline = read(O / 'verification_baseline_v2.json')
    folder = Path(baseline['gpu0']['folder'])
    current = read(folder / 'progress.json')['updates']
    assert current > baseline['gpu0']['updates']
    before_path = Path(baseline['gpu0']['checkpoint'])
    assert sha(before_path) == baseline['gpu0']['sha256']
    after_path = folder / f'model_{current:03d}.pt'
    before = torch.load(before_path, map_location='cpu', weights_only=False)
    after = torch.load(after_path, map_location='cpu', weights_only=False)
    records = [r for r in read(folder / 'learning_audit.json')
               if before['updates'] < r['updates'] <= after['updates']]
    assert len(records) == after['updates'] - before['updates'] > 0
    models = {}
    for name in ['actor', 'critic']:
        old, new = before[name + '_state_dict'], after[name + '_state_dict']
        assert old.keys() == new.keys()
        assert all(torch.isfinite(v).all() for v in new.values())
        changes = {k: float((new[k] - old[k]).abs().max()) for k in old}
        assert max(changes.values()) > 0
        expected = sum(r['optimizer'][name + '_steps'] for r in records)
        old_opt = before[name + '_optimizer']['state']
        new_opt = after[name + '_optimizer']['state']
        assert expected > 0 and old_opt.keys() == new_opt.keys()
        assert all(int(new_opt[k]['step']) - int(old_opt[k]['step']) == expected
                   for k in old_opt)
        assert all(torch.isfinite(v).all() for state in new_opt.values()
                   for v in state.values() if torch.is_tensor(v))
        models[name] = {'max_abs_parameter_change': max(changes.values()),
                        'changed_tensors': sum(v > 0 for v in changes.values()),
                        'new_adam_steps_per_parameter': expected, 'finite': True}
    status = read(O / 'parallel_status.json')
    assert status['state'] == 'running'
    verified = next(x for x in status['first_updates_verified'] if x['gpu'] == 1)
    gpu1_folder = O / verified['phase']
    weight_audit_path = gpu1_folder / (Path(verified['checkpoint']).stem + '_weight_audit.json')
    weight_audit = read(weight_audit_path)
    assert weight_audit['passed'] and weight_audit['local_updates'] > 0
    assert weight_audit['checkpoint_sha256'] == sha(gpu1_folder / verified['checkpoint'])
    assert weight_audit['initial_sha256'] == sha(gpu1_folder / 'initial.pt')
    assert all(weight_audit['models'][n]['max_abs_parameter_change'] > 0 and
               weight_audit['models'][n]['adam_steps_per_parameter'] > 0 and
               weight_audit['models'][n]['finite'] for n in ['actor', 'critic'])
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                    '--format=csv,noheader'], text=True)
    groups = {int(x.split()[0]): int(x.split()[1]) for x in
              subprocess.check_output(['ps', '-eo', 'pid=,pgid='], text=True).splitlines()}
    legacy = read(M / 'status.json')
    assert legacy['state'] == 'running' and legacy['phase'] == folder.name
    expected = {legacy['child_pid']: 'GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4',
                status['active']['1']['pid']: 'GPU-cde23916-c5fa-f18e-a343-127f732cc2f6'}
    observed = {}
    for line in apps.splitlines():
        uuid, pid, _ = [s.strip() for s in line.split(',')]
        pgid = groups.get(int(pid))
        if pgid in expected:
            assert uuid == expected[pgid]
            observed[pgid] = {'gpu_uuid': uuid, 'python_pid': int(pid)}
    assert observed.keys() == expected.keys()
    result = {
        'passed': True, 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'gpu0': {'seed': 63117, 'arm': 'original', 'before_updates': before['updates'],
                 'after_updates': after['updates'], 'models': models,
                 'before_checkpoint': str(before_path), 'before_sha256': sha(before_path),
                 'after_checkpoint': str(after_path), 'after_sha256': sha(after_path)},
        'gpu1': {'seed': 63111, 'arm': 'revised', 'before_updates': 64,
                 'verified_updates': 64 + weight_audit['local_updates'],
                 'current_updates': read(gpu1_folder / 'progress.json')['updates'],
                 'models': weight_audit['models'], 'weight_audit': str(weight_audit_path)},
        'live_gpu_bindings': observed, 'compute_apps': apps,
        'gpu_telemetry': subprocess.check_output(['nvidia-smi',
            '--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,power.draw',
            '--format=csv,noheader'], text=True),
        'parallel_owner_pid': status['owner_pid'], 'legacy_handoff_complete': False,
        'current_gpu0_chunk_untouched': True, 'action_limits_changed': False,
        'physical_success_criteria_changed': False, 'physical_repair_claimed': False,
    }
    with (O / 'dual_gpu_weight_verification_v2.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
